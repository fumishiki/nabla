
use std::cell::{Cell, RefCell};
use std::fmt;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};

use nabla_core::backend::Backend;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;


type GradSlot<T, B> = Rc<RefCell<Option<Tensor<T, B>>>>;

type WeakSlot<T, B> = Weak<RefCell<Option<Tensor<T, B>>>>;

type BackwardFn<T, B> = Box<dyn Fn(&Tensor<T, B>)>;

pub(crate) fn accum_cell<T: Scalar, B: Backend>(
    cell: &RefCell<Option<Tensor<T, B>>>,
    delta: &Tensor<T, B>,
) {
    let mut borrow = cell.borrow_mut();
    *borrow = Some(match borrow.take() {
        None => delta.clone(),
        Some(existing) => &existing + delta,
    });
}

fn accum_weak_slot<T: Scalar, B: Backend>(
    slot_weak: &Weak<RefCell<Option<Tensor<T, B>>>>,
    delta: &Tensor<T, B>,
) {
    if let Some(slot) = slot_weak.upgrade() {
        accum_cell(&slot, delta);
    }
}


pub(crate) struct TapeEntry<T: Scalar, B: Backend> {
    /// Gradient accumulated for this node's output tensor.
    pub(super) grad: RefCell<Option<Tensor<T, B>>>,
    /// Propagates `out_grad` upstream to inputs.
    pub(super) backward: BackwardFn<T, B>,
    /// Tape indices of input entries (for reachability in backward).
    pub(super) deps: Vec<usize>,
    /// Name of the operation that created this entry (for Debug display).
    pub(super) op_name: &'static str,
}

impl<T: Scalar, B: Backend> TapeEntry<T, B> {
    pub(crate) fn new(
        backward: impl Fn(&Tensor<T, B>) + 'static,
        deps: Vec<usize>,
        op_name: &'static str,
    ) -> Rc<Self> {
        Rc::new(Self {
            grad: RefCell::new(None),
            backward: Box::new(backward),
            deps,
            op_name,
        })
    }

    /// Accumulate `delta` into this entry's gradient.
    pub(super) fn accum(&self, delta: &Tensor<T, B>) {
        accum_cell(&self.grad, delta);
    }
}


/// Reverse-mode automatic differentiation tape.
pub struct Tape<T: Scalar, B: Backend> {
    pub(super) entries: RefCell<Vec<Rc<TapeEntry<T, B>>>>,
    /// When `true`, calls to `variable()` will panic (no-grad scope).
    no_grad_active: std::cell::Cell<bool>,
}

impl<T: Scalar, B: Backend> Tape<T, B> {
    /// Create a new empty tape.
    #[must_use]
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            entries: RefCell::new(Vec::new()),
            no_grad_active: std::cell::Cell::new(false),
        })
    }

    /// Record `entry` on the tape and return the same `Rc`.
    fn push(self: &Rc<Self>, entry: Rc<TapeEntry<T, B>>) -> Rc<TapeEntry<T, B>> {
        self.entries.borrow_mut().push(Rc::clone(&entry));
        entry
    }

    /// Create a tracked leaf variable on this tape.
    ///
    /// # Errors
    ///
    /// Returns `Err` if called inside a [`Tape::no_grad`] scope.
    pub fn variable(self: &Rc<Self>, data: Tensor<T, B>) -> Result<Variable<T, B>> {
        if self.no_grad_active.get() {
            return Err(nabla_core::error::Error::invalid("variable() called inside no_grad scope"));
        }
        let grad_slot: GradSlot<T, B> = Rc::new(RefCell::new(None));
        Ok(Variable {
            data: Rc::new(data),
            tape_entry: None,
            grad_slot: Some(Rc::clone(&grad_slot)),
            retained: Cell::new(false),
            tape: Rc::clone(self),
            entry_idx: None,
            _not_send: PhantomData,
        })
    }

    /// Alias for [`Tape::variable`].
    pub fn var(self: &Rc<Self>, data: Tensor<T, B>) -> Result<Variable<T, B>> {
        self.variable(data)
    }

    /// Create Variables from a slice of parameter tensors.
    ///
    /// Convenience method for wrapping module parameters as tracked leaves.
    pub fn track_params(self: &Rc<Self>, params: &[&Tensor<T, B>]) -> Result<Vec<Variable<T, B>>> {
        params.iter().map(|p| self.variable((*p).clone())).collect()
    }

    /// Execute `f` without recording operations on this tape.
    ///
    /// Operations inside `f` that create new `Variable`s via a **different** tape
    /// (or none at all) will not be tracked.  The simplest usage is to compute
    /// with raw `Tensor`s inside `f` and return regular (non-tracked) values.
    ///
    /// This is a convenience wrapper — it temporarily sets an internal flag so
    /// that any `variable()` call on *this* tape during `f` panics, preventing
    /// accidental tracking.
    pub fn no_grad<F, R>(self: &Rc<Self>, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        struct Guard<'a>(&'a std::cell::Cell<bool>);
        impl Drop for Guard<'_> {
            fn drop(&mut self) { self.0.set(false); }
        }
        self.no_grad_active.set(true);
        let _guard = Guard(&self.no_grad_active);
        f()
    }
}


/// A tensor tracked on an autodiff tape.
pub struct Variable<T: Scalar, B: Backend> {
    pub(super) data: Rc<Tensor<T, B>>,
    /// `None` for leaf variables; `Some` for derived variables.
    pub(super) tape_entry: Option<Rc<TapeEntry<T, B>>>,
    /// `Some` only for leaf variables.
    pub(super) grad_slot: Option<GradSlot<T, B>>,
    /// When true, `grad()` reads from `tape_entry.grad` after backward.
    retained: Cell<bool>,
    pub(super) tape: Rc<Tape<T, B>>,
    /// Index of this variable's entry in the tape (None for leaves).
    pub(super) entry_idx: Option<usize>,
    /// Makes Variable !Send + !Sync (Rc already is, but be explicit).
    pub(super) _not_send: PhantomData<*const ()>,
}

impl<T: Scalar, B: Backend> fmt::Debug for Variable<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, c) = self.data.shape();
        match &self.tape_entry {
            None => write!(f, "Variable({r}x{c}, leaf)"),
            Some(entry) => write!(f, "Variable({r}x{c}, grad_fn={})", entry.op_name),
        }
    }
}

impl<T: Scalar, B: Backend> Variable<T, B> {
    /// Access the underlying tensor data.
    #[must_use]
    pub fn data(&self) -> &Tensor<T, B> {
        &self.data
    }

    /// Gradient accumulated after [`Variable::backward`].
    ///
    /// Returns the leaf gradient, or the retained gradient if [`Variable::retain_grad`]
    /// was called on an intermediate variable.
    /// Returns `Err` if no gradient is available.
    pub fn grad(&self) -> Result<Tensor<T, B>> {
        if let Some(g) = self.grad_slot.as_ref().and_then(|s| s.borrow().clone()) {
            return Ok(g);
        }
        if self.retained.get()
            && let Some(entry) = &self.tape_entry
            && let Some(g) = entry.grad.borrow().clone()
        {
            return Ok(g);
        }
        Err(nabla_core::error::Error::NoGradient)
    }

    /// Borrow the gradient without cloning, returning `Ref<Tensor<T, B>>`.
    pub fn grad_ref(&self) -> Result<std::cell::Ref<'_, Tensor<T, B>>> {
        if let Some(slot) = &self.grad_slot {
            let borrow = slot.borrow();
            if borrow.is_some() {
                return Ok(std::cell::Ref::map(borrow, |opt| {
                    // SAFETY: guarded by is_some() check above
                    opt.as_ref().unwrap_or_else(|| unreachable!())
                }));
            }
        }
        if self.retained.get()
            && let Some(entry) = &self.tape_entry
        {
            let borrow = entry.grad.borrow();
            if borrow.is_some() {
                return Ok(std::cell::Ref::map(borrow, |opt| {
                    // SAFETY: guarded by is_some() check above
                    opt.as_ref().unwrap_or_else(|| unreachable!())
                }));
            }
        }
        Err(nabla_core::error::Error::NoGradient)
    }

    /// Mark this intermediate variable to retain its gradient after backward.
    ///
    /// By default only leaf variables store gradients. Call this on an
    /// intermediate variable so that [`Variable::grad`] returns its gradient after
    /// [`Variable::backward`] completes.
    pub fn retain_grad(&self) {
        self.retained.set(true);
    }

    // -----------------------------------------------------------------------
    // Construction helpers
    // -----------------------------------------------------------------------

    /// Build a derived Variable, pushing its entry onto the tape.
    pub(crate) fn derived(
        tape: &Rc<Tape<T, B>>,
        data: Tensor<T, B>,
        entry: Rc<TapeEntry<T, B>>,
    ) -> Self {
        let idx = tape.entries.borrow().len();
        let entry = tape.push(entry);
        Self {
            data: Rc::new(data),
            tape_entry: Some(entry),
            grad_slot: None,
            retained: Cell::new(false),
            tape: Rc::clone(tape),
            entry_idx: Some(idx),
            _not_send: PhantomData,
        }
    }

    /// Collect dep indices from input variables for reachability tracking.
    pub(crate) fn deps_of(inputs: &[Option<usize>]) -> Vec<usize> {
        inputs.iter().copied().flatten().collect()
    }

    /// Weak reference to this variable's `TapeEntry` (for backward closures).
    fn entry_weak(&self) -> Option<Weak<TapeEntry<T, B>>> {
        self.tape_entry.as_ref().map(Rc::downgrade)
    }

    /// Weak reference to the leaf gradient slot (for backward closures).
    fn slot_weak(&self) -> Option<WeakSlot<T, B>> {
        self.grad_slot.as_ref().map(Rc::downgrade)
    }

    /// Capture weak refs for use in backward closures.
    pub(crate) fn input_refs(&self) -> (Option<Weak<TapeEntry<T, B>>>, Option<WeakSlot<T, B>>) {
        (self.entry_weak(), self.slot_weak())
    }

    // -----------------------------------------------------------------------
    // Upstream gradient propagation
    // -----------------------------------------------------------------------

    /// Route `delta` to either the `TapeEntry` accumulator or the leaf slot.
    pub(crate) fn propagate(
        entry: Option<&Weak<TapeEntry<T, B>>>,
        slot: Option<&WeakSlot<T, B>>,
        delta: &Tensor<T, B>,
    ) {
        if let Some(w) = entry
            && let Some(e) = w.upgrade()
        {
            e.accum(delta);
            return;
        }
        if let Some(w) = slot {
            accum_weak_slot(w, delta);
        }
    }

    /// Shorthand: propagate using a (entry, slot) tuple.
    #[inline]
    pub(crate) fn prop(
        refs: &(Option<Weak<TapeEntry<T, B>>>, Option<WeakSlot<T, B>>),
        delta: &Tensor<T, B>,
    ) {
        Self::propagate(refs.0.as_ref(), refs.1.as_ref(), delta);
    }

    // -----------------------------------------------------------------------
    // Forward operations (each records its backward closure)
    // -----------------------------------------------------------------------

    /// Element-wise addition.
    #[must_use]
    pub fn add_var(&self, rhs: &Self) -> Self {
        let out = &*self.data + &*rhs.data;
        let deps = Self::deps_of(&[self.entry_idx, rhs.entry_idx]);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, g);
            Self::prop(&rr, g);
        }, deps, "add");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise subtraction.
    #[must_use]
    pub fn sub_var(&self, rhs: &Self) -> Self {
        let out = &*self.data - &*rhs.data;
        let deps = Self::deps_of(&[self.entry_idx, rhs.entry_idx]);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, g);
            Self::prop(&rr, &(-g));
        }, deps, "sub");
        Self::derived(&self.tape, out, entry)
    }

    /// Negation.
    #[must_use]
    pub fn neg_var(&self) -> Self {
        let out = -&*self.data;
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &(-g));
        }, deps, "neg");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise multiplication (Hadamard product).
    #[must_use]
    pub fn emul(&self, rhs: &Self) -> Self {
        let out = (*self.data).emul(&*rhs.data);
        let deps = Self::deps_of(&[self.entry_idx, rhs.entry_idx]);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let (a_data, b_data) = (Rc::clone(&self.data), Rc::clone(&rhs.data));
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&*b_data));
            Self::prop(&rr, &g.emul(&*a_data));
        }, deps, "emul");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise division.
    #[must_use]
    pub fn ediv(&self, rhs: &Self) -> Self {
        let out = (*self.data).ediv(&*rhs.data);
        let deps = Self::deps_of(&[self.entry_idx, rhs.entry_idx]);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let (a_data, b_data) = (Rc::clone(&self.data), Rc::clone(&rhs.data));
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.ediv(&*b_data));
            let b_sq = (*b_data).emul(&*b_data);
            Self::prop(&rr, &g.emul(&(-&*a_data).ediv(&b_sq)));
        }, deps, "ediv");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise power `a^b`.
    #[must_use]
    pub fn epow(&self, rhs: &Self) -> Self {
        let out = (*self.data).epow(&*rhs.data);
        let deps = Self::deps_of(&[self.entry_idx, rhs.entry_idx]);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let (a_data, b_data) = (Rc::clone(&self.data), Rc::clone(&rhs.data));
        let result = out.clone();
        let entry = TapeEntry::new(move |g| {
            let one = T::one_impl();
            let (m, n) = a_data.shape();
            let a_pow_bm1 = Tensor::from_fn(m, n, |r, c| {
                a_data.get(r, c).math_powf(b_data.get(r, c) - one)
            });
            Self::prop(&lr, &g.emul(&b_data.emul(&a_pow_bm1)));
            Self::prop(&rr, &g.emul(&result.emul(&a_data.ln())));
        }, deps, "epow");
        Self::derived(&self.tape, out, entry)
    }

    /// Matrix multiply (`self @ rhs`).
    #[must_use]
    pub fn matmul(&self, rhs: &Self) -> Self {
        let out = &*self.data * &*rhs.data;
        let deps = Self::deps_of(&[self.entry_idx, rhs.entry_idx]);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let (a_data, b_data) = (Rc::clone(&self.data), Rc::clone(&rhs.data));
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.matmul_nt(&*b_data));
            Self::prop(&rr, &(*a_data).matmul_tn(g));
        }, deps, "matmul");
        Self::derived(&self.tape, out, entry)
    }

    /// Scalar multiply `self * s`.
    #[must_use]
    pub fn scale(&self, s: T) -> Self {
        let out = &*self.data * s;
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &(g * s));
        }, deps, "scale");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `exp(x)`.
    #[must_use]
    pub fn exp(&self) -> Self {
        let out = self.data.exp();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let exp_a = out.clone();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&exp_a));
        }, deps, "exp");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `ln(x)`.
    #[must_use]
    pub fn ln(&self) -> Self {
        let out = self.data.ln();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let a_data = Rc::clone(&self.data);
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.ediv(&*a_data));
        }, deps, "ln");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `sin(x)`.
    #[must_use]
    pub fn sin(&self) -> Self {
        let out = self.data.sin();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let cos_a = self.data.cos();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&cos_a));
        }, deps, "sin");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `cos(x)`.
    #[must_use]
    pub fn cos(&self) -> Self {
        let out = self.data.cos();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let neg_sin_a = -&(*self.data).sin();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&neg_sin_a));
        }, deps, "cos");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `tanh(x)`.
    #[must_use]
    pub fn tanh(&self) -> Self {
        let out = self.data.tanh();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let (nrows, ncols) = out.shape();
        let ones = Tensor::fill(nrows, ncols, T::one_impl());
        let sech2 = &ones - &out.emul(&out);
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&sech2));
        }, deps, "tanh");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `sqrt(x)`.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        let out = self.data.sqrt();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let two = T::one_impl() + T::one_impl();
        let two_sqrt_a = &out * two;
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.ediv(&two_sqrt_a));
        }, deps, "sqrt");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `a^p` for scalar `p`.
    #[must_use]
    pub fn powf(&self, p: T) -> Self {
        let out = self.data.powf(p);
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let one = T::one_impl();
        let coeff = &(*self.data).powf(p - one) * p;
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&coeff));
        }, deps, "powf");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise ReLU: `max(x, 0)`.
    #[must_use]
    pub fn relu(&self) -> Self {
        let out = self.data.relu();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.relu_backward(&*input));
        }, deps, "relu");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        let out = self.data.sigmoid();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let sig_out = out.clone();
        let entry = TapeEntry::new(move |g| {
            let (m, n) = sig_out.shape();
            let ones = Tensor::fill(m, n, T::one_impl());
            let dsig = sig_out.emul(&(&ones - &sig_out));
            Self::prop(&lr, &g.emul(&dsig));
        }, deps, "sigmoid");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise GELU.
    #[must_use]
    pub fn gelu(&self) -> Self {
        let out = self.data.gelu();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.gelu_backward(&*input));
        }, deps, "gelu");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise absolute value.
    #[must_use]
    pub fn abs(&self) -> Self {
        let out = self.data.abs();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.abs_backward(&*input));
        }, deps, "abs");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `ln(1 + x)`.
    #[must_use]
    pub fn log1p(&self) -> Self {
        let out = self.data.log1p();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(move |g| {
            let (m, n) = input.shape();
            let ones = Tensor::fill(m, n, T::one_impl());
            Self::prop(&lr, &g.ediv(&(&ones + &*input)));
        }, deps, "log1p");
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise SiLU (Swish): `x * sigmoid(x)`.
    #[must_use]
    pub fn silu(&self) -> Self {
        let out = self.data.silu();
        let deps = Self::deps_of(&[self.entry_idx]);
        let lr = self.input_refs();
        let input = Rc::clone(&self.data);
        let entry = TapeEntry::new(move |g| {
            let sig = input.sigmoid();
            let (m, n) = input.shape();
            let ones = Tensor::fill(m, n, T::one_impl());
            let dsilu = sig.emul(&(&ones + &input.emul(&(&ones - &sig))));
            Self::prop(&lr, &g.emul(&dsilu));
        }, deps, "silu");
        Self::derived(&self.tape, out, entry)
    }
}
