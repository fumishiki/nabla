// autograd.rs — Tape-based reverse-mode automatic differentiation.
//
// Design:
//   - Rc-based, single-threaded (Variable is !Send + !Sync via PhantomData).
//   - TapeEntry records the backward closure and accumulates gradient in-place.
//   - Leaf variables (created via Tape::variable) have tape_entry = None;
//     their gradient is stored in a dedicated Rc<RefCell<Option<Tensor<T,B>>>>.
//   - No Rc cycles: backward closures capture Weak<TapeEntry> for inputs so
//     the only strong Rc<TapeEntry> chain goes forward through the tape's Vec.

#![allow(clippy::missing_errors_doc)]

use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::{Rc, Weak};

use nabla_core::backend::Backend;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

// ---------------------------------------------------------------------------
// Internal: gradient accumulator shared by leaf variables.
// ---------------------------------------------------------------------------

/// Shared gradient slot for a leaf variable.
type GradSlot<T, B> = Rc<RefCell<Option<Tensor<T, B>>>>;

/// Weak reference to a leaf gradient slot, used inside backward closures.
type WeakSlot<T, B> = Weak<RefCell<Option<Tensor<T, B>>>>;

/// Backward propagation closure type.
type BackwardFn<T, B> = Box<dyn Fn(&Tensor<T, B>)>;

/// Propagate `delta` into a raw `Weak<RefCell<Option<Tensor>>>` slot.
fn accum_weak_slot<T: Scalar, B: Backend>(
    slot_weak: &Weak<RefCell<Option<Tensor<T, B>>>>,
    delta: &Tensor<T, B>,
) {
    if let Some(slot) = slot_weak.upgrade() {
        let mut borrow = slot.borrow_mut();
        *borrow = Some(match borrow.take() {
            None => delta.clone(),
            Some(existing) => &existing + delta,
        });
    }
}

// ---------------------------------------------------------------------------
// TapeEntry
// ---------------------------------------------------------------------------

/// One node in the computation graph recorded during the forward pass.
struct TapeEntry<T: Scalar, B: Backend> {
    /// Gradient accumulated for this node's output tensor.
    grad: RefCell<Option<Tensor<T, B>>>,
    /// Propagates `out_grad` upstream to inputs.
    backward: BackwardFn<T, B>,
}

impl<T: Scalar, B: Backend> TapeEntry<T, B> {
    fn new(backward: impl Fn(&Tensor<T, B>) + 'static) -> Rc<Self> {
        Rc::new(Self {
            grad: RefCell::new(None),
            backward: Box::new(backward),
        })
    }

    /// Accumulate `delta` into this entry's gradient.
    fn accum(&self, delta: &Tensor<T, B>) {
        let mut g = self.grad.borrow_mut();
        *g = Some(match g.take() {
            None => delta.clone(),
            Some(existing) => &existing + delta,
        });
    }
}

// ---------------------------------------------------------------------------
// Tape
// ---------------------------------------------------------------------------

/// Computation tape recording forward operations for reverse-mode AD.
///
/// Create leaf [`Variable`]s via [`Tape::variable`], perform operations on
/// them, then call [`Variable::backward`] on the scalar output.  Read leaf
/// gradients with [`Variable::grad`].
pub struct Tape<T: Scalar, B: Backend> {
    entries: RefCell<Vec<Rc<TapeEntry<T, B>>>>,
}

impl<T: Scalar, B: Backend> Tape<T, B> {
    /// Create a new empty tape.
    #[must_use]
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            entries: RefCell::new(Vec::new()),
        })
    }

    /// Record `entry` on the tape and return the same `Rc`.
    fn push(self: &Rc<Self>, entry: Rc<TapeEntry<T, B>>) -> Rc<TapeEntry<T, B>> {
        self.entries.borrow_mut().push(Rc::clone(&entry));
        entry
    }

    /// Create a tracked leaf variable on this tape.
    pub fn variable(self: &Rc<Self>, data: Tensor<T, B>) -> Variable<T, B> {
        let grad_slot: GradSlot<T, B> = Rc::new(RefCell::new(None));
        Variable {
            data,
            tape_entry: None,
            grad_slot: Some(Rc::clone(&grad_slot)),
            tape: Rc::clone(self),
            _not_send: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Variable
// ---------------------------------------------------------------------------

/// A tensor tracked on a tape for automatic differentiation.
///
/// Leaf variables are created via [`Tape::variable`] and store gradients in
/// an internal slot accessible via [`Variable::grad`].
///
/// Derived variables (results of operations) do not have a `grad_slot`; read
/// their gradient from the leaf variables that they depend on.
pub struct Variable<T: Scalar, B: Backend> {
    data: Tensor<T, B>,
    /// `None` for leaf variables; `Some` for derived variables.
    tape_entry: Option<Rc<TapeEntry<T, B>>>,
    /// `Some` only for leaf variables.
    grad_slot: Option<GradSlot<T, B>>,
    tape: Rc<Tape<T, B>>,
    /// Makes Variable !Send + !Sync (Rc already is, but be explicit).
    _not_send: PhantomData<*const ()>,
}

impl<T: Scalar, B: Backend> Variable<T, B> {
    /// Access the underlying tensor data.
    #[must_use]
    pub fn data(&self) -> &Tensor<T, B> {
        &self.data
    }

    /// Gradient accumulated after [`Variable::backward`].
    ///
    /// Returns `None` for non-leaf variables or before `backward` is called.
    #[must_use]
    pub fn grad(&self) -> Option<Tensor<T, B>> {
        self.grad_slot.as_ref().and_then(|s| s.borrow().clone())
    }

    // -----------------------------------------------------------------------
    // Construction helpers
    // -----------------------------------------------------------------------

    /// Build a derived Variable, pushing its entry onto the tape.
    fn derived(tape: &Rc<Tape<T, B>>, data: Tensor<T, B>, entry: Rc<TapeEntry<T, B>>) -> Self {
        let entry = tape.push(entry);
        Self {
            data,
            tape_entry: Some(entry),
            grad_slot: None,
            tape: Rc::clone(tape),
            _not_send: PhantomData,
        }
    }

    /// Weak reference to this variable's `TapeEntry` (for backward closures).
    fn entry_weak(&self) -> Option<Weak<TapeEntry<T, B>>> {
        self.tape_entry.as_ref().map(Rc::downgrade)
    }

    /// Weak reference to the leaf gradient slot (for backward closures).
    fn slot_weak(&self) -> Option<WeakSlot<T, B>> {
        self.grad_slot.as_ref().map(Rc::downgrade)
    }

    // -----------------------------------------------------------------------
    // Upstream gradient propagation
    // -----------------------------------------------------------------------

    /// Route `delta` to either the `TapeEntry` accumulator or the leaf slot.
    fn propagate(
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

    // -----------------------------------------------------------------------
    // Forward operations (each records its backward closure)
    // -----------------------------------------------------------------------

    /// Element-wise addition.
    ///
    /// backward: `grad_a += out_grad`, `grad_b += out_grad`.
    #[must_use]
    pub fn add_var(&self, rhs: &Self) -> Self {
        let out = &self.data + &rhs.data;
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let (be, bs) = (rhs.entry_weak(), rhs.slot_weak());
        let entry = TapeEntry::new(move |g| {
            Self::propagate(ae.as_ref(), as_.as_ref(), g);
            Self::propagate(be.as_ref(), bs.as_ref(), g);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise subtraction.
    ///
    /// backward: `grad_a += out_grad`, `grad_b -= out_grad`.
    #[must_use]
    pub fn sub_var(&self, rhs: &Self) -> Self {
        let out = &self.data - &rhs.data;
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let (be, bs) = (rhs.entry_weak(), rhs.slot_weak());
        let entry = TapeEntry::new(move |g| {
            Self::propagate(ae.as_ref(), as_.as_ref(), g);
            let ng = -g;
            Self::propagate(be.as_ref(), bs.as_ref(), &ng);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Negation.
    ///
    /// backward: `grad_a -= out_grad`.
    #[must_use]
    pub fn neg_var(&self) -> Self {
        let out = -&self.data;
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let entry = TapeEntry::new(move |g| {
            let ng = -g;
            Self::propagate(ae.as_ref(), as_.as_ref(), &ng);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise multiplication (Hadamard product).
    ///
    /// backward: `grad_a += out_grad ∘ b`, `grad_b += out_grad ∘ a`.
    #[must_use]
    pub fn emul(&self, rhs: &Self) -> Self {
        let out = self.data.emul(&rhs.data);
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let (be, bs) = (rhs.entry_weak(), rhs.slot_weak());
        let b_data = rhs.data.clone();
        let a_data = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            let da = g.emul(&b_data);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
            let db = g.emul(&a_data);
            Self::propagate(be.as_ref(), bs.as_ref(), &db);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Matrix multiply (`self @ rhs`).
    ///
    /// backward: `grad_a += out_grad @ rhs^T`, `grad_b += self^T @ out_grad`.
    #[must_use]
    pub fn matmul(&self, rhs: &Self) -> Self {
        let out = &self.data * &rhs.data;
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let (be, bs) = (rhs.entry_weak(), rhs.slot_weak());
        let b_t = rhs.data.t();
        let a_t = self.data.t();
        let entry = TapeEntry::new(move |g| {
            let da = g * &b_t;
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
            let db = &a_t * g;
            Self::propagate(be.as_ref(), bs.as_ref(), &db);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Scalar multiply `self * s`.
    ///
    /// backward: `grad_a += out_grad * s`.
    #[must_use]
    pub fn scale(&self, s: T) -> Self {
        let out = &self.data * s;
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let entry = TapeEntry::new(move |g| {
            let da = g * s;
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `exp(x)`.
    ///
    /// backward: `grad_a += out_grad * exp(a)`.
    #[must_use]
    pub fn exp(&self) -> Self {
        let out = self.data.exp();
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let exp_a = out.clone();
        let entry = TapeEntry::new(move |g| {
            let da = g.emul(&exp_a);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `ln(x)`.
    ///
    /// backward: `grad_a += out_grad / a`.
    #[must_use]
    pub fn ln(&self) -> Self {
        let out = self.data.ln();
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let a_data = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            let da = g.ediv(&a_data);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `sin(x)`.
    ///
    /// backward: `grad_a += out_grad * cos(a)`.
    #[must_use]
    pub fn sin(&self) -> Self {
        let out = self.data.sin();
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let cos_a = self.data.cos();
        let entry = TapeEntry::new(move |g| {
            let da = g.emul(&cos_a);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `cos(x)`.
    ///
    /// backward: `grad_a += out_grad * (-sin(a))`.
    #[must_use]
    pub fn cos(&self) -> Self {
        let out = self.data.cos();
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let neg_sin_a = -&self.data.sin();
        let entry = TapeEntry::new(move |g| {
            let da = g.emul(&neg_sin_a);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `tanh(x)`.
    ///
    /// backward: `grad_a += out_grad * (1 - tanh(a)^2)`.
    #[must_use]
    pub fn tanh(&self) -> Self {
        let out = self.data.tanh();
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        // sech²(a) = 1 - tanh²(a), computed element-wise.
        let one = T::one_impl();
        let two = one + one;
        let (nrows, ncols) = out.shape();
        let sech2 = Tensor::from_fn(nrows, ncols, |i, j| one - out.get(i, j).math_powf(two));
        let entry = TapeEntry::new(move |g| {
            let da = g.emul(&sech2);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `sqrt(x)`.
    ///
    /// backward: `grad_a += out_grad / (2 * sqrt(a))`.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        let out = self.data.sqrt();
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        // 2 * sqrt(a) = 2 * out
        let two = T::one_impl() + T::one_impl();
        let two_sqrt_a = &out * two;
        let entry = TapeEntry::new(move |g| {
            let da = g.ediv(&two_sqrt_a);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `a^p` for scalar `p`.
    ///
    /// backward: `grad_a += out_grad * p * a^(p-1)`.
    #[must_use]
    pub fn powf(&self, p: T) -> Self {
        let out = self.data.powf(p);
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        // p * a^(p-1) stored as a tensor coefficient.
        let one = T::one_impl();
        let p_minus_1 = p - one;
        let a_pm1 = self.data.powf(p_minus_1);
        let (nrows, ncols) = a_pm1.shape();
        let coeff = Tensor::from_fn(nrows, ncols, |i, j| a_pm1.get(i, j).math_mul(p));
        let entry = TapeEntry::new(move |g| {
            let da = g.emul(&coeff);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Sum all elements → scalar Variable of shape `(1, 1)`.
    ///
    /// backward: broadcast `out_grad[0,0]` to fill the input shape.
    #[must_use]
    pub fn sum_all_var(&self) -> Self {
        let s = self.data.sum_all();
        let (nrows, ncols) = self.data.shape();
        let out = Tensor::fill(1, 1, s);
        let (ae, as_) = (self.entry_weak(), self.slot_weak());
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let da = Tensor::fill(nrows, ncols, g_val);
            Self::propagate(ae.as_ref(), as_.as_ref(), &da);
        });
        Self::derived(&self.tape, out, entry)
    }

    // -----------------------------------------------------------------------
    // Backward pass
    // -----------------------------------------------------------------------

    /// Run reverse-mode AD from this variable.
    ///
    /// Seeds this variable's gradient with `ones(shape)`, then walks the tape
    /// in reverse order, calling each recorded backward closure.
    ///
    /// # Errors
    ///
    /// Always succeeds; the `Result` return allows callers to use `?`.
    pub fn backward(&self) -> Result<()> {
        let (nrows, ncols) = self.data.shape();
        let seed = Tensor::fill(nrows, ncols, T::one_impl());

        // Deposit seed into this variable's accumulator.
        if let Some(entry) = &self.tape_entry {
            entry.accum(&seed);
        } else if let Some(slot) = &self.grad_slot {
            let mut borrow = slot.borrow_mut();
            *borrow = Some(match borrow.take() {
                None => seed.clone(),
                Some(existing) => &existing + &seed,
            });
        }

        // Reverse topological walk: entries were pushed in forward order.
        let entries = self.tape.entries.borrow();
        for entry in entries.iter().rev() {
            let g_opt = entry.grad.borrow().clone();
            if let Some(g) = g_opt {
                (entry.backward)(&g);
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Operator overloads
// ---------------------------------------------------------------------------

impl<T: Scalar, B: Backend> Add for &Variable<T, B> {
    type Output = Variable<T, B>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_var(rhs)
    }
}

impl<T: Scalar, B: Backend> Sub for &Variable<T, B> {
    type Output = Variable<T, B>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_var(rhs)
    }
}

impl<T: Scalar, B: Backend> Neg for &Variable<T, B> {
    type Output = Variable<T, B>;

    fn neg(self) -> Self::Output {
        self.neg_var()
    }
}

/// Matrix multiply (`&a * &b` ≡ `a @ b`).
impl<T: Scalar, B: Backend> Mul for &Variable<T, B> {
    type Output = Variable<T, B>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.matmul(rhs)
    }
}

/// Scalar multiply (`&variable * scalar`).
impl<T: Scalar, B: Backend> Mul<T> for &Variable<T, B> {
    type Output = Variable<T, B>;

    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

// ---------------------------------------------------------------------------
// Prep-based gradient API (DifferentiationInterface.jl pattern)
// ---------------------------------------------------------------------------

/// Preparation handle for amortized gradient computation.
///
/// Records the input shape at prep time; subsequent [`gradient`] calls assert
/// shape consistency.  Each call still builds a fresh tape (dynamic graph) —
/// the main benefit is API alignment with the DifferentiationInterface spec and
/// shape validation on reuse.
#[cfg(feature = "cpu")]
pub struct GradPrep<T: Scalar> {
    /// Expected input shape `(rows, cols)`.
    pub input_shape: (usize, usize),
    _phantom: PhantomData<T>,
}

/// One-time prep: records input shape for validation on subsequent [`gradient`] calls.
#[cfg(feature = "cpu")]
pub fn gradient_prep<T, F>(_f: &F, x: &Tensor<T, nabla_core::backend::Cpu>) -> GradPrep<T>
where
    T: Scalar,
    F: Fn(&Variable<T, nabla_core::backend::Cpu>) -> Variable<T, nabla_core::backend::Cpu>,
{
    GradPrep {
        input_shape: x.shape(),
        _phantom: PhantomData,
    }
}

/// Compute gradient of scalar-valued `f` at `x`, reusing `prep` for shape validation.
///
/// Returns `None` if backward produces no gradient (should not happen for leaf variables).
#[cfg(feature = "cpu")]
pub fn gradient<T, F>(
    f: &F,
    x: &Tensor<T, nabla_core::backend::Cpu>,
    prep: &GradPrep<T>,
) -> Option<Tensor<T, nabla_core::backend::Cpu>>
where
    T: Scalar,
    F: Fn(&Variable<T, nabla_core::backend::Cpu>) -> Variable<T, nabla_core::backend::Cpu>,
{
    assert_eq!(
        x.shape(),
        prep.input_shape,
        "nabla::gradient: input shape {:?} != prep shape {:?}",
        x.shape(),
        prep.input_shape
    );
    let tape = Tape::new();
    let x_var = tape.variable(x.clone());
    let y_var = f(&x_var);
    let _ = y_var.backward();
    x_var.grad()
}

/// Single-use gradient: compute `nabla f(x)` without creating a [`GradPrep`].
///
/// Builds a fresh tape, evaluates `f`, runs backward, and returns the gradient.
/// For repeated calls with the same `f`, prefer [`gradient_prep`] + [`gradient`].
#[cfg(feature = "cpu")]
pub fn grad<T, F>(
    f: F,
    x: &Tensor<T, nabla_core::backend::Cpu>,
) -> Option<Tensor<T, nabla_core::backend::Cpu>>
where
    T: Scalar,
    F: Fn(&Variable<T, nabla_core::backend::Cpu>) -> Variable<T, nabla_core::backend::Cpu>,
{
    let tape = Tape::new();
    let x_var = tape.variable(x.clone());
    let y_var = f(&x_var);
    let _ = y_var.backward();
    x_var.grad()
}
