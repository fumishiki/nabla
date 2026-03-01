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
use std::ops::{Add, Div, Mul, Neg, Sub};
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

/// Accumulate `delta` into a shared gradient cell.
fn accum_cell<T: Scalar, B: Backend>(cell: &RefCell<Option<Tensor<T, B>>>, delta: &Tensor<T, B>) {
    let mut borrow = cell.borrow_mut();
    *borrow = Some(match borrow.take() {
        None => delta.clone(),
        Some(existing) => &existing + delta,
    });
}

/// Propagate `delta` into a raw `Weak<RefCell<Option<Tensor>>>` slot.
fn accum_weak_slot<T: Scalar, B: Backend>(
    slot_weak: &Weak<RefCell<Option<Tensor<T, B>>>>,
    delta: &Tensor<T, B>,
) {
    if let Some(slot) = slot_weak.upgrade() {
        accum_cell(&slot, delta);
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
        accum_cell(&self.grad, delta);
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
    /// # Panics
    ///
    /// Panics if called inside a [`Tape::no_grad`] scope.
    pub fn variable(self: &Rc<Self>, data: Tensor<T, B>) -> Variable<T, B> {
        assert!(
            !self.no_grad_active.get(),
            "nabla: Tape::variable() called inside a no_grad scope"
        );
        let grad_slot: GradSlot<T, B> = Rc::new(RefCell::new(None));
        Variable {
            data,
            tape_entry: None,
            grad_slot: Some(Rc::clone(&grad_slot)),
            tape: Rc::clone(self),
            _not_send: PhantomData,
        }
    }

    /// Alias for [`Tape::variable`].
    pub fn var(self: &Rc<Self>, data: Tensor<T, B>) -> Variable<T, B> {
        self.variable(data)
    }

    /// Create Variables from a slice of parameter tensors.
    ///
    /// Convenience method for wrapping module parameters as tracked leaves.
    pub fn track_params(self: &Rc<Self>, params: &[&Tensor<T, B>]) -> Vec<Variable<T, B>> {
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
        self.no_grad_active.set(true);
        let result = f();
        self.no_grad_active.set(false);
        result
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

    /// Capture weak refs for use in backward closures.
    fn input_refs(&self) -> (Option<Weak<TapeEntry<T, B>>>, Option<WeakSlot<T, B>>) {
        (self.entry_weak(), self.slot_weak())
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

    /// Shorthand: propagate using a (entry, slot) tuple.
    #[inline]
    fn prop(refs: &(Option<Weak<TapeEntry<T, B>>>, Option<WeakSlot<T, B>>), delta: &Tensor<T, B>) {
        Self::propagate(refs.0.as_ref(), refs.1.as_ref(), delta);
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
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, g);
            Self::prop(&rr, g);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise subtraction.
    ///
    /// backward: `grad_a += out_grad`, `grad_b -= out_grad`.
    #[must_use]
    pub fn sub_var(&self, rhs: &Self) -> Self {
        let out = &self.data - &rhs.data;
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, g);
            let ng = -g;
            Self::prop(&rr, &ng);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Negation.
    ///
    /// backward: `grad_a -= out_grad`.
    #[must_use]
    pub fn neg_var(&self) -> Self {
        let out = -&self.data;
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            let ng = -g;
            Self::prop(&lr, &ng);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise multiplication (Hadamard product).
    ///
    /// backward: `grad_a += out_grad ∘ b`, `grad_b += out_grad ∘ a`.
    #[must_use]
    pub fn emul(&self, rhs: &Self) -> Self {
        let out = self.data.emul(&rhs.data);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let (a_data, b_data) = (self.data.clone(), rhs.data.clone());
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&b_data));
            Self::prop(&rr, &g.emul(&a_data));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise division.
    ///
    /// backward: `grad_a += out_grad / b`, `grad_b += out_grad * (-a / b²)`.
    #[must_use]
    pub fn ediv(&self, rhs: &Self) -> Self {
        let out = self.data.ediv(&rhs.data);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let (a_data, b_data) = (self.data.clone(), rhs.data.clone());
        let entry = TapeEntry::new(move |g| {
            // d/da (a/b) = 1/b
            Self::prop(&lr, &g.ediv(&b_data));
            // d/db (a/b) = -a/b²
            let neg_a = -&a_data;
            let b_sq = b_data.emul(&b_data);
            Self::prop(&rr, &g.emul(&neg_a.ediv(&b_sq)));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise power `a^b`.
    ///
    /// backward: `grad_a += out_grad * b * a^(b-1)`, `grad_b += out_grad * a^b * ln(a)`.
    #[must_use]
    pub fn epow(&self, rhs: &Self) -> Self {
        let out = self.data.epow(&rhs.data);
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let (a_data, b_data) = (self.data.clone(), rhs.data.clone());
        let result = out.clone();
        let entry = TapeEntry::new(move |g| {
            // d/da (a^b) = b * a^(b-1)
            let one = T::one_impl();
            let (m, n) = a_data.shape();
            let a_pow_bm1 = Tensor::from_fn(m, n, |r, c| {
                a_data.get(r, c).math_powf(b_data.get(r, c) - one)
            });
            Self::prop(&lr, &g.emul(&b_data.emul(&a_pow_bm1)));
            // d/db (a^b) = a^b * ln(a)
            Self::prop(&rr, &g.emul(&result.emul(&a_data.ln())));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Matrix multiply (`self @ rhs`).
    ///
    /// backward: `grad_a += out_grad @ rhs^T`, `grad_b += self^T @ out_grad`.
    #[must_use]
    pub fn matmul(&self, rhs: &Self) -> Self {
        let out = &self.data * &rhs.data;
        let (lr, rr) = (self.input_refs(), rhs.input_refs());
        let (a_t, b_t) = (self.data.t(), rhs.data.t());
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &(g * &b_t));
            Self::prop(&rr, &(&a_t * g));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Scalar multiply `self * s`.
    ///
    /// backward: `grad_a += out_grad * s`.
    #[must_use]
    pub fn scale(&self, s: T) -> Self {
        let out = &self.data * s;
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &(g * s));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `exp(x)`.
    ///
    /// backward: `grad_a += out_grad * exp(a)`.
    #[must_use]
    pub fn exp(&self) -> Self {
        let out = self.data.exp();
        let lr = self.input_refs();
        let exp_a = out.clone();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&exp_a));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `ln(x)`.
    ///
    /// backward: `grad_a += out_grad / a`.
    #[must_use]
    pub fn ln(&self) -> Self {
        let out = self.data.ln();
        let lr = self.input_refs();
        let a_data = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.ediv(&a_data));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `sin(x)`.
    ///
    /// backward: `grad_a += out_grad * cos(a)`.
    #[must_use]
    pub fn sin(&self) -> Self {
        let out = self.data.sin();
        let lr = self.input_refs();
        let cos_a = self.data.cos();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&cos_a));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `cos(x)`.
    ///
    /// backward: `grad_a += out_grad * (-sin(a))`.
    #[must_use]
    pub fn cos(&self) -> Self {
        let out = self.data.cos();
        let lr = self.input_refs();
        let neg_sin_a = -&self.data.sin();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&neg_sin_a));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `tanh(x)`.
    ///
    /// backward: `grad_a += out_grad * (1 - tanh(a)^2)`.
    #[must_use]
    pub fn tanh(&self) -> Self {
        let out = self.data.tanh();
        let lr = self.input_refs();
        let (nrows, ncols) = out.shape();
        let ones = Tensor::fill(nrows, ncols, T::one_impl());
        let sech2 = &ones - &out.emul(&out); // 1 - tanh²(x)
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&sech2));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `sqrt(x)`.
    ///
    /// backward: `grad_a += out_grad / (2 * sqrt(a))`.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        let out = self.data.sqrt();
        let lr = self.input_refs();
        let two = T::one_impl() + T::one_impl();
        let two_sqrt_a = &out * two;
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.ediv(&two_sqrt_a));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `a^p` for scalar `p`.
    ///
    /// backward: `grad_a += out_grad * p * a^(p-1)`.
    #[must_use]
    pub fn powf(&self, p: T) -> Self {
        let out = self.data.powf(p);
        let lr = self.input_refs();
        let one = T::one_impl();
        let coeff = &self.data.powf(p - one) * p; // p * a^(p-1)
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&coeff));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise ReLU: `max(x, 0)`.
    ///
    /// backward: `grad * (input > 0)` — gradient flows only where input was positive.
    #[must_use]
    pub fn relu(&self) -> Self {
        let out = self.data.relu();
        let lr = self.input_refs();
        let input = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            // Mask: 1 where input > 0, 0 otherwise.
            let (m, n) = input.shape();
            let mask = Tensor::from_fn(m, n, |r, c| {
                if input.get(r, c).to_f64() > 0.0 {
                    T::one_impl()
                } else {
                    T::zero()
                }
            });
            Self::prop(&lr, &g.emul(&mask));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    ///
    /// backward: `grad * output * (1 - output)`.
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        let out = self.data.sigmoid();
        let lr = self.input_refs();
        let sig_out = out.clone();
        let entry = TapeEntry::new(move |g| {
            let (m, n) = sig_out.shape();
            let ones = Tensor::fill(m, n, T::one_impl());
            let dsig = sig_out.emul(&(&ones - &sig_out)); // σ(x) * (1 - σ(x))
            Self::prop(&lr, &g.emul(&dsig));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise GELU (tanh approximation).
    ///
    /// backward: approximate derivative using the exact GELU gradient:
    /// `0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2 / 2) / sqrt(2 * pi)`.
    #[must_use]
    pub fn gelu(&self) -> Self {
        let out = self.data.gelu();
        let lr = self.input_refs();
        let input = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            let inv_sqrt2 = T::from_f64(std::f64::consts::FRAC_1_SQRT_2); // 1/√2
            let inv_sqrt_2pi = T::from_f64(1.0 / (2.0 * std::f64::consts::PI).sqrt());
            let half = T::from_f64(0.5);
            let (m, n) = input.shape();
            let dgelu = Tensor::from_fn(m, n, |r, c| {
                let x = input.get(r, c);
                let cdf = half * (T::one_impl() + (x * inv_sqrt2).math_erf());
                let pdf = (T::from_f64(-0.5) * x * x).math_exp() * inv_sqrt_2pi;
                cdf + x * pdf
            });
            Self::prop(&lr, &g.emul(&dgelu));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise absolute value.
    ///
    /// backward: `grad * sign(input)`.
    #[must_use]
    pub fn abs(&self) -> Self {
        let out = self.data.abs();
        let lr = self.input_refs();
        let input = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            let (m, n) = input.shape();
            let sign = Tensor::from_fn(m, n, |r, c| {
                let x = input.get(r, c).to_f64();
                if x > 0.0 {
                    T::one_impl()
                } else if x < 0.0 {
                    T::zero() - T::one_impl()
                } else {
                    T::zero()
                }
            });
            Self::prop(&lr, &g.emul(&sign));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise `ln(1 + x)`.
    ///
    /// backward: `grad / (1 + input)`.
    #[must_use]
    pub fn log1p(&self) -> Self {
        let out = self.data.log1p();
        let lr = self.input_refs();
        let input = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            let (m, n) = input.shape();
            let ones = Tensor::fill(m, n, T::one_impl());
            let denom = &ones + &input;
            Self::prop(&lr, &g.ediv(&denom));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise SiLU (Swish): `x * sigmoid(x)`.
    ///
    /// backward: `grad * sigmoid(x) * (1 + x * (1 - sigmoid(x)))`.
    #[must_use]
    pub fn silu(&self) -> Self {
        let out = self.data.silu();
        let lr = self.input_refs();
        let input = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            let sig = input.sigmoid();
            let (m, n) = input.shape();
            let ones = Tensor::fill(m, n, T::one_impl());
            // dsilu = sigmoid * (1 + x * (1 - sigmoid))
            let dsilu = sig.emul(&(&ones + &input.emul(&(&ones - &sig))));
            Self::prop(&lr, &g.emul(&dsilu));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Sum along `axis` (0 = column-wise → 1×n, 1 = row-wise → m×1).
    ///
    /// backward: broadcast grad back by expanding along the reduced axis.
    #[must_use]
    pub fn sum_axis_var(&self, axis: usize) -> Self {
        let out = self.data.sum_axis(axis);
        let lr = self.input_refs();
        let (in_rows, in_cols) = self.data.shape();
        let entry = TapeEntry::new(move |g| {
            // g has shape (1, ncols) or (nrows, 1); expand to input shape.
            Self::prop(&lr, &g.expand(in_rows, in_cols));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Mean of all elements → scalar Variable of shape `(1, 1)`.
    ///
    /// backward: `grad / n_elements` broadcast to input shape.
    #[must_use]
    pub fn mean_var(&self) -> Self {
        let (nrows, ncols) = self.data.shape();
        let n = T::from_f64((nrows * ncols) as f64);
        let s = self.data.sum_all();
        let out = Tensor::fill(1, 1, s / n);
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0) / n;
            Self::prop(&lr, &Tensor::fill(nrows, ncols, g_val));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Cross-entropy loss: fused log-softmax + NLL.
    ///
    /// Forward: `mean(-sum(targets * log_softmax(logits), axis=1))` per sample.
    /// Backward for logits: `(softmax(logits) - targets) / batch_size`.
    ///
    /// `targets` should be one-hot or probability tensors (not class indices).
    #[must_use]
    pub fn cross_entropy(&self, targets: &Tensor<T, B>) -> Self {
        let (batch, n) = self.data.shape();
        assert_eq!(
            targets.shape(),
            (batch, n),
            "nabla: cross_entropy shape mismatch -- logits {}x{} vs targets {}x{}",
            batch,
            n,
            targets.nrows(),
            targets.ncols()
        );
        // Forward: log_softmax along axis=1 (rows), then -mean(sum(targets * log_softmax)).
        let log_sm = self.data.log_softmax(1);
        let loss_val = log_sm.cross_entropy_loss(targets);
        let out = Tensor::fill(1, 1, loss_val);

        let lr = self.input_refs();
        let logits_data = self.data.clone();
        let tgt = targets.clone();
        let entry = TapeEntry::new(move |g| {
            let sm = logits_data.softmax(1);
            let inv_batch = T::from_f64(1.0 / batch as f64);
            let g_val = g.get(0, 0);
            // dL/d(logits) = (softmax - targets) / batch * upstream_grad
            let delta = &(&sm - &tgt) * (g_val * inv_batch);
            Self::prop(&lr, &delta);
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
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            Self::prop(&lr, &Tensor::fill(nrows, ncols, g_val));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Softmax along `axis`.
    ///
    /// backward: `grad * softmax - softmax * sum(grad * softmax, axis)`.
    #[must_use]
    pub fn softmax(&self, axis: usize) -> Self {
        let out = self.data.softmax(axis);
        let lr = self.input_refs();
        let sm = out.clone();
        let entry = TapeEntry::new(move |g| {
            let gs = g.emul(&sm);
            let sum_gs = gs.sum_axis(axis);
            let (m, n) = sm.shape();
            let delta = &gs - &sm.emul(&sum_gs.expand(m, n));
            Self::prop(&lr, &delta);
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Reshape to `(nrows, ncols)`.
    ///
    /// backward: reshape grad back to original shape.
    #[must_use]
    pub fn reshape(&self, nrows: usize, ncols: usize) -> Self {
        let (orig_r, orig_c) = self.data.shape();
        let out = self.data.reshape(nrows, ncols);
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.reshape(orig_r, orig_c));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Transpose.
    ///
    /// backward: transpose the gradient.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let out = self.data.t();
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.t());
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Linear forward: `x @ weight^T + bias` (all tracked).
    ///
    /// backward: `grad_x = g @ W`, `grad_w = g^T @ x`, `grad_b = sum(g, axis=0)`.
    #[must_use]
    pub fn linear_forward(&self, weight: &Self, bias: &Self) -> Self {
        let out = &(&self.data * &weight.data.t()) + &bias.data;
        let (xr, wr, br) = (self.input_refs(), weight.input_refs(), bias.input_refs());
        let (x_data, w_data) = (self.data.clone(), weight.data.clone());
        let entry = TapeEntry::new(move |g| {
            Self::prop(&xr, &(g * &w_data));
            Self::prop(&wr, &(&g.t() * &x_data));
            Self::prop(&br, &g.sum_axis(0));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Dropout with probability `p`. No-op when `training` is false.
    ///
    /// backward: `grad * mask * scale`.
    #[must_use]
    pub fn dropout(&self, p: f64, training: bool) -> Self {
        if !training || p <= 0.0 {
            return self.scale(T::one_impl()); // identity through tape
        }
        let (m, n) = self.data.shape();
        if p >= 1.0 {
            let out = Tensor::zeros(m, n);
            let lr = self.input_refs();
            let entry = TapeEntry::new(move |_g| {
                Self::prop(&lr, &Tensor::zeros(m, n));
            });
            return Self::derived(&self.tape, out, entry);
        }
        let scale = T::from_f64(1.0 / (1.0 - p));
        let threshold = (p * (u64::MAX as f64)) as u64;
        let mut s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0xDEAD_BEEF_u64, |d| d.as_nanos() as u64);
        // Build mask
        let mask = Tensor::from_fn(m, n, |_r, _c| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s < threshold { T::zero() } else { T::one_impl() }
        });
        let out = self.data.emul(&mask).emul(&Tensor::fill(m, n, scale));
        let lr = self.input_refs();
        let entry = TapeEntry::new(move |g| {
            Self::prop(&lr, &g.emul(&mask).emul(&Tensor::fill(m, n, scale)));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Element-wise clamp to `[lo, hi]`.
    ///
    /// backward: grad passes through where `lo <= x <= hi`, zero otherwise.
    #[must_use]
    pub fn clamp(&self, lo: T, hi: T) -> Self {
        let out = self.data.clamp(lo, hi);
        let lr = self.input_refs();
        let input = self.data.clone();
        let entry = TapeEntry::new(move |g| {
            let (m, n) = input.shape();
            let lo_f = lo.to_f64();
            let hi_f = hi.to_f64();
            let mask = Tensor::from_fn(m, n, |r, c| {
                let v = input.get(r, c).to_f64();
                if v >= lo_f && v <= hi_f { T::one_impl() } else { T::zero() }
            });
            Self::prop(&lr, &g.emul(&mask));
        });
        Self::derived(&self.tape, out, entry)
    }

    /// Cross-entropy with integer class indices (as `T` values cast to usize).
    ///
    /// Converts indices to one-hot internally, then delegates to [`cross_entropy`].
    #[must_use]
    pub fn cross_entropy_indices(&self, targets: &Tensor<T, B>) -> Self {
        let (batch, n) = self.data.shape();
        let one_hot = Tensor::from_fn(batch, n, |r, c| {
            let idx = targets.get(r, 0).to_f64() as usize;
            if c == idx { T::one_impl() } else { T::zero() }
        });
        self.cross_entropy(&one_hot)
    }

    /// MSE loss: `mean((self - target)^2)` → scalar `(1,1)`.
    ///
    /// backward: `2 * (self - target) / n`.
    #[must_use]
    pub fn mse_loss(&self, target: &Self) -> Self {
        let diff = &self.data - &target.data;
        let (m, n) = diff.shape();
        let count = T::from_f64((m * n) as f64);
        let sq_sum = diff.emul(&diff).sum_all();
        let out = Tensor::fill(1, 1, sq_sum / count);
        let (lr, rr) = (self.input_refs(), target.input_refs());
        let two_over_n = T::from_f64(2.0) / count;
        let entry = TapeEntry::new(move |g| {
            let g_val = g.get(0, 0);
            let delta = &diff * (g_val * two_over_n);
            Self::prop(&lr, &delta);
            Self::prop(&rr, &(-&delta));
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
    /// Returns `Err` if any gradient contains NaN or Inf.
    pub fn backward(&self) -> Result<()> {
        let (nrows, ncols) = self.data.shape();
        let seed = Tensor::fill(nrows, ncols, T::one_impl());

        // Deposit seed into this variable's accumulator.
        if let Some(entry) = &self.tape_entry {
            entry.accum(&seed);
        } else if let Some(slot) = &self.grad_slot {
            accum_cell(slot, &seed);
        }

        // Reverse topological walk: entries were pushed in forward order.
        let entries = self.tape.entries.borrow();
        for entry in entries.iter().rev() {
            let g_opt = entry.grad.borrow().clone();
            if let Some(g) = g_opt {
                // Check for NaN/Inf in gradient before propagating.
                let (m, n) = g.shape();
                for r in 0..m {
                    for c in 0..n {
                        let v = g.get(r, c).to_f64();
                        if v.is_nan() || v.is_infinite() {
                            return Err(nabla_core::error::Error::eval(
                                format!("NaN/Inf detected in gradient at ({r}, {c})")
                            ));
                        }
                    }
                }
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

/// Scalar divide (`&variable / scalar`) — delegates to `scale(1/rhs)`.
impl<T: Scalar, B: Backend> Div<T> for &Variable<T, B> {
    type Output = Variable<T, B>;

    fn div(self, rhs: T) -> Self::Output {
        self.scale(T::one_impl() / rhs)
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
    grad_impl(f, x)
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
    grad_impl(&f, x)
}

#[cfg(feature = "cpu")]
fn grad_impl<T, F>(
    f: &F,
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

// ---------------------------------------------------------------------------
// Gradient utilities
// ---------------------------------------------------------------------------

/// Clip gradient norms in-place by global norm.
///
/// Computes the total L2 norm across all gradients and, if it exceeds
/// `max_norm`, scales every gradient down uniformly so the total norm equals
/// `max_norm`.  Returns the total norm **before** clipping.
pub fn clip_grad_norm<T: Scalar, B: Backend>(grads: &mut [Tensor<T, B>], max_norm: f64) -> f64 {
    let total_norm = grads
        .iter()
        .map(|g| g.norm().to_f64().powi(2))
        .sum::<f64>()
        .sqrt();
    if total_norm > max_norm {
        let scale = T::from_f64(max_norm / total_norm);
        for g in grads.iter_mut() {
            *g = &*g * scale;
        }
    }
    total_norm
}

/// Zero out all gradients in-place, preserving their shapes.
pub fn zero_grad<T: Scalar, B: Backend>(grads: &mut [Tensor<T, B>]) {
    for g in grads.iter_mut() {
        *g = &*g * T::zero();
    }
}

/// Scale all gradients in-place by a scalar factor.
pub fn scale_grad<T: Scalar, B: Backend>(grads: &mut [Tensor<T, B>], factor: T) {
    for g in grads.iter_mut() {
        *g = &*g * factor;
    }
}
