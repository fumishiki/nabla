use std::rc::Rc;

use crate::optim::{GradScaler, LrSchedule, Optimizer, ScheduleState, lr_at_step};
use nabla_core::backend::Backend;
use nabla_core::backend::error::Error as TrainError;
use nabla_core::error::Result;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;
use nabla_ml::autograd::{Tape, Variable};
use nabla_ml::module::Module;

pub struct TrainState {
    pub epoch: usize,
    pub step: usize,
    pub grad_accum: usize,
    pub rng_state: Option<u64>,
}

pub struct TrainStepOut<T: Scalar, B: Backend> {
    pub loss: Variable<T, B>,
    pub params: Vec<Variable<T, B>>,
}

pub enum HookAction {
    Continue,
    Stop,
}

pub enum TrainEvent {
    Step {
        epoch: usize,
        step: usize,
        loss: f64,
    },
    EpochEnd {
        epoch: usize,
        steps: usize,
    },
    EvalStep {
        epoch: usize,
        step: usize,
        loss: f64,
    },
    EvalEnd {
        epoch: usize,
        steps: usize,
        avg_loss: f64,
    },
}

pub trait TrainHook {
    fn on_event(&mut self, event: &TrainEvent) -> HookAction;
}

pub struct Trainer<T: Scalar, B: Backend, M: Module<T, B>, O: Optimizer<T, B>> {
    model: M,
    optimizer: O,
    schedule: Option<(LrSchedule, f64)>,
    scaler: Option<GradScaler>,
    grad_clip: Option<f64>,
    nan_policy: Option<GradNanPolicy>,
    metrics: Option<MetricStats>,
    metrics_scope: MetricsScope,
    early_stop: Option<EarlyStopState>,
    state: TrainState,
    hooks: Vec<Box<dyn TrainHook>>,
    _phantom: std::marker::PhantomData<(T, B)>,
}

impl<T: Scalar, B: Backend, M: Module<T, B>, O: Optimizer<T, B>> Trainer<T, B, M, O> {
    pub fn new(model: M, optimizer: O) -> Self {
        Self {
            model,
            optimizer,
            schedule: None,
            scaler: None,
            grad_clip: None,
            nan_policy: None,
            metrics: None,
            metrics_scope: MetricsScope::Train,
            early_stop: None,
            state: TrainState {
                epoch: 0,
                step: 0,
                grad_accum: 1,
                rng_state: None,
            },
            hooks: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    #[must_use]
    pub fn with_schedule(mut self, schedule: LrSchedule, base_lr: f64) -> Self {
        self.schedule = Some((schedule, base_lr));
        self
    }

    #[must_use]
    pub fn with_grad_scaler(mut self, scaler: GradScaler) -> Self {
        self.scaler = Some(scaler);
        self
    }

    #[must_use]
    pub fn with_grad_accum(mut self, steps: usize) -> Self {
        self.state.grad_accum = steps.max(1);
        self
    }

    #[must_use]
    pub fn with_nan_policy(mut self, policy: GradNanPolicy) -> Self {
        self.nan_policy = Some(policy);
        self
    }

    #[must_use]
    pub fn with_metrics(mut self) -> Self {
        self.metrics = Some(MetricStats::new());
        self.metrics_scope = MetricsScope::Train;
        self
    }

    #[must_use]
    pub fn with_metrics_scope(mut self, scope: MetricsScope) -> Self {
        self.metrics_scope = scope;
        self
    }

    #[must_use]
    pub fn with_early_stop(mut self, cfg: EarlyStop) -> Self {
        self.early_stop = Some(EarlyStopState::new(cfg));
        self
    }

    #[must_use]
    pub fn with_grad_clip(mut self, max_norm: f64) -> Self {
        self.grad_clip = if max_norm > 0.0 { Some(max_norm) } else { None };
        self
    }

    pub fn add_hook(&mut self, hook: Box<dyn TrainHook>) {
        self.hooks.push(hook);
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    pub fn scaler_mut(&mut self) -> Option<&mut GradScaler> {
        self.scaler.as_mut()
    }

    pub fn metrics(&self) -> Option<&MetricStats> {
        self.metrics.as_ref()
    }

    pub fn early_stop_triggered(&self) -> bool {
        self.early_stop.as_ref().map_or(false, |s| s.triggered)
    }

    pub fn state(&self) -> &TrainState {
        &self.state
    }

    pub fn set_state(&mut self, state: TrainState) {
        self.state = state;
    }

    pub fn set_rng_state(&mut self, rng_state: Option<u64>) {
        self.state.rng_state = rng_state;
    }

    pub fn schedule_state(&self) -> Option<ScheduleState> {
        self.schedule.as_ref().map(|(s, base_lr)| ScheduleState {
            base_lr: *base_lr,
            schedule: s.clone(),
        })
    }

    pub fn train_epoch<D, F, Batch>(&mut self, loader: D, step_fn: F) -> Result<usize>
    where
        D: Iterator<Item = Batch>,
        F: FnMut(&mut M, &Batch, &Rc<Tape<T, B>>) -> Result<TrainStepOut<T, B>>,
    {
        self.train_epoch_with_max_steps(loader, step_fn, None)
    }

    #[allow(clippy::too_many_lines)]
    pub fn train_epoch_with_max_steps<D, F, Batch>(
        &mut self,
        mut loader: D,
        mut step_fn: F,
        max_steps: Option<usize>,
    ) -> Result<usize>
    where
        D: Iterator<Item = Batch>,
        F: FnMut(&mut M, &Batch, &Rc<Tape<T, B>>) -> Result<TrainStepOut<T, B>>,
    {
        let mut steps_done = 0usize;
        let mut accum_grads: Option<Vec<Tensor<T, B>>> = None;
        let mut accum_count = 0usize;

        while let Some(batch) = loader.next() {
            let tape = Tape::new();
            let mut out = step_fn(&mut self.model, &batch, &tape)?;
            if let Some(scaler) = &self.scaler {
                let scale = T::from_f64(scaler.scale_factor());
                out.loss = out.loss.scale(scale);
            }
            out.loss.backward()?;

            let mut grads = Vec::with_capacity(out.params.len());
            for v in &out.params {
                grads.push(v.grad()?);
            }

            if let Some(scaler) = &mut self.scaler {
                let ok = scaler.unscale_and_update(&mut grads);
                if !ok {
                    accum_grads = None;
                    accum_count = 0;
                    continue;
                }
            }

            if self.has_non_finite(&grads) {
                match self.nan_policy {
                    Some(GradNanPolicy::Skip) => {
                        accum_grads = None;
                        accum_count = 0;
                        continue;
                    }
                    Some(GradNanPolicy::Stop) => {
                        return Err(TrainError::EvalError("non-finite gradient".to_owned()));
                    }
                    None => {}
                }
            }

            if accum_grads.is_none() {
                let mut buf = Vec::with_capacity(grads.len());
                for g in &grads {
                    let (r, c) = g.shape();
                    buf.push(Tensor::zeros(r, c));
                }
                accum_grads = Some(buf);
            }

            if let Some(buf) = &mut accum_grads {
                for (i, g) in grads.iter().enumerate() {
                    buf[i] = &buf[i] + g;
                }
            }
            accum_count += 1;

            if accum_count >= self.state.grad_accum {
                if let Some((schedule, base_lr)) = &self.schedule {
                    let lr = lr_at_step(schedule, *base_lr, self.state.step);
                    self.optimizer.set_lr(lr);
                }
                if let Some(buf) = &mut accum_grads {
                    if let Some(max_norm) = self.grad_clip {
                        clip_grad_norm(buf, max_norm);
                    }
                }
                let mut params = self.model.parameters_mut();
                if let Some(buf) = &accum_grads {
                    let grad_refs: Vec<&Tensor<T, B>> = buf.iter().collect();
                    self.optimizer.step(&mut params, &grad_refs);
                }
                accum_count = 0;
                if let Some(buf) = &mut accum_grads {
                    for g in buf.iter_mut() {
                        let (r, c) = g.shape();
                        *g = Tensor::zeros(r, c);
                    }
                }
                self.state.step += 1;
                steps_done += 1;

                let need_loss = self.metrics_scope.track_train() || !self.hooks.is_empty();
                if need_loss {
                    let loss_val = out.loss.data().get(0, 0).to_f64();
                    if self.metrics_scope.track_train() {
                        if let Some(stats) = &mut self.metrics {
                            stats.update(loss_val);
                        }
                    }
                    if !self.hooks.is_empty()
                        && self.fire_hooks(TrainEvent::Step {
                            epoch: self.state.epoch,
                            step: self.state.step,
                            loss: loss_val,
                        })
                    {
                        break;
                    }
                }
                if let Some(max) = max_steps {
                    if steps_done >= max {
                        break;
                    }
                }
            }
        }

        self.fire_hooks(TrainEvent::EpochEnd {
            epoch: self.state.epoch,
            steps: steps_done,
        });
        if self.metrics_scope.track_train() {
            if let Some(stats) = &self.metrics {
                if let Some(early) = &mut self.early_stop {
                    if early.should_stop(stats.last) {
                        early.triggered = true;
                    }
                }
            }
        }
        self.state.epoch += 1;

        Ok(steps_done)
    }

    pub fn eval_epoch<D, F, Batch>(&mut self, mut loader: D, mut eval_fn: F) -> Result<f64>
    where
        D: Iterator<Item = Batch>,
        F: FnMut(&M, &Batch) -> Result<f64>,
    {
        let prev = self.model.training();
        self.model.set_training(false);
        let mut count = 0usize;
        let mut sum = 0.0f64;
        while let Some(batch) = loader.next() {
            let loss = eval_fn(&self.model, &batch)?;
            sum += loss;
            count += 1;
            if self.metrics_scope.track_eval() {
                if let Some(stats) = &mut self.metrics {
                    stats.update(loss);
                }
                let _ = self.fire_hooks(TrainEvent::EvalStep {
                    epoch: self.state.epoch,
                    step: count,
                    loss,
                });
            }
        }
        self.model.set_training(prev);
        if count == 0 {
            return Err(TrainError::InvalidDimension("empty eval loader".to_owned()));
        }
        let avg = sum / count as f64;
        if self.metrics_scope.track_eval() {
            let _ = self.fire_hooks(TrainEvent::EvalEnd {
                epoch: self.state.epoch,
                steps: count,
                avg_loss: avg,
            });
        }
        Ok(avg)
    }

    fn fire_hooks(&mut self, event: TrainEvent) -> bool {
        let mut stop = false;
        for hook in &mut self.hooks {
            if let HookAction::Stop = hook.on_event(&event) {
                stop = true;
            }
        }
        stop
    }

    fn has_non_finite(&self, grads: &[Tensor<T, B>]) -> bool {
        for g in grads {
            let (m, n) = g.shape();
            for r in 0..m {
                for c in 0..n {
                    let v = g.get(r, c).to_f64();
                    if !v.is_finite() {
                        return true;
                    }
                }
            }
        }
        false
    }
}

pub enum GradNanPolicy {
    Skip,
    Stop,
}

pub enum MetricsScope {
    None,
    Train,
    Eval,
    Both,
}

impl MetricsScope {
    fn track_train(&self) -> bool {
        matches!(self, MetricsScope::Train | MetricsScope::Both)
    }

    fn track_eval(&self) -> bool {
        matches!(self, MetricsScope::Eval | MetricsScope::Both)
    }
}

pub enum EarlyStop {
    LossBelow(f64),
    NoImprove { patience: usize, min_delta: f64 },
}

pub struct MetricStats {
    pub last: f64,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    count: usize,
}

impl MetricStats {
    #[must_use]
    pub fn new() -> Self {
        Self {
            last: 0.0,
            avg: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            count: 0,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.last = value;
        self.count += 1;
        self.avg += (value - self.avg) / self.count as f64;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }
}

pub struct EarlyStopState {
    cfg: EarlyStop,
    best: f64,
    bad_epochs: usize,
    triggered: bool,
}

impl EarlyStopState {
    fn new(cfg: EarlyStop) -> Self {
        Self {
            cfg,
            best: f64::INFINITY,
            bad_epochs: 0,
            triggered: false,
        }
    }

    fn should_stop(&mut self, loss: f64) -> bool {
        match self.cfg {
            EarlyStop::LossBelow(threshold) => loss < threshold,
            EarlyStop::NoImprove {
                patience,
                min_delta,
            } => {
                if loss < self.best - min_delta {
                    self.best = loss;
                    self.bad_epochs = 0;
                    false
                } else {
                    self.bad_epochs += 1;
                    self.bad_epochs >= patience
                }
            }
        }
    }
}

pub fn clip_grad_norm<T: Scalar, B: Backend>(grads: &mut [Tensor<T, B>], max_norm: f64) -> f64 {
    if max_norm <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for g in grads.iter() {
        let (m, n) = g.shape();
        for r in 0..m {
            for c in 0..n {
                let v = g.get(r, c).to_f64();
                sum += v * v;
            }
        }
    }
    let norm = sum.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        let scale_t = T::from_f64(scale);
        for g in grads.iter_mut() {
            *g = g.map(|x| x * scale_t);
        }
    }
    norm
}
