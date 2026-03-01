use std::collections::HashMap;

use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;
use nabla_ml::autograd::Variable;
use nabla_ml::module::Module;

pub trait Optimizer<T: Scalar, B: Backend> {
    fn step(&mut self, params: &mut [&mut Tensor<T, B>], grads: &[&Tensor<T, B>]);

    fn step_slices(&mut self, params: &mut [Tensor<T, B>], grads: &[Tensor<T, B>]) {
        let mut param_refs: Vec<&mut Tensor<T, B>> = params.iter_mut().collect();
        let grad_refs: Vec<&Tensor<T, B>> = grads.iter().collect();
        self.step(&mut param_refs, &grad_refs);
    }

    fn step_with_vars<M: Module<T, B>>(&mut self, module: &mut M, param_vars: &[Variable<T, B>]) {
        let params = module.parameters_mut();
        assert_eq!(
            params.len(),
            param_vars.len(),
            "nabla-train: step_with_vars param_vars length ({}) doesn't match module parameters count ({})",
            param_vars.len(),
            params.len(),
        );
        let grads: Vec<Tensor<T, B>> = param_vars
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.grad().unwrap_or_else(|_| panic!(
                "nabla-train: step_with_vars: param_vars[{i}] has no gradient (call backward first)"
            ))
            })
            .collect();
        let grad_refs: Vec<&Tensor<T, B>> = grads.iter().collect();
        let mut param_refs: Vec<&mut Tensor<T, B>> = params;
        self.step(&mut param_refs, &grad_refs);
    }

    fn reset(&mut self);

    fn set_lr(&mut self, _lr: f64) {}
}

#[derive(Clone, Copy)]
pub enum OptimKind {
    AdamW,
    Adam,
    Sgd,
}

pub struct OptimMeta {
    pub kind: OptimKind,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub step_count: usize,
}

pub trait OptimState<T: Scalar, B: Backend> {
    fn kind(&self) -> OptimKind;
    fn state_tensors(&self) -> Vec<(String, Tensor<T, B>)>;
    fn load_state_tensors(&mut self, tensors: &[(String, Tensor<T, B>)]) -> Result<(), String>;
    fn meta(&self) -> OptimMeta;
    fn load_meta(&mut self, meta: &OptimMeta) -> Result<(), String>;

    fn meta_pairs(&self) -> Vec<(String, String)> {
        optim_meta_pairs(&self.meta())
    }

    fn load_meta_pairs(&mut self, map: &HashMap<String, String>) -> Result<(), String> {
        let meta = parse_optim_meta(map)?;
        self.load_meta(&meta)
    }
}

#[inline]
fn warmup_lr(base_lr: f64, step: usize, warmup_steps: usize) -> f64 {
    base_lr * step as f64 / warmup_steps.max(1) as f64
}

#[inline]
fn decay_progress(step: usize, start: usize, total: usize) -> f64 {
    (step - start) as f64 / (total - start).max(1) as f64
}

#[inline]
fn ramp(start: f64, end: f64, step: usize, total_steps: usize) -> f64 {
    start + (end - start) * step as f64 / total_steps.max(1) as f64
}

#[inline]
fn cosine_decay(high: f64, low: f64, progress: f64) -> f64 {
    low + 0.5 * (high - low) * (1.0 + (std::f64::consts::PI * progress).cos())
}

#[derive(Clone)]
pub enum LrSchedule {
    Cosine {
        warmup_steps: usize,
        total_steps: usize,
        min_lr: f64,
    },
    Linear {
        warmup_steps: usize,
        total_steps: usize,
    },
    OneCycle {
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
    },
    Step {
        step_size: usize,
        gamma: f64,
    },
}

#[derive(Clone)]
pub struct ScheduleState {
    pub base_lr: f64,
    pub schedule: LrSchedule,
}

#[must_use]
pub fn lr_at_step(schedule: &LrSchedule, base_lr: f64, step: usize) -> f64 {
    match schedule {
        LrSchedule::Cosine {
            warmup_steps,
            total_steps,
            min_lr,
        } => {
            if step < *warmup_steps {
                warmup_lr(base_lr, step, *warmup_steps)
            } else {
                let progress = decay_progress(step, *warmup_steps, *total_steps);
                cosine_decay(base_lr, *min_lr, progress)
            }
        }
        LrSchedule::Linear {
            warmup_steps,
            total_steps,
        } => {
            if step < *warmup_steps {
                warmup_lr(base_lr, step, *warmup_steps)
            } else {
                let progress = decay_progress(step, *warmup_steps, *total_steps);
                base_lr * (1.0 - progress).max(0.0)
            }
        }
        LrSchedule::OneCycle {
            max_lr,
            total_steps,
            pct_start,
        } => {
            let up_steps = (*total_steps as f64 * pct_start) as usize;
            if step < up_steps {
                ramp(base_lr, *max_lr, step, up_steps)
            } else {
                let progress = decay_progress(step, up_steps, *total_steps);
                cosine_decay(*max_lr, base_lr, progress)
            }
        }
        LrSchedule::Step { step_size, gamma } => {
            let k = step / (*step_size).max(1);
            base_lr * gamma.powi(k as i32)
        }
    }
}

#[must_use]
pub fn schedule_pairs(state: &ScheduleState) -> Vec<(String, String)> {
    let mut out = Vec::new();
    out.push(("schedule.enabled".to_owned(), "1".to_owned()));
    out.push(("schedule.base_lr".to_owned(), state.base_lr.to_string()));
    match &state.schedule {
        LrSchedule::Cosine {
            warmup_steps,
            total_steps,
            min_lr,
        } => {
            out.push(("schedule.kind".to_owned(), "cosine".to_owned()));
            out.push(("schedule.warmup_steps".to_owned(), warmup_steps.to_string()));
            out.push(("schedule.total_steps".to_owned(), total_steps.to_string()));
            out.push(("schedule.min_lr".to_owned(), min_lr.to_string()));
        }
        LrSchedule::Linear {
            warmup_steps,
            total_steps,
        } => {
            out.push(("schedule.kind".to_owned(), "linear".to_owned()));
            out.push(("schedule.warmup_steps".to_owned(), warmup_steps.to_string()));
            out.push(("schedule.total_steps".to_owned(), total_steps.to_string()));
        }
        LrSchedule::OneCycle {
            max_lr,
            total_steps,
            pct_start,
        } => {
            out.push(("schedule.kind".to_owned(), "onecycle".to_owned()));
            out.push(("schedule.max_lr".to_owned(), max_lr.to_string()));
            out.push(("schedule.total_steps".to_owned(), total_steps.to_string()));
            out.push(("schedule.pct_start".to_owned(), pct_start.to_string()));
        }
        LrSchedule::Step { step_size, gamma } => {
            out.push(("schedule.kind".to_owned(), "step".to_owned()));
            out.push(("schedule.step_size".to_owned(), step_size.to_string()));
            out.push(("schedule.gamma".to_owned(), gamma.to_string()));
        }
    }
    out
}

pub fn parse_schedule_state(
    map: &HashMap<String, String>,
) -> Result<Option<ScheduleState>, String> {
    let enabled = map
        .get("schedule.enabled")
        .map(String::as_str)
        .unwrap_or("0");
    if enabled != "1" {
        return Ok(None);
    }
    let base_lr = parse_f64(map, "schedule.base_lr")?;
    let kind = parse_str(map, "schedule.kind")?;
    let schedule = match kind {
        "cosine" => LrSchedule::Cosine {
            warmup_steps: parse_usize(map, "schedule.warmup_steps")?,
            total_steps: parse_usize(map, "schedule.total_steps")?,
            min_lr: parse_f64(map, "schedule.min_lr")?,
        },
        "linear" => LrSchedule::Linear {
            warmup_steps: parse_usize(map, "schedule.warmup_steps")?,
            total_steps: parse_usize(map, "schedule.total_steps")?,
        },
        "onecycle" => LrSchedule::OneCycle {
            max_lr: parse_f64(map, "schedule.max_lr")?,
            total_steps: parse_usize(map, "schedule.total_steps")?,
            pct_start: parse_f64(map, "schedule.pct_start")?,
        },
        "step" => LrSchedule::Step {
            step_size: parse_usize(map, "schedule.step_size")?,
            gamma: parse_f64(map, "schedule.gamma")?,
        },
        _ => return Err(format!("bad schedule.kind: {kind}")),
    };
    Ok(Some(ScheduleState { base_lr, schedule }))
}

pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    consecutive_ok: usize,
}

#[must_use]
pub fn optim_meta_pairs(meta: &OptimMeta) -> Vec<(String, String)> {
    vec![
        ("optim.kind".to_owned(), optim_kind_str(meta.kind)),
        ("optim.lr".to_owned(), meta.lr.to_string()),
        ("optim.beta1".to_owned(), meta.beta1.to_string()),
        ("optim.beta2".to_owned(), meta.beta2.to_string()),
        ("optim.eps".to_owned(), meta.eps.to_string()),
        (
            "optim.weight_decay".to_owned(),
            meta.weight_decay.to_string(),
        ),
        ("optim.momentum".to_owned(), meta.momentum.to_string()),
        ("optim.step_count".to_owned(), meta.step_count.to_string()),
    ]
}

pub fn parse_optim_meta(map: &HashMap<String, String>) -> Result<OptimMeta, String> {
    let kind = parse_optim_kind(parse_str(map, "optim.kind")?)?;
    Ok(OptimMeta {
        kind,
        lr: parse_f64(map, "optim.lr")?,
        beta1: parse_f64(map, "optim.beta1")?,
        beta2: parse_f64(map, "optim.beta2")?,
        eps: parse_f64(map, "optim.eps")?,
        weight_decay: parse_f64(map, "optim.weight_decay")?,
        momentum: parse_f64(map, "optim.momentum")?,
        step_count: parse_usize(map, "optim.step_count")?,
    })
}

pub fn optim_kind_str(kind: OptimKind) -> String {
    match kind {
        OptimKind::AdamW => "adamw".to_owned(),
        OptimKind::Adam => "adam".to_owned(),
        OptimKind::Sgd => "sgd".to_owned(),
    }
}

pub fn parse_optim_kind(s: &str) -> Result<OptimKind, String> {
    match s {
        "adamw" => Ok(OptimKind::AdamW),
        "adam" => Ok(OptimKind::Adam),
        "sgd" => Ok(OptimKind::Sgd),
        _ => Err(format!("bad optim.kind: {s}")),
    }
}

fn parse_f64(map: &HashMap<String, String>, key: &str) -> Result<f64, String> {
    let v = parse_str(map, key)?;
    v.parse::<f64>().map_err(|_| format!("bad {key}: {v}"))
}

fn parse_usize(map: &HashMap<String, String>, key: &str) -> Result<usize, String> {
    let v = parse_str(map, key)?;
    v.parse::<usize>().map_err(|_| format!("bad {key}: {v}"))
}

fn parse_str<'a>(map: &'a HashMap<String, String>, key: &str) -> Result<&'a str, String> {
    map.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing {key}"))
}

pub struct GradScalerState {
    pub scale: f64,
    pub growth_factor: f64,
    pub backoff_factor: f64,
    pub growth_interval: usize,
    pub consecutive_ok: usize,
}

impl GradScaler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            consecutive_ok: 0,
        }
    }

    #[must_use]
    pub fn scale_loss<T: Scalar, B: Backend>(&self, loss: &Tensor<T, B>) -> Tensor<T, B> {
        loss * T::from_f64(self.scale)
    }

    pub fn unscale_and_update<T: Scalar, B: Backend>(
        &mut self,
        grads: &mut [Tensor<T, B>],
    ) -> bool {
        let inv_scale = T::from_f64(1.0 / self.scale);
        let mut has_inf_nan = false;
        'outer: for g in grads.iter() {
            let (m, n) = g.shape();
            for r in 0..m {
                for c in 0..n {
                    let v = g.get(r, c).to_f64();
                    if !v.is_finite() {
                        has_inf_nan = true;
                        break 'outer;
                    }
                }
            }
        }
        if has_inf_nan {
            self.scale *= self.backoff_factor;
            self.consecutive_ok = 0;
            for g in grads.iter_mut() {
                let (m, n) = g.shape();
                *g = Tensor::zeros(m, n);
            }
            false
        } else {
            for g in grads.iter_mut() {
                *g = g.map(|x| x * inv_scale);
            }
            self.consecutive_ok += 1;
            if self.consecutive_ok >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_ok = 0;
            }
            true
        }
    }

    #[must_use]
    pub fn scale_factor(&self) -> f64 {
        self.scale
    }

    #[must_use]
    pub fn state(&self) -> GradScalerState {
        GradScalerState {
            scale: self.scale,
            growth_factor: self.growth_factor,
            backoff_factor: self.backoff_factor,
            growth_interval: self.growth_interval,
            consecutive_ok: self.consecutive_ok,
        }
    }

    pub fn load_state(&mut self, state: &GradScalerState) {
        self.scale = state.scale;
        self.growth_factor = state.growth_factor;
        self.backoff_factor = state.backoff_factor;
        self.growth_interval = state.growth_interval;
        self.consecutive_ok = state.consecutive_ok;
    }
}
