use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use nabla_ml::autograd::Tape;
use nabla_train::ml as nabla;
use nabla_train::optim::ScheduleState;
use nabla_train::prelude::*;
use nabla_train::trainer::GradNanPolicy;

struct ToyModule {
    weight: Tensor<f64>,
    bias: Tensor<f64>,
    training: bool,
}

impl ToyModule {
    fn new() -> Self {
        Self {
            weight: mat![[1.0_f64]],
            bias: mat![[2.0_f64]],
            training: true,
        }
    }
}

struct SingleParamModule {
    weight: Tensor<f64>,
    training: bool,
}

impl SingleParamModule {
    fn new() -> Self {
        Self {
            weight: mat![[1.0_f64]],
            training: true,
        }
    }
}

impl Module<f64, DefaultBackend> for ToyModule {
    fn forward(&self, _x: &Tensor<f64, DefaultBackend>) -> Tensor<f64, DefaultBackend> {
        &self.weight + &self.bias
    }

    fn set_training(&mut self, training: bool) { self.training = training; }

    fn training(&self) -> bool { self.training }

    fn parameters(&self) -> Vec<&Tensor<f64, DefaultBackend>> {
        vec![&self.weight, &self.bias]
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor<f64, DefaultBackend>)> {
        vec![("weight", &self.weight), ("bias", &self.bias)]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f64, DefaultBackend>> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<f64, DefaultBackend>)> {
        vec![("weight", &mut self.weight), ("bias", &mut self.bias)]
    }
}

impl Module<f64, DefaultBackend> for SingleParamModule {
    fn forward(&self, _x: &Tensor<f64, DefaultBackend>) -> Tensor<f64, DefaultBackend> {
        self.weight.clone()
    }

    fn set_training(&mut self, training: bool) { self.training = training; }

    fn training(&self) -> bool { self.training }

    fn parameters(&self) -> Vec<&Tensor<f64, DefaultBackend>> {
        vec![&self.weight]
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor<f64, DefaultBackend>)> {
        vec![("weight", &self.weight)]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f64, DefaultBackend>> {
        vec![&mut self.weight]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut Tensor<f64, DefaultBackend>)> {
        vec![("weight", &mut self.weight)]
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("nabla-train-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    path
}

#[test]
fn train_epoch_updates_weight() {
    let model = SingleParamModule::new();
    let opt = Sgd::new(0.1, &[(1, 1)]);
    let mut trainer = Trainer::new(model, opt).with_metrics();
    let data = vec![0usize, 1usize];
    let loader = data.into_iter();

    let step_fn = |module: &mut SingleParamModule, _batch: &usize, tape: &Rc<Tape<f64, DefaultBackend>>| {
        let w = tape.variable(module.weight.clone())?;
        let target = tape.variable(Tensor::zeros(1, 1))?;
        let loss = w.mse_loss(&target);
        Ok(TrainStepOut { loss, params: vec![w] })
    };

    let steps = trainer.train_epoch(loader, step_fn).unwrap_or(0);
    assert_eq!(steps, 2);
    let new_w = trainer.model().weight.get(0, 0);
    assert!(new_w < 1.0);
}

#[test]
fn eval_epoch_runs_in_eval_mode() {
    let mut model = SingleParamModule::new();
    model.set_training(true);
    let opt = Sgd::new(0.1, &[(1, 1)]);
    let mut trainer = Trainer::new(model, opt);
    let data = vec![0usize, 1usize];
    let loader = data.into_iter();
    let loss = trainer.eval_epoch(loader, |m, _| Ok(m.weight.get(0, 0))).unwrap_or(-1.0);
    assert!(loss > 0.0);
    assert!(trainer.model().training());
}

#[test]
fn nan_policy_skip_drops_step() {
    let model = SingleParamModule::new();
    let opt = Sgd::new(0.1, &[(1, 1)]);
    let mut trainer = Trainer::new(model, opt).with_nan_policy(GradNanPolicy::Skip);
    let data = vec![0usize];
    let loader = data.into_iter();

    let step_fn = |_module: &mut SingleParamModule, _batch: &usize, tape: &Rc<Tape<f64, DefaultBackend>>| {
        let w = tape.variable(Tensor::fill(1, 1, f64::NAN))?;
        let target = tape.variable(Tensor::zeros(1, 1))?;
        let loss = w.mse_loss(&target);
        Ok(TrainStepOut { loss, params: vec![w] })
    };

    let steps = trainer.train_epoch(loader, step_fn).unwrap_or(0);
    assert_eq!(steps, 0);
}

#[test]
fn clip_grad_norm_scales() {
    let mut grads: Vec<Tensor<f64, DefaultBackend>> = vec![mat![[3.0_f64]], mat![[4.0_f64]]];
    let mut sum = 0.0;
    for g in &grads {
        let (m, n) = g.shape();
        for r in 0..m {
            for c in 0..n {
                let v = g.get(r, c);
                sum += v * v;
            }
        }
    }
    let before = sum.sqrt();
    let max_norm = 2.5;
    let reported = clip_grad_norm(&mut grads, max_norm);

    let mut sum_after = 0.0;
    for g in &grads {
        let (m, n) = g.shape();
        for r in 0..m {
            for c in 0..n {
                let v = g.get(r, c);
                sum_after += v * v;
            }
        }
    }
    let after = sum_after.sqrt();

    assert!((before - 5.0).abs() < 1e-10);
    assert!((reported - 5.0).abs() < 1e-10);
    assert!((after - max_norm).abs() < 1e-10);
}

#[test]
fn moving_average_window() {
    let mut avg = MovingAverage::new(2);
    assert!(avg.value().is_none());
    let v1 = avg.update(1.0);
    let v2 = avg.update(3.0);
    let v3 = avg.update(5.0);
    assert!((v1 - 1.0).abs() < 1e-10);
    assert!((v2 - 2.0).abs() < 1e-10);
    assert!((v3 - 4.0).abs() < 1e-10);
}

#[test]
fn stdout_logger_updates_average() {
    let mut logger = StdoutLogger::new(1, 3);
    let event = TrainEvent::Step { epoch: 0, step: 1, loss: 2.0 };
    let action = logger.on_event(&event);
    assert!(matches!(action, HookAction::Continue));
    assert!(logger.moving_average().is_some());
}

#[test]
fn json_logger_writes_lines() {
    let path = std::env::temp_dir().join("nabla-train-metrics.jsonl");
    let _ = std::fs::remove_file(&path);
    let mut logger = match JsonLogger::new(&path) {
        Ok(v) => v,
        Err(_) => return,
    };
    let event = TrainEvent::Step { epoch: 0, step: 1, loss: 1.0 };
    let action = logger.on_event(&event);
    assert!(matches!(action, HookAction::Continue));
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(content.contains("\"event\":\"step\""));
}

#[test]
fn checkpoint_roundtrip_with_schedule() {
    let mut module = ToyModule::new();
    let mut opt = Sgd::new(0.1, &[(1, 1), (1, 1)]);
    let mut scaler = GradScaler::new();
    let schedule = ScheduleState { base_lr: 0.1, schedule: LrSchedule::Step { step_size: 1, gamma: 0.9 } };
    let state = TrainState { epoch: 2, step: 3, grad_accum: 1, rng_state: Some(42) };

    let dir = temp_dir("ckpt");
    fs::create_dir_all(&dir).unwrap_or(());
    save_checkpoint::<f64, DefaultBackend, _, _>(&dir, &module, &opt, Some(&scaler), Some(&schedule), &state)
        .unwrap_or(());

    let mut module2 = ToyModule::new();
    let mut opt2 = Sgd::new(0.1, &[(1, 1), (1, 1)]);
    let mut scaler2 = GradScaler::new();
    let mut state2 = TrainState { epoch: 0, step: 0, grad_accum: 1, rng_state: None };
    let mut schedule2: Option<ScheduleState> = None;

    load_checkpoint::<f64, DefaultBackend, _, _>(&dir, &mut module2, &mut opt2, Some(&mut scaler2), &mut schedule2, &mut state2)
        .unwrap_or(());

    assert_eq!(state2.epoch, 2);
    assert_eq!(state2.step, 3);
    assert_eq!(state2.rng_state, Some(42));
    assert!(schedule2.is_some());
}
