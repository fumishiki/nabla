use nabla_train::prelude::*;
use nabla_train::optim::{GroupOptimizer, ParamGroupConfig, ParamMatch, ParamSelector};
use nabla_train::ml as nabla;

struct GroupModule {
    weight: Tensor<f64>,
    bias: Tensor<f64>,
    training: bool,
}

impl GroupModule {
    fn new() -> Self {
        Self {
            weight: mat![[2.0_f64]],
            bias: mat![[3.0_f64]],
            training: true,
        }
    }
}

impl Module<f64, DefaultBackend> for GroupModule {
    fn forward(&self, x: &Tensor<f64, DefaultBackend>) -> Tensor<f64, DefaultBackend> {
        x.clone()
    }

    fn set_training(&mut self, training: bool) { self.training = training; }

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

#[test]
fn sgd_step_updates_param() {
    let mut param = mat![[1.0_f64]];
    let grad = mat![[1.0_f64]];
    let mut opt = Sgd::new(0.1, &[(1, 1)]);
    let mut params: Vec<&mut Tensor<f64>> = vec![&mut param];
    let grads: Vec<&Tensor<f64>> = vec![&grad];
    opt.step(&mut params, &grads);
    let v = param.get(0, 0);
    assert!((v - 0.9).abs() < 1e-10);
}

#[test]
fn adam_step_updates_param() {
    let mut param = mat![[1.0_f64]];
    let grad = mat![[1.0_f64]];
    let mut opt = Adam::new(0.1, &[(1, 1)]);
    let mut params: Vec<&mut Tensor<f64>> = vec![&mut param];
    let grads: Vec<&Tensor<f64>> = vec![&grad];
    opt.step(&mut params, &grads);
    let v = param.get(0, 0);
    assert!(v < 1.0);
}

#[test]
fn adamw_step_updates_param() {
    let mut param = mat![[1.0_f64]];
    let grad = mat![[1.0_f64]];
    let mut opt = AdamW::new(0.1, &[(1, 1)]);
    let mut params: Vec<&mut Tensor<f64>> = vec![&mut param];
    let grads: Vec<&Tensor<f64>> = vec![&grad];
    opt.step(&mut params, &grads);
    let v = param.get(0, 0);
    assert!(v < 1.0);
}

#[test]
fn linear_schedule_reaches_zero() {
    let sched = LrSchedule::Linear { warmup_steps: 0, total_steps: 10 };
    let lr0 = lr_at_step(&sched, 1.0, 0);
    let lr_end = lr_at_step(&sched, 1.0, 10);
    assert!((lr0 - 1.0).abs() < 1e-10);
    assert!(lr_end <= 1e-10);
}

#[test]
fn step_schedule_decays() {
    let sched = LrSchedule::Step { step_size: 2, gamma: 0.1 };
    let lr0 = lr_at_step(&sched, 1.0, 0);
    let lr2 = lr_at_step(&sched, 1.0, 2);
    let lr4 = lr_at_step(&sched, 1.0, 4);
    assert!((lr0 - 1.0).abs() < 1e-10);
    assert!((lr2 - 0.1).abs() < 1e-10);
    assert!((lr4 - 0.01).abs() < 1e-10);
}

#[test]
fn onecycle_reaches_max() {
    let sched = LrSchedule::OneCycle { max_lr: 1.0, total_steps: 10, pct_start: 0.5 };
    let lr_up = lr_at_step(&sched, 0.1, 3);
    assert!(lr_up > 0.1);
}

#[test]
fn grad_scaler_unscale() {
    let mut scaler = GradScaler::new();
    let mut grads: Vec<Tensor<f64, DefaultBackend>> = vec![mat![[2.0_f64]]];
    let ok = scaler.unscale_and_update(&mut grads);
    assert!(ok);
    let v = grads[0].get(0, 0);
    assert!(v < 2.0);
}

#[test]
fn grad_scaler_detects_inf() {
    let mut scaler = GradScaler::new();
    let mut grads: Vec<Tensor<f64, DefaultBackend>> = vec![mat![[f64::INFINITY]]];
    let ok = scaler.unscale_and_update(&mut grads);
    assert!(!ok);
    let v = grads[0].get(0, 0);
    assert!(v.abs() < 1e-12);
}

#[test]
fn group_optimizer_weight_decay_exclusion() {
    let mut module = GroupModule::new();
    let bias_group = ParamGroupConfig::sgd(
        "no_decay",
        ParamSelector::Any(vec![ParamMatch::Suffix("bias".to_owned())]),
        1.0,
        0.0,
        0.0,
    );
    let decay_group = ParamGroupConfig::sgd(
        "decay",
        ParamSelector::All,
        1.0,
        0.0,
        0.1,
    );
    let mut opt = match GroupOptimizer::from_module(&module, &[bias_group], Some(decay_group)) {
        Ok(v) => v,
        Err(e) => panic!("group optimizer build failed: {e}"),
    };

    let mut params = module.parameters_mut();
    let grads = [Tensor::zeros(1, 1), Tensor::zeros(1, 1)];
    let grad_refs: Vec<&Tensor<f64, DefaultBackend>> = grads.iter().collect();
    opt.step(&mut params, &grad_refs);

    let weight = module.weight.get(0, 0);
    let bias = module.bias.get(0, 0);
    assert!((weight - 1.8).abs() < 1e-10);
    assert!((bias - 3.0).abs() < 1e-10);
}
