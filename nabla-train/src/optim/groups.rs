use std::collections::HashMap;

use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;
use nabla_ml::module::Module;

use crate::optim::alg::{Adam, AdamW, Sgd};
use crate::optim::core::{OptimKind, OptimMeta, OptimState, Optimizer, parse_optim_kind};

#[derive(Clone)]
pub enum ParamMatch {
    Equals(String),
    Prefix(String),
    Suffix(String),
    Contains(String),
}

#[derive(Clone)]
pub enum ParamSelector {
    All,
    Any(Vec<ParamMatch>),
}

impl ParamSelector {
    #[must_use]
    pub fn matches(&self, name: &str) -> bool {
        match self {
            ParamSelector::All => true,
            ParamSelector::Any(matches) => matches.iter().any(|m| match m {
                ParamMatch::Equals(v) => name == v,
                ParamMatch::Prefix(v) => name.starts_with(v),
                ParamMatch::Suffix(v) => name.ends_with(v),
                ParamMatch::Contains(v) => name.contains(v),
            }),
        }
    }
}

#[derive(Clone)]
pub struct ParamGroupConfig {
    pub name: String,
    pub selector: ParamSelector,
    pub kind: OptimKind,
    pub lr: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl ParamGroupConfig {
    #[must_use]
    pub fn adamw(
        name: impl Into<String>,
        selector: ParamSelector,
        lr: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            name: name.into(),
            selector,
            kind: OptimKind::AdamW,
            lr,
            weight_decay,
            momentum: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    #[must_use]
    pub fn adam(name: impl Into<String>, selector: ParamSelector, lr: f64) -> Self {
        Self {
            name: name.into(),
            selector,
            kind: OptimKind::Adam,
            lr,
            weight_decay: 0.0,
            momentum: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    #[must_use]
    pub fn sgd(
        name: impl Into<String>,
        selector: ParamSelector,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            name: name.into(),
            selector,
            kind: OptimKind::Sgd,
            lr,
            weight_decay,
            momentum,
            beta1: 0.0,
            beta2: 0.0,
            eps: 0.0,
        }
    }

    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    #[must_use]
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

pub enum ParamExclusionPreset {
    Bias,
    Norm,
    Embedding,
}

impl ParamExclusionPreset {
    fn patterns(&self) -> &'static [&'static str] {
        match self {
            ParamExclusionPreset::Bias => &["bias"],
            ParamExclusionPreset::Norm => &["norm", "bn", "ln"],
            ParamExclusionPreset::Embedding => &["embed", "embedding"],
        }
    }
}

pub struct GroupOptimizer<T: Scalar, B: Backend> {
    groups: Vec<GroupEntry<T, B>>,
    param_count: usize,
    base_lr: f64,
}

impl<T: Scalar, B: Backend> GroupOptimizer<T, B> {
    pub fn from_module<M: Module<T, B>>(
        module: &M,
        groups: &[ParamGroupConfig],
        default_group: Option<ParamGroupConfig>,
    ) -> Result<Self, String> {
        let named = module.named_parameters();
        Self::from_named_params(&named, groups, default_group)
    }

    pub fn from_named_params(
        named: &[(&str, &Tensor<T, B>)],
        groups: &[ParamGroupConfig],
        default_group: Option<ParamGroupConfig>,
    ) -> Result<Self, String> {
        let total = named.len();
        let mut assigned = vec![false; total];
        let mut buckets: Vec<(ParamGroupConfig, Vec<usize>)> = groups
            .iter()
            .cloned()
            .map(|cfg| (cfg, Vec::new()))
            .collect();
        let mut default_bucket = default_group.map(|cfg| (cfg, Vec::new()));

        for (i, (name, _)) in named.iter().enumerate() {
            let mut placed = false;
            for (cfg, idxs) in buckets.iter_mut() {
                if cfg.selector.matches(name) {
                    idxs.push(i);
                    assigned[i] = true;
                    placed = true;
                    break;
                }
            }
            if !placed {
                if let Some((_, idxs)) = default_bucket.as_mut() {
                    idxs.push(i);
                    assigned[i] = true;
                }
            }
        }

        if assigned.iter().any(|v| !*v) {
            return Err("nabla-train: group assignment incomplete".to_owned());
        }

        let mut group_entries = Vec::new();
        for (cfg, idxs) in buckets.into_iter() {
            if idxs.is_empty() {
                continue;
            }
            let shapes: Vec<(usize, usize)> = idxs.iter().map(|&i| named[i].1.shape()).collect();
            group_entries.push(GroupEntry::new(cfg, idxs, &shapes));
        }
        if let Some((cfg, idxs)) = default_bucket {
            if !idxs.is_empty() {
                let shapes: Vec<(usize, usize)> =
                    idxs.iter().map(|&i| named[i].1.shape()).collect();
                group_entries.push(GroupEntry::new(cfg, idxs, &shapes));
            }
        }

        let base_lr = group_entries.first().map(|g| g.base_lr).unwrap_or(1.0);
        Ok(Self {
            groups: group_entries,
            param_count: total,
            base_lr,
        })
    }

    pub fn adamw_with_decay_exclusions<M: Module<T, B>>(
        module: &M,
        lr: f64,
        weight_decay: f64,
        exclude: &[&str],
    ) -> Result<Self, String> {
        let mut matches = Vec::new();
        for s in exclude {
            matches.push(ParamMatch::Suffix((*s).to_owned()));
            matches.push(ParamMatch::Contains((*s).to_owned()));
        }
        let no_decay = ParamGroupConfig::adamw("no_decay", ParamSelector::Any(matches), lr, 0.0);
        let decay = ParamGroupConfig::adamw("decay", ParamSelector::All, lr, weight_decay);
        Self::from_module(module, &[no_decay], Some(decay))
    }

    pub fn adamw_with_decay_presets<M: Module<T, B>>(
        module: &M,
        lr: f64,
        weight_decay: f64,
        presets: &[ParamExclusionPreset],
    ) -> Result<Self, String> {
        let mut matches = Vec::new();
        for preset in presets {
            for pat in preset.patterns() {
                matches.push(ParamMatch::Suffix((*pat).to_owned()));
                matches.push(ParamMatch::Contains((*pat).to_owned()));
            }
        }
        let no_decay = ParamGroupConfig::adamw("no_decay", ParamSelector::Any(matches), lr, 0.0);
        let decay = ParamGroupConfig::adamw("decay", ParamSelector::All, lr, weight_decay);
        Self::from_module(module, &[no_decay], Some(decay))
    }

    #[must_use]
    pub fn groups(&self) -> usize {
        self.groups.len()
    }
}

impl<T: Scalar, B: Backend> Optimizer<T, B> for GroupOptimizer<T, B> {
    fn step(&mut self, params: &mut [&mut Tensor<T, B>], grads: &[&Tensor<T, B>]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "nabla-train: GroupOptimizer params/grads length mismatch"
        );
        assert_eq!(
            params.len(),
            self.param_count,
            "nabla-train: GroupOptimizer param count mismatch"
        );

        for group in &mut self.groups {
            let mut local_params: Vec<Tensor<T, B>> = Vec::with_capacity(group.indices.len());
            let mut local_grads: Vec<Tensor<T, B>> = Vec::with_capacity(group.indices.len());
            for &idx in &group.indices {
                local_params.push(params[idx].clone());
                local_grads.push(grads[idx].clone());
            }
            group.optimizer.step_slices(&mut local_params, &local_grads);
            for (local_idx, &param_idx) in group.indices.iter().enumerate() {
                *params[param_idx] = local_params[local_idx].clone();
            }
        }
    }

    fn reset(&mut self) {
        for group in &mut self.groups {
            group.optimizer.reset();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        if self.base_lr <= 0.0 {
            for group in &mut self.groups {
                group.optimizer.set_lr(lr);
            }
            return;
        }
        let scale = lr / self.base_lr;
        for group in &mut self.groups {
            group.optimizer.set_lr(group.base_lr * scale);
        }
    }
}

impl<T: Scalar, B: Backend> OptimState<T, B> for GroupOptimizer<T, B> {
    fn kind(&self) -> OptimKind {
        OptimKind::AdamW
    }

    fn state_tensors(&self) -> Vec<(String, Tensor<T, B>)> {
        let mut out = Vec::new();
        for (gi, group) in self.groups.iter().enumerate() {
            for (name, t) in group.optimizer.state_tensors() {
                out.push((format!("g{gi}.{name}"), t));
            }
        }
        out
    }

    fn load_state_tensors(&mut self, tensors: &[(String, Tensor<T, B>)]) -> Result<(), String> {
        let mut buckets: Vec<Vec<(String, Tensor<T, B>)>> = vec![Vec::new(); self.groups.len()];
        for (name, t) in tensors {
            let (g, rest) = name
                .split_once('.')
                .ok_or_else(|| format!("bad group tensor key: {name}"))?;
            let idx = g
                .strip_prefix('g')
                .ok_or_else(|| format!("bad group tensor key: {name}"))?;
            let gi = idx
                .parse::<usize>()
                .map_err(|_| format!("bad group index: {idx}"))?;
            if gi >= buckets.len() {
                return Err(format!("group index out of range: {gi}"));
            }
            buckets[gi].push((rest.to_owned(), t.clone()));
        }
        for (gi, group) in self.groups.iter_mut().enumerate() {
            group.optimizer.load_state_tensors(&buckets[gi])?;
        }
        Ok(())
    }

    fn meta(&self) -> OptimMeta {
        OptimMeta {
            kind: OptimKind::AdamW,
            lr: self.base_lr,
            beta1: 0.0,
            beta2: 0.0,
            eps: 0.0,
            weight_decay: 0.0,
            momentum: 0.0,
            step_count: 0,
        }
    }

    fn load_meta(&mut self, _meta: &OptimMeta) -> Result<(), String> {
        Ok(())
    }

    fn meta_pairs(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        out.push((
            "optim.group.count".to_owned(),
            self.groups.len().to_string(),
        ));
        for (gi, group) in self.groups.iter().enumerate() {
            out.push((format!("optim.group.{gi}.name"), group.name.clone()));
            let meta = group.optimizer.meta();
            out.push((format!("optim.group.{gi}.kind"), optim_kind_str(meta.kind)));
            out.push((format!("optim.group.{gi}.lr"), meta.lr.to_string()));
            out.push((format!("optim.group.{gi}.beta1"), meta.beta1.to_string()));
            out.push((format!("optim.group.{gi}.beta2"), meta.beta2.to_string()));
            out.push((format!("optim.group.{gi}.eps"), meta.eps.to_string()));
            out.push((
                format!("optim.group.{gi}.weight_decay"),
                meta.weight_decay.to_string(),
            ));
            out.push((
                format!("optim.group.{gi}.momentum"),
                meta.momentum.to_string(),
            ));
            out.push((
                format!("optim.group.{gi}.step_count"),
                meta.step_count.to_string(),
            ));
        }
        out
    }

    fn load_meta_pairs(&mut self, map: &HashMap<String, String>) -> Result<(), String> {
        let count = parse_group_usize(map, "optim.group.count")?;
        if count != self.groups.len() {
            return Err("optim.group.count mismatch".to_owned());
        }
        for (gi, group) in self.groups.iter_mut().enumerate() {
            let name = parse_group_str(map, gi, "name")?;
            if name != group.name {
                return Err(format!("optim.group.{gi}.name mismatch"));
            }
            let meta = OptimMeta {
                kind: parse_group_kind(map, gi)?,
                lr: parse_group_f64(map, gi, "lr")?,
                beta1: parse_group_f64(map, gi, "beta1")?,
                beta2: parse_group_f64(map, gi, "beta2")?,
                eps: parse_group_f64(map, gi, "eps")?,
                weight_decay: parse_group_f64(map, gi, "weight_decay")?,
                momentum: parse_group_f64(map, gi, "momentum")?,
                step_count: parse_group_usize(map, &format!("optim.group.{gi}.step_count"))?,
            };
            group.optimizer.load_meta(&meta)?;
        }
        if let Some(first) = self.groups.first() {
            self.base_lr = first.optimizer.meta().lr;
        }
        Ok(())
    }
}

struct GroupEntry<T: Scalar, B: Backend> {
    indices: Vec<usize>,
    name: String,
    base_lr: f64,
    optimizer: GroupOptim<T, B>,
}

impl<T: Scalar, B: Backend> GroupEntry<T, B> {
    fn new(cfg: ParamGroupConfig, indices: Vec<usize>, shapes: &[(usize, usize)]) -> Self {
        Self {
            indices,
            name: cfg.name.clone(),
            base_lr: cfg.lr,
            optimizer: GroupOptim::new(cfg, shapes),
        }
    }
}

enum GroupOptim<T: Scalar, B: Backend> {
    AdamW(AdamW<T, B>),
    Adam(Adam<T, B>),
    Sgd(Sgd<T, B>),
}

impl<T: Scalar, B: Backend> GroupOptim<T, B> {
    fn new(cfg: ParamGroupConfig, shapes: &[(usize, usize)]) -> Self {
        match cfg.kind {
            OptimKind::AdamW => {
                let opt = AdamW::new(cfg.lr, shapes)
                    .beta1(cfg.beta1)
                    .beta2(cfg.beta2)
                    .eps(cfg.eps)
                    .weight_decay(cfg.weight_decay);
                GroupOptim::AdamW(opt)
            }
            OptimKind::Adam => {
                let opt = Adam::new(cfg.lr, shapes)
                    .beta1(cfg.beta1)
                    .beta2(cfg.beta2)
                    .eps(cfg.eps);
                GroupOptim::Adam(opt)
            }
            OptimKind::Sgd => {
                let opt = Sgd::new(cfg.lr, shapes)
                    .momentum(cfg.momentum)
                    .weight_decay(cfg.weight_decay);
                GroupOptim::Sgd(opt)
            }
        }
    }

    fn step_slices(&mut self, params: &mut [Tensor<T, B>], grads: &[Tensor<T, B>]) {
        match self {
            GroupOptim::AdamW(opt) => opt.step_slices(params, grads),
            GroupOptim::Adam(opt) => opt.step_slices(params, grads),
            GroupOptim::Sgd(opt) => opt.step_slices(params, grads),
        }
    }

    fn reset(&mut self) {
        match self {
            GroupOptim::AdamW(opt) => opt.reset(),
            GroupOptim::Adam(opt) => opt.reset(),
            GroupOptim::Sgd(opt) => opt.reset(),
        }
    }

    fn set_lr(&mut self, lr: f64) {
        match self {
            GroupOptim::AdamW(opt) => opt.set_lr(lr),
            GroupOptim::Adam(opt) => opt.set_lr(lr),
            GroupOptim::Sgd(opt) => opt.set_lr(lr),
        }
    }

    fn meta(&self) -> OptimMeta {
        match self {
            GroupOptim::AdamW(opt) => opt.meta(),
            GroupOptim::Adam(opt) => opt.meta(),
            GroupOptim::Sgd(opt) => opt.meta(),
        }
    }

    fn load_meta(&mut self, meta: &OptimMeta) -> Result<(), String> {
        match self {
            GroupOptim::AdamW(opt) => opt.load_meta(meta),
            GroupOptim::Adam(opt) => opt.load_meta(meta),
            GroupOptim::Sgd(opt) => opt.load_meta(meta),
        }
    }

    fn state_tensors(&self) -> Vec<(String, Tensor<T, B>)> {
        match self {
            GroupOptim::AdamW(opt) => opt.state_tensors(),
            GroupOptim::Adam(opt) => opt.state_tensors(),
            GroupOptim::Sgd(opt) => opt.state_tensors(),
        }
    }

    fn load_state_tensors(&mut self, tensors: &[(String, Tensor<T, B>)]) -> Result<(), String> {
        match self {
            GroupOptim::AdamW(opt) => opt.load_state_tensors(tensors),
            GroupOptim::Adam(opt) => opt.load_state_tensors(tensors),
            GroupOptim::Sgd(opt) => opt.load_state_tensors(tensors),
        }
    }
}

fn optim_kind_str(kind: OptimKind) -> String {
    match kind {
        OptimKind::AdamW => "adamw".to_owned(),
        OptimKind::Adam => "adam".to_owned(),
        OptimKind::Sgd => "sgd".to_owned(),
    }
}

fn parse_group_kind(map: &HashMap<String, String>, idx: usize) -> Result<OptimKind, String> {
    let key = format!("optim.group.{idx}.kind");
    let s = map
        .get(&key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing {key}"))?;
    parse_optim_kind(s)
}

fn parse_group_str(
    map: &HashMap<String, String>,
    idx: usize,
    field: &str,
) -> Result<String, String> {
    let key = format!("optim.group.{idx}.{field}");
    map.get(&key)
        .cloned()
        .ok_or_else(|| format!("missing {key}"))
}

fn parse_group_f64(map: &HashMap<String, String>, idx: usize, field: &str) -> Result<f64, String> {
    let key = format!("optim.group.{idx}.{field}");
    let s = map
        .get(&key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing {key}"))?;
    s.parse::<f64>().map_err(|_| format!("bad {key}: {s}"))
}

fn parse_group_usize(map: &HashMap<String, String>, key: &str) -> Result<usize, String> {
    let s = map
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing {key}"))?;
    s.parse::<usize>().map_err(|_| format!("bad {key}: {s}"))
}
