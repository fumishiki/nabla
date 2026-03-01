use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::optim::{
    GradScaler, GradScalerState, OptimState, ScheduleState, parse_schedule_state, schedule_pairs,
};
use crate::trainer::TrainState;
use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_ml::module::{Module, StateError, load_tensors, save_tensors};

#[derive(Debug)]
pub enum CheckpointError {
    Io(io::Error),
    State(StateError),
    Optim(String),
    Parse(String),
}

impl From<io::Error> for CheckpointError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<StateError> for CheckpointError {
    fn from(e: StateError) -> Self {
        Self::State(e)
    }
}

const META_FILE: &str = "meta.txt";
const PARAMS_FILE: &str = "params.nbla";
const OPTIM_FILE: &str = "optim.nbla";

pub fn save_checkpoint<T, B, M, O>(
    dir: &Path,
    module: &M,
    optimizer: &O,
    scaler: Option<&GradScaler>,
    schedule: Option<&ScheduleState>,
    state: &TrainState,
) -> Result<(), CheckpointError>
where
    T: Scalar,
    B: Backend,
    M: Module<T, B>,
    O: OptimState<T, B>,
{
    fs::create_dir_all(dir)?;

    let params_path = dir.join(PARAMS_FILE);
    let params = module.state_dict();
    save_tensors::<T, B>(&params, &params_path)?;

    let optim_path = dir.join(OPTIM_FILE);
    let optim_tensors = optimizer.state_tensors();
    let optim_refs: Vec<(&str, &nabla_core::tensor::Tensor<T, B>)> =
        optim_tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    save_tensors::<T, B>(&optim_refs, &optim_path)?;

    let meta_path = dir.join(META_FILE);
    write_meta(&meta_path, optimizer, scaler, schedule, state)?;

    Ok(())
}

pub fn load_checkpoint<T, B, M, O>(
    dir: &Path,
    module: &mut M,
    optimizer: &mut O,
    scaler: Option<&mut GradScaler>,
    schedule: &mut Option<ScheduleState>,
    state: &mut TrainState,
) -> Result<(), CheckpointError>
where
    T: Scalar,
    B: Backend,
    M: Module<T, B>,
    O: OptimState<T, B>,
{
    let params_path = dir.join(PARAMS_FILE);
    let params = load_tensors::<T, B>(&params_path)?;
    let param_refs: Vec<(&str, &nabla_core::tensor::Tensor<T, B>)> =
        params.iter().map(|(k, v)| (k.as_str(), v)).collect();
    module.load_state_dict(&param_refs)?;

    let optim_path = dir.join(OPTIM_FILE);
    let optim_tensors = load_tensors::<T, B>(&optim_path)?;
    optimizer
        .load_state_tensors(&optim_tensors)
        .map_err(CheckpointError::Optim)?;

    let meta_path = dir.join(META_FILE);
    let (scaler_state, train_state, schedule_state) = read_meta(&meta_path, optimizer)?;
    if let (Some(s), Some(st)) = (scaler, scaler_state.as_ref()) {
        s.load_state(st);
    }
    *state = train_state;
    *schedule = schedule_state;

    Ok(())
}

fn write_meta<T, B, O>(
    path: &Path,
    optim: &O,
    scaler: Option<&GradScaler>,
    schedule: Option<&ScheduleState>,
    state: &TrainState,
) -> Result<(), CheckpointError>
where
    T: Scalar,
    B: Backend,
    O: OptimState<T, B>,
{
    let mut f = File::create(path)?;
    writeln!(f, "version=1")?;
    writeln!(f, "epoch={}", state.epoch)?;
    writeln!(f, "step={}", state.step)?;
    writeln!(f, "grad_accum={}", state.grad_accum)?;
    if let Some(rng) = state.rng_state {
        writeln!(f, "rng.state={rng}")?;
    }

    for (k, v) in optim.meta_pairs() {
        writeln!(f, "{k}={v}")?;
    }
    if let Some(schedule) = schedule {
        for (k, v) in schedule_pairs(schedule) {
            writeln!(f, "{k}={v}")?;
        }
    }

    if let Some(s) = scaler {
        let st = s.state();
        writeln!(f, "scaler.enabled=1")?;
        writeln!(f, "scaler.scale={}", st.scale)?;
        writeln!(f, "scaler.growth_factor={}", st.growth_factor)?;
        writeln!(f, "scaler.backoff_factor={}", st.backoff_factor)?;
        writeln!(f, "scaler.growth_interval={}", st.growth_interval)?;
        writeln!(f, "scaler.consecutive_ok={}", st.consecutive_ok)?;
    } else {
        writeln!(f, "scaler.enabled=0")?;
    }

    Ok(())
}

fn read_meta<T, B, O>(
    path: &Path,
    optim: &mut O,
) -> Result<(Option<GradScalerState>, TrainState, Option<ScheduleState>), CheckpointError>
where
    T: Scalar,
    B: Backend,
    O: OptimState<T, B>,
{
    let mut f = File::open(path)?;
    let mut buf = String::new();
    f.read_to_string(&mut buf)?;
    let mut map = HashMap::<String, String>::new();
    for line in buf.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| CheckpointError::Parse(line.to_owned()))?;
        map.insert(k.to_owned(), v.to_owned());
    }

    let epoch = parse_usize(&map, "epoch")?;
    let step = parse_usize(&map, "step")?;
    let grad_accum = parse_usize(&map, "grad_accum")?;

    optim
        .load_meta_pairs(&map)
        .map_err(CheckpointError::Optim)?;

    let scaler_enabled = parse_usize(&map, "scaler.enabled").unwrap_or(0) == 1;
    let scaler = if scaler_enabled {
        Some(GradScalerState {
            scale: parse_f64(&map, "scaler.scale")?,
            growth_factor: parse_f64(&map, "scaler.growth_factor")?,
            backoff_factor: parse_f64(&map, "scaler.backoff_factor")?,
            growth_interval: parse_usize(&map, "scaler.growth_interval")?,
            consecutive_ok: parse_usize(&map, "scaler.consecutive_ok")?,
        })
    } else {
        None
    };

    let rng_state = parse_usize(&map, "rng.state").ok().map(|v| v as u64);
    let schedule = parse_schedule_state(&map).map_err(CheckpointError::Optim)?;

    Ok((
        scaler,
        TrainState {
            epoch,
            step,
            grad_accum,
            rng_state,
        },
        schedule,
    ))
}

fn parse_f64(map: &HashMap<String, String>, key: &str) -> Result<f64, CheckpointError> {
    let v = parse_str(map, key)?;
    v.parse::<f64>()
        .map_err(|_| CheckpointError::Parse(format!("bad {key}: {v}")))
}

fn parse_usize(map: &HashMap<String, String>, key: &str) -> Result<usize, CheckpointError> {
    let v = parse_str(map, key)?;
    v.parse::<usize>()
        .map_err(|_| CheckpointError::Parse(format!("bad {key}: {v}")))
}

fn parse_str<'a>(map: &'a HashMap<String, String>, key: &str) -> Result<&'a str, CheckpointError> {
    map.get(key)
        .map(String::as_str)
        .ok_or_else(|| CheckpointError::Parse(format!("missing {key}")))
}

pub fn checkpoint_dir(base: &Path, name: &str) -> PathBuf {
    base.join(name)
}
