pub mod alg;
pub mod core;
pub mod groups;

pub use core::{
    GradScaler, GradScalerState, LrSchedule, OptimKind, OptimMeta, OptimState, Optimizer,
    ScheduleState, lr_at_step, parse_schedule_state, schedule_pairs,
};
pub use alg::{adamw_step, Adam, AdamW, Sgd};
pub use groups::{GroupOptimizer, ParamExclusionPreset, ParamGroupConfig, ParamMatch, ParamSelector};
