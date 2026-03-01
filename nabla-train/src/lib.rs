//! nabla-train — training stack for nabla.

#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic, missing_docs)]
#![allow(
    clippy::return_self_not_must_use,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::type_complexity,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::match_same_arms,
    clippy::write_with_newline,
    clippy::struct_excessive_bools,
    clippy::needless_pass_by_value,
    clippy::if_not_else,
    clippy::enum_glob_use,
    clippy::manual_div_ceil,
    clippy::new_without_default,
    clippy::needless_range_loop,
    clippy::len_without_is_empty,
    clippy::ref_as_ptr,
    clippy::collapsible_if,
    clippy::explicit_auto_deref,
    clippy::explicit_into_iter_loop,
    clippy::explicit_iter_loop,
    clippy::implicit_hasher,
    clippy::items_after_statements,
    clippy::iter_without_into_iter,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::match_wildcard_for_single_variants,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_pattern_matching,
    clippy::should_implement_trait,
    clippy::too_many_arguments,
    clippy::unnecessary_map_or,
    clippy::unused_self,
    clippy::while_let_on_iterator,
    clippy::assigning_clones,
    clippy::duplicated_attributes,
    clippy::if_same_then_else
)]

#[allow(missing_docs)]
pub mod benchmark;
#[allow(missing_docs)]
pub mod checkpoint;
#[allow(missing_docs)]
pub mod dataloader;
#[allow(missing_docs)]
pub mod dist;
#[allow(missing_docs)]
#[cfg(feature = "cpu")]
pub mod gguf;
#[allow(missing_docs)]
pub mod metrics;
#[allow(missing_docs)]
pub mod onnx;
#[allow(missing_docs)]
pub mod optim;
#[allow(missing_docs)]
pub mod profiler;
#[allow(missing_docs)]
pub mod quantize;
#[allow(missing_docs)]
pub mod trainer;

pub use nabla_ml as ml;

/// Absorbs the tape/backward/grad-collection/optimizer-step ceremony into one call.
///
/// Returns `Result<Tensor<T,B>>` containing the loss tensor data.
///
/// # Usage
/// ```ignore
/// let loss = train_step!(model, optimizer, tape, |x, out| {
///     mse_loss(out, &target)
/// })?;
/// ```
#[macro_export]
macro_rules! train_step {
    ($model:expr, $optimizer:expr, $tape:expr, |$x:ident, $out:ident| $loss_expr:expr) => {{
        let $x = $tape.variable($x.clone())?;
        let __result = $model.forward_var_tracked(&$x, &$tape)?;
        let $out = &__result.output;
        let __loss = $loss_expr;
        __loss.backward()?;
        let __grads: ::std::vec::Vec<_> = __result
            .param_vars
            .iter()
            .map(|v| v.grad())
            .collect::<::std::result::Result<_, _>>()?;
        let __grad_refs: ::std::vec::Vec<_> = __grads.iter().collect();
        $optimizer.step(&mut $model.parameters_mut(), &__grad_refs);
        ::std::result::Result::Ok(__loss.data().clone())
    }};
}

#[allow(missing_docs)]
pub mod prelude {
    pub use crate::benchmark::{
        AccuracyResult, BenchBatcher, BenchmarkDataset, BenchmarkReport, PerplexityResult,
        compute_accuracy, compute_perplexity, run_benchmark,
    };
    pub use crate::checkpoint::{
        CheckpointError, checkpoint_dir, load_checkpoint, save_checkpoint,
    };
    pub use crate::dataloader::{
        Batcher, DataLoader, Dataset, Sampler, Subset, VecBatcher, split_dataset,
    };
    pub use crate::dist::CpuAllReduce;
    #[cfg(feature = "cpu")]
    pub use crate::gguf::{
        GgufExportConfig, GgufQuantType, GgufTensor, GgufValue, ImportanceMatrix, MixingPreset,
        export_gguf, mixing_quant_type, quantize as gguf_quantize, write_gguf,
    };
    pub use crate::metrics::{JsonLogger, MovingAverage, StdoutLogger};
    pub use crate::onnx::{
        DimSpec, OnnxAttr, OnnxExporter, OnnxGraph, OnnxInitializer, OnnxModel, OnnxNode, OnnxOp,
        TensorSpec, batched_input, export_sequential, export_sequential_flat, image_input,
        nlp_input,
    };
    pub use crate::optim::{
        Adam, AdamW, GradScaler, GradScalerState, GroupOptimizer, LrSchedule, OptimKind, OptimMeta,
        OptimState, Optimizer, ParamExclusionPreset, ParamGroupConfig, ParamMatch, ParamSelector,
        ScheduleState, Sgd, adamw_step, lr_at_step,
    };
    pub use crate::profiler::{
        BottleneckKind, GpuTimer, HardwareSpec, KernelRecord, LayerStats, Profiler, RooflineResult,
        ThroughputStats, VramStats, matmul_flops, matmul_tflops, roofline,
    };
    pub use crate::quantize::{
        CalibrationStats, QuantizedWeight, dequant_matmul, dequantize, pack_int4, quantize_awq,
        quantize_awq_default, unpack_int4,
    };
    pub use crate::trainer::{
        EarlyStop, GradNanPolicy, HookAction, MetricStats, MetricsScope, TrainEvent, TrainHook,
        TrainState, TrainStepOut, Trainer, clip_grad_norm,
    };
    pub use nabla_ml::prelude::*;
}
