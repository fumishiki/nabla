//! Profiling infrastructure for kernel timing, throughput, roofline analysis, and VRAM tracking.
//!
//! CPU timing via `std::time::Instant`; CUDA-aware sync via `cuda_synchronize` before measurement.
//! JSON export for integration with external analysis tools.

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

/// Hardware spec for roofline analysis (P6-PROF-05).
#[derive(Clone, Debug)]
pub struct HardwareSpec {
    /// Peak compute in TFLOPS (e.g. 495.0 for GH200 TF32).
    pub peak_tflops: f64,
    /// Peak memory bandwidth in TB/s (e.g. 4.0 for HBM3e).
    pub peak_bandwidth_tb_s: f64,
}

impl HardwareSpec {
    #[must_use]
    pub fn new(peak_tflops: f64, peak_bandwidth_tb_s: f64) -> Self {
        Self {
            peak_tflops,
            peak_bandwidth_tb_s,
        }
    }

    /// GH200 preset: 495 TFLOPS TF32, 4.0 TB/s HBM3e.
    #[must_use]
    pub fn gh200() -> Self {
        Self::new(495.0, 4.0)
    }

    /// A100 80GB preset: 312 TFLOPS TF32, 2.0 TB/s HBM2e.
    #[must_use]
    pub fn a100_80gb() -> Self {
        Self::new(312.0, 2.0)
    }

    /// H100 SXM preset: 989 TFLOPS TF32, 3.35 TB/s HBM3.
    #[must_use]
    pub fn h100_sxm() -> Self {
        Self::new(989.0, 3.35)
    }
}

/// Whether a kernel is compute-bound or memory-bound (P6-PROF-05).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BottleneckKind {
    Compute,
    Memory,
}

impl BottleneckKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Memory => "memory",
        }
    }
}

/// Roofline analysis result (P6-PROF-05).
#[derive(Clone, Debug)]
pub struct RooflineResult {
    pub arithmetic_intensity: f64,
    pub attainable_tflops: f64,
    pub bottleneck: BottleneckKind,
}

/// Computes roofline: min(peak_TFLOPS, AI * peak_BW) (P6-PROF-05).
#[must_use]
pub fn roofline(hw: &HardwareSpec, flops: f64, bytes_transferred: f64) -> RooflineResult {
    let ai = if bytes_transferred > 0.0 {
        flops / bytes_transferred
    } else {
        f64::INFINITY
    };
    let memory_roof = ai * hw.peak_bandwidth_tb_s * 1e12 / 1e12;
    let attainable = hw.peak_tflops.min(memory_roof);
    let bottleneck = if memory_roof < hw.peak_tflops {
        BottleneckKind::Memory
    } else {
        BottleneckKind::Compute
    };
    RooflineResult {
        arithmetic_intensity: ai,
        attainable_tflops: attainable,
        bottleneck,
    }
}

/// Per-kernel timing record (P6-PROF-01).
#[derive(Clone, Debug)]
pub struct KernelRecord {
    pub name: String,
    pub elapsed_ms: f64,
    pub flops: Option<f64>,
    pub bytes: Option<f64>,
}

impl KernelRecord {
    /// TFLOPS for this kernel (P6-PROF-04).
    #[must_use]
    pub fn tflops(&self) -> Option<f64> {
        self.flops.map(|f| f / (self.elapsed_ms * 1e-3) / 1e12)
    }
}

/// Per-layer profiling stats (P6-PROF-02).
#[derive(Clone, Debug)]
pub struct LayerStats {
    pub name: String,
    pub forward_ms: f64,
    pub backward_ms: f64,
    pub vram_bytes: u64,
    pub tflops: Option<f64>,
}

/// VRAM snapshot (P6-PROF-06).
#[derive(Clone, Debug, Default)]
pub struct VramStats {
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub per_layer: Vec<(String, u64)>,
}

/// Throughput stats (P6-PROF-03).
#[derive(Clone, Debug)]
pub struct ThroughputStats {
    pub tokens_per_sec: f64,
    pub ms_per_token: f64,
    pub batch_throughput: f64,
}

impl ThroughputStats {
    /// Compute throughput from token count, batch size, and elapsed wall time (P6-PROF-03).
    #[must_use]
    pub fn compute(total_tokens: u64, batch_size: u64, elapsed_ms: f64) -> Self {
        let elapsed_s = elapsed_ms / 1000.0;
        let tok_s = if elapsed_s > 0.0 {
            total_tokens as f64 / elapsed_s
        } else {
            0.0
        };
        let ms_tok = if total_tokens > 0 {
            elapsed_ms / total_tokens as f64
        } else {
            0.0
        };
        let batch_tp = if elapsed_s > 0.0 {
            batch_size as f64 / elapsed_s
        } else {
            0.0
        };
        Self {
            tokens_per_sec: tok_s,
            ms_per_token: ms_tok,
            batch_throughput: batch_tp,
        }
    }
}

/// Matmul FLOPS: 2*M*K*N (P6-PROF-04).
#[must_use]
pub fn matmul_flops(m: u64, k: u64, n: u64) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64
}

/// Matmul TFLOPS given dimensions and elapsed time (P6-PROF-04).
#[must_use]
pub fn matmul_tflops(m: u64, k: u64, n: u64, elapsed_ms: f64) -> f64 {
    matmul_flops(m, k, n) / (elapsed_ms * 1e-3) / 1e12
}

/// Timer that uses `cuda_synchronize` before sampling when CUDA is active (P6-PROF-01).
pub struct GpuTimer {
    start: Instant,
}

impl GpuTimer {
    /// Start timing. Calls `cuda_synchronize` first to drain pending GPU work.
    pub fn start() -> Self {
        #[cfg(feature = "cuda")]
        nabla_core::cuda_synchronize();
        Self {
            start: Instant::now(),
        }
    }

    /// Stop timing. Calls `cuda_synchronize` to ensure GPU work is complete, returns elapsed ms.
    pub fn stop(&self) -> f64 {
        #[cfg(feature = "cuda")]
        nabla_core::cuda_synchronize();
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

/// Main profiler collecting kernel records, layer stats, and VRAM (P6-PROF-01~07).
pub struct Profiler {
    kernels: Vec<KernelRecord>,
    layers: Vec<LayerStats>,
    vram: VramStats,
    hw: Option<HardwareSpec>,
    throughput_timer: Option<Instant>,
    throughput_tokens: u64,
    throughput_batch: u64,
}

impl Profiler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            layers: Vec::new(),
            vram: VramStats::default(),
            hw: None,
            throughput_timer: None,
            throughput_tokens: 0,
            throughput_batch: 0,
        }
    }

    #[must_use]
    pub fn with_hardware(mut self, hw: HardwareSpec) -> Self {
        self.hw = Some(hw);
        self
    }

    /// Record a kernel timing (P6-PROF-01).
    pub fn record_kernel(
        &mut self,
        name: impl Into<String>,
        elapsed_ms: f64,
        flops: Option<f64>,
        bytes: Option<f64>,
    ) {
        self.kernels.push(KernelRecord {
            name: name.into(),
            elapsed_ms,
            flops,
            bytes,
        });
    }

    /// Record per-layer stats (P6-PROF-02).
    pub fn record_layer(
        &mut self,
        name: impl Into<String>,
        forward_ms: f64,
        backward_ms: f64,
        vram_bytes: u64,
        tflops: Option<f64>,
    ) {
        self.layers.push(LayerStats {
            name: name.into(),
            forward_ms,
            backward_ms,
            vram_bytes,
            tflops,
        });
    }

    /// Update VRAM tracking (P6-PROF-06).
    pub fn update_vram(&mut self, current: u64, peak: u64) {
        self.vram.current_bytes = current;
        if peak > self.vram.peak_bytes {
            self.vram.peak_bytes = peak;
        }
    }

    /// Record per-layer VRAM (P6-PROF-06).
    pub fn record_layer_vram(&mut self, name: impl Into<String>, bytes: u64) {
        self.vram.per_layer.push((name.into(), bytes));
    }

    /// Start throughput measurement (P6-PROF-03).
    pub fn start_throughput(&mut self, batch_size: u64) {
        self.throughput_timer = Some(Instant::now());
        self.throughput_tokens = 0;
        self.throughput_batch = batch_size;
    }

    /// Accumulate tokens for throughput (P6-PROF-03).
    pub fn add_tokens(&mut self, n: u64) {
        self.throughput_tokens += n;
    }

    /// Finalize and return throughput stats (P6-PROF-03).
    #[must_use]
    pub fn throughput(&self) -> Option<ThroughputStats> {
        self.throughput_timer.map(|t| {
            ThroughputStats::compute(
                self.throughput_tokens,
                self.throughput_batch,
                t.elapsed().as_secs_f64() * 1000.0,
            )
        })
    }

    /// Roofline analysis for a specific kernel (P6-PROF-05).
    #[must_use]
    pub fn roofline_for(&self, kernel_idx: usize) -> Option<RooflineResult> {
        let hw = self.hw.as_ref()?;
        let k = self.kernels.get(kernel_idx)?;
        let flops = k.flops?;
        let bytes = k.bytes?;
        Some(roofline(hw, flops, bytes))
    }

    /// Per-kernel TFLOPS summary (P6-PROF-04).
    #[must_use]
    pub fn kernel_tflops(&self) -> Vec<(&str, Option<f64>)> {
        self.kernels
            .iter()
            .map(|k| (k.name.as_str(), k.tflops()))
            .collect()
    }

    /// Aggregate kernel time by name.
    #[must_use]
    pub fn kernel_breakdown(&self) -> Vec<(String, f64, usize)> {
        let mut map: HashMap<String, (f64, usize)> = HashMap::new();
        for k in &self.kernels {
            let e = map.entry(k.name.clone()).or_insert((0.0, 0));
            e.0 += k.elapsed_ms;
            e.1 += 1;
        }
        let mut out: Vec<_> = map
            .into_iter()
            .map(|(name, (ms, count))| (name, ms, count))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }

    pub fn kernels(&self) -> &[KernelRecord] {
        &self.kernels
    }
    pub fn layers(&self) -> &[LayerStats] {
        &self.layers
    }
    pub fn vram(&self) -> &VramStats {
        &self.vram
    }

    /// Export full profile as JSON string (P6-PROF-07).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("{\n");
        // Kernel breakdown
        s.push_str("  \"kernels\": [\n");
        for (i, k) in self.kernels.iter().enumerate() {
            let tflops = k.tflops().map_or("null".to_string(), |v| format!("{v:.4}"));
            use std::fmt::Write;
            let _ = write!(
                s,
                "    {{\"name\":\"{}\",\"elapsed_ms\":{:.4},\"flops\":{},\"bytes\":{},\"tflops\":{}}}",
                json_escape(&k.name),
                k.elapsed_ms,
                k.flops.map_or("null".to_string(), |v| format!("{v:.0}")),
                k.bytes.map_or("null".to_string(), |v| format!("{v:.0}")),
                tflops,
            );
            if i + 1 < self.kernels.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ],\n");
        // Layer stats
        s.push_str("  \"layers\": [\n");
        for (i, l) in self.layers.iter().enumerate() {
            use std::fmt::Write;
            let _ = write!(
                s,
                "    {{\"name\":\"{}\",\"forward_ms\":{:.4},\"backward_ms\":{:.4},\"vram_bytes\":{},\"tflops\":{}}}",
                json_escape(&l.name),
                l.forward_ms,
                l.backward_ms,
                l.vram_bytes,
                l.tflops.map_or("null".to_string(), |v| format!("{v:.4}")),
            );
            if i + 1 < self.layers.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ],\n");
        // Roofline per kernel
        s.push_str("  \"roofline\": [\n");
        let mut roofline_entries = Vec::new();
        for (i, k) in self.kernels.iter().enumerate() {
            if let Some(r) = self.roofline_for(i) {
                roofline_entries.push(format!(
                    "    {{\"kernel\":\"{}\",\"arithmetic_intensity\":{:.4},\"attainable_tflops\":{:.4},\"bottleneck\":\"{}\"}}",
                    json_escape(&k.name), r.arithmetic_intensity, r.attainable_tflops, r.bottleneck.as_str(),
                ));
            }
        }
        for (i, entry) in roofline_entries.iter().enumerate() {
            s.push_str(entry);
            if i + 1 < roofline_entries.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ],\n");
        // VRAM
        {
            use std::fmt::Write;
            let _ = write!(
                s,
                "  \"vram\": {{\"current_bytes\":{},\"peak_bytes\":{},\"per_layer\":[",
                self.vram.current_bytes, self.vram.peak_bytes,
            );
            for (i, (name, bytes)) in self.vram.per_layer.iter().enumerate() {
                let _ = write!(
                    s,
                    "{{\"name\":\"{}\",\"bytes\":{bytes}}}",
                    json_escape(name)
                );
                if i + 1 < self.vram.per_layer.len() {
                    s.push(',');
                }
            }
        }
        s.push_str("]},\n");
        // Throughput
        if let Some(tp) = self.throughput() {
            use std::fmt::Write;
            let _ = write!(
                s,
                "  \"throughput\": {{\"tokens_per_sec\":{:.2},\"ms_per_token\":{:.4},\"batch_throughput\":{:.2}}}\n",
                tp.tokens_per_sec, tp.ms_per_token, tp.batch_throughput,
            );
        } else {
            s.push_str("  \"throughput\": null\n");
        }
        s.push('}');
        s
    }

    /// Write JSON to a file (P6-PROF-07).
    pub fn write_json(&self, path: impl AsRef<std::path::Path>) -> Result<(), std::io::Error> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(self.to_json().as_bytes())
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}
