use std::fmt::Write as FmtWrite;

use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;
use nabla_core::tensor::Tensor;

use crate::dataloader::{Batcher, DataLoader, Dataset};

// -- P6-BENCH-01: Dataset loader via generic DataLoader --

/// A benchmark dataset holding (input, target_label) pairs.
pub struct BenchmarkDataset<T: Scalar, B: Backend> {
    inputs: Vec<Tensor<T, B>>,
    labels: Vec<usize>,
}

impl<T: Scalar, B: Backend> BenchmarkDataset<T, B> {
    #[must_use]
    pub fn new(inputs: Vec<Tensor<T, B>>, labels: Vec<usize>) -> Self {
        assert_eq!(inputs.len(), labels.len(), "nabla-bench: inputs/labels length mismatch");
        Self { inputs, labels }
    }

    #[must_use]
    pub fn len(&self) -> usize { self.inputs.len() }
}

impl<T: Scalar, B: Backend> Dataset for BenchmarkDataset<T, B> {
    type Item = (Tensor<T, B>, usize);
    fn len(&self) -> usize { self.inputs.len() }
    fn get(&self, idx: usize) -> Self::Item { (self.inputs[idx].clone(), self.labels[idx]) }
}

/// Batcher that stacks benchmark samples into (batched_input, label_vec).
#[derive(Clone, Copy)]
pub struct BenchBatcher;

impl<T: Scalar, B: Backend> Batcher<(Tensor<T, B>, usize)> for BenchBatcher {
    type Output = (Vec<Tensor<T, B>>, Vec<usize>);
    fn batch(&self, items: Vec<(Tensor<T, B>, usize)>) -> Self::Output {
        let (inputs, labels): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        (inputs, labels)
    }
}

// -- P6-BENCH-02: Perplexity measurement --

/// Per-sample perplexity score.
pub struct PerplexityResult {
    pub per_sample_loss: Vec<f64>,
    pub mean_loss: f64,
    pub perplexity: f64,
}

/// Compute perplexity over a dataset using a model's forward pass.
///
/// `forward_fn` takes an input tensor and returns logits `(1, vocab_size)`.
/// The target label is used to extract the cross-entropy loss for that token.
pub fn compute_perplexity<T, B, F>(
    loader: &DataLoader<BenchmarkDataset<T, B>, BenchBatcher>,
    forward_fn: &F,
    num_classes: usize,
) -> PerplexityResult
where
    T: Scalar,
    B: Backend,
    F: Fn(&Tensor<T, B>) -> Tensor<T, B>,
{
    let mut losses = Vec::new();
    for (inputs, labels) in loader.iter() {
        for (input, label) in inputs.iter().zip(labels.iter()) {
            let logits = forward_fn(input);
            let log_sm = logits.log_softmax(1);
            let (_, nc) = log_sm.shape();
            let nc = nc.min(num_classes);
            let nll = if *label < nc { -log_sm.get(0, *label).to_f64() } else { 0.0 };
            losses.push(nll);
        }
    }
    let n = losses.len().max(1) as f64;
    let mean_loss = losses.iter().sum::<f64>() / n;
    let perplexity = mean_loss.exp();
    PerplexityResult { per_sample_loss: losses, mean_loss, perplexity }
}

// -- P6-BENCH-03: Accuracy / Top-k accuracy --

/// Accuracy evaluation results.
pub struct AccuracyResult {
    pub correct: usize,
    pub total: usize,
    pub accuracy: f64,
    pub topk_correct: usize,
    pub topk_accuracy: f64,
    pub k: usize,
    pub per_sample_correct: Vec<bool>,
    pub per_sample_topk_correct: Vec<bool>,
}

/// Compute accuracy and top-k accuracy over a dataset.
///
/// `forward_fn` takes an input tensor and returns logits `(1, num_classes)`.
pub fn compute_accuracy<T, B, F>(
    loader: &DataLoader<BenchmarkDataset<T, B>, BenchBatcher>,
    forward_fn: &F,
    k: usize,
) -> AccuracyResult
where
    T: Scalar,
    B: Backend,
    F: Fn(&Tensor<T, B>) -> Tensor<T, B>,
{
    let k = k.max(1);
    let mut correct = 0usize;
    let mut topk_correct = 0usize;
    let mut total = 0usize;
    let mut per_correct = Vec::new();
    let mut per_topk = Vec::new();
    for (inputs, labels) in loader.iter() {
        for (input, label) in inputs.iter().zip(labels.iter()) {
            let logits = forward_fn(input);
            let (_, nc) = logits.shape();
            let pred = logits.argmax().1;
            let hit = pred == *label;
            correct += hit as usize;
            per_correct.push(hit);
            let actual_k = k.min(nc);
            let (_, topk_indices) = logits.topk(actual_k, 1);
            let in_topk = (0..actual_k).any(|j| topk_indices.get(0, j).to_f64() as usize == *label);
            topk_correct += in_topk as usize;
            per_topk.push(in_topk);
            total += 1;
        }
    }
    let t = total.max(1) as f64;
    AccuracyResult {
        correct, total, accuracy: correct as f64 / t,
        topk_correct, topk_accuracy: topk_correct as f64 / t,
        k, per_sample_correct: per_correct, per_sample_topk_correct: per_topk,
    }
}

// -- P6-BENCH-04: JSON output --

/// Full benchmark report combining perplexity and accuracy.
pub struct BenchmarkReport {
    pub perplexity: PerplexityResult,
    pub accuracy: AccuracyResult,
}

impl BenchmarkReport {
    #[must_use]
    pub fn new(perplexity: PerplexityResult, accuracy: AccuracyResult) -> Self {
        Self { perplexity, accuracy }
    }

    /// Serialize the report to a JSON string (no serde dependency).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("{\n");
        // summary
        s.push_str("  \"summary\": {\n");
        let _ = write!(s, "    \"perplexity\": {:.6},\n", self.perplexity.perplexity);
        let _ = write!(s, "    \"mean_cross_entropy\": {:.6},\n", self.perplexity.mean_loss);
        let _ = write!(s, "    \"accuracy\": {:.6},\n", self.accuracy.accuracy);
        let _ = write!(s, "    \"top{}_accuracy\": {:.6},\n", self.accuracy.k, self.accuracy.topk_accuracy);
        let _ = write!(s, "    \"total_samples\": {},\n", self.accuracy.total);
        let _ = write!(s, "    \"correct\": {},\n", self.accuracy.correct);
        let _ = write!(s, "    \"topk_correct\": {}\n", self.accuracy.topk_correct);
        s.push_str("  },\n");
        // per-sample
        s.push_str("  \"per_sample\": [\n");
        let n = self.perplexity.per_sample_loss.len();
        for i in 0..n {
            let loss = self.perplexity.per_sample_loss[i];
            let correct = self.accuracy.per_sample_correct.get(i).copied().unwrap_or(false);
            let topk = self.accuracy.per_sample_topk_correct.get(i).copied().unwrap_or(false);
            let _ = write!(s, "    {{\"idx\": {i}, \"nll\": {loss:.6}, \"correct\": {correct}, \"topk_correct\": {topk}}}");
            if i + 1 < n { s.push(','); }
            s.push('\n');
        }
        s.push_str("  ]\n");
        s.push('}');
        s
    }

    /// Write the JSON report to a file.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

/// Run a full benchmark evaluation (perplexity + accuracy + top-k) and produce a report.
pub fn run_benchmark<T, B, F>(
    loader: &DataLoader<BenchmarkDataset<T, B>, BenchBatcher>,
    forward_fn: &F,
    num_classes: usize,
    k: usize,
) -> BenchmarkReport
where
    T: Scalar,
    B: Backend,
    F: Fn(&Tensor<T, B>) -> Tensor<T, B>,
{
    let perplexity = compute_perplexity(loader, forward_fn, num_classes);
    let accuracy = compute_accuracy(loader, forward_fn, k);
    BenchmarkReport::new(perplexity, accuracy)
}
