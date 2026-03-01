use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::trainer::{HookAction, TrainEvent, TrainHook};

pub struct MovingAverage {
    window: usize,
    buf: VecDeque<f64>,
    sum: f64,
}

impl MovingAverage {
    #[must_use]
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            buf: VecDeque::new(),
            sum: 0.0,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.buf.push_back(value);
        self.sum += value;
        if self.buf.len() > self.window {
            if let Some(v) = self.buf.pop_front() {
                self.sum -= v;
            }
        }
        self.value().unwrap_or(value)
    }

    #[must_use]
    pub fn value(&self) -> Option<f64> {
        if self.buf.is_empty() {
            None
        } else {
            Some(self.sum / self.buf.len() as f64)
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }
}

pub struct StdoutLogger {
    every: usize,
    avg: MovingAverage,
}

impl StdoutLogger {
    #[must_use]
    pub fn new(every: usize, window: usize) -> Self {
        Self {
            every: every.max(1),
            avg: MovingAverage::new(window),
        }
    }

    #[must_use]
    pub fn moving_average(&self) -> Option<f64> {
        self.avg.value()
    }
}

impl TrainHook for StdoutLogger {
    fn on_event(&mut self, event: &TrainEvent) -> HookAction {
        match event {
            TrainEvent::Step { epoch, step, loss } => {
                let avg = self.avg.update(*loss);
                if step % self.every == 0 {
                    println!("epoch={epoch} step={step} loss={loss:.6} avg={avg:.6}");
                }
            }
            TrainEvent::EpochEnd { epoch, steps } => {
                if let Some(avg) = self.avg.value() {
                    println!("epoch={epoch} steps={steps} avg_loss={avg:.6}");
                }
            }
            TrainEvent::EvalStep { epoch, step, loss } => {
                let avg = self.avg.update(*loss);
                if step % self.every == 0 {
                    println!("eval epoch={epoch} step={step} loss={loss:.6} avg={avg:.6}");
                }
            }
            TrainEvent::EvalEnd {
                epoch,
                steps,
                avg_loss,
            } => {
                println!("eval epoch={epoch} steps={steps} avg_loss={avg_loss:.6}");
            }
        }
        HookAction::Continue
    }
}

pub struct JsonLogger {
    writer: BufWriter<std::fs::File>,
    failed: bool,
}

impl JsonLogger {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("json logger open failed: {e}"))?;
        Ok(Self {
            writer: BufWriter::new(file),
            failed: false,
        })
    }
}

impl TrainHook for JsonLogger {
    fn on_event(&mut self, event: &TrainEvent) -> HookAction {
        if self.failed {
            return HookAction::Stop;
        }
        let line = match event {
            TrainEvent::Step { epoch, step, loss } => {
                format!(
                    "{{\"event\":\"step\",\"epoch\":{epoch},\"step\":{step},\"loss\":{loss}}}\n"
                )
            }
            TrainEvent::EpochEnd { epoch, steps } => {
                format!("{{\"event\":\"epoch_end\",\"epoch\":{epoch},\"steps\":{steps}}}\n")
            }
            TrainEvent::EvalStep { epoch, step, loss } => {
                format!(
                    "{{\"event\":\"eval_step\",\"epoch\":{epoch},\"step\":{step},\"loss\":{loss}}}\n"
                )
            }
            TrainEvent::EvalEnd {
                epoch,
                steps,
                avg_loss,
            } => {
                format!(
                    "{{\"event\":\"eval_end\",\"epoch\":{epoch},\"steps\":{steps},\"avg_loss\":{avg_loss}}}\n"
                )
            }
        };
        if let Err(_) = self.writer.write_all(line.as_bytes()) {
            self.failed = true;
            return HookAction::Stop;
        }
        if let Err(_) = self.writer.flush() {
            self.failed = true;
            return HookAction::Stop;
        }
        HookAction::Continue
    }
}
