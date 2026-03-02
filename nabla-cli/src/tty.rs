//! Terminal color helpers and animated spinner (pure std, no deps).
//!
//! Colors are suppressed when `NO_COLOR` env var is set or `TERM=dumb`.

use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

static USE_COLOR: OnceLock<bool> = OnceLock::new();

/// True when the terminal supports ANSI color (respects NO_COLOR / TERM=dumb, checks isatty).
pub fn use_color() -> bool {
    *USE_COLOR.get_or_init(|| {
        std::env::var_os("NO_COLOR").is_none()
            && std::env::var("TERM").map(|t| t != "dumb").unwrap_or(true)
            && is_tty(1)
    })
}

#[cfg(unix)]
fn is_tty(fd: i32) -> bool {
    unsafe extern "C" { fn isatty(fd: i32) -> i32; }
    unsafe { isatty(fd) != 0 }
}

#[cfg(not(unix))]
fn is_tty(_fd: i32) -> bool { false }

macro_rules! paint {
    ($name:ident, $code:literal) => {
        pub fn $name(s: &str) -> String {
            if use_color() { format!(concat!("\x1b[", $code, "m{}\x1b[0m"), s) }
            else { s.to_owned() }
        }
    };
}

paint!(bold,   "1");
paint!(dim,    "2");
paint!(cyan,   "36");
paint!(green,  "32");
paint!(yellow, "33");

// ---------------------------------------------------------------------------
// Animated spinner (background thread → stderr)
// ---------------------------------------------------------------------------

const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub struct Spinner {
    stop:   Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    /// Start spinner with `msg` label; animates until dropped or `finish()` called.
    pub fn new(msg: &str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        if !use_color() {
            return Self { stop, handle: None };
        }
        let stop2 = Arc::clone(&stop);
        let msg   = msg.to_owned();
        let handle = thread::spawn(move || {
            let mut stderr = std::io::stderr();
            let clear = " ".repeat(msg.len() + 8);
            let mut i = 0usize;
            while !stop2.load(Ordering::Relaxed) {
                let _ = write!(stderr, "\r  \x1b[36m{}\x1b[0m  {}", FRAMES[i % FRAMES.len()], msg);
                let _ = stderr.flush();
                thread::sleep(Duration::from_millis(80));
                i += 1;
            }
            let _ = write!(stderr, "\r{}\r", clear);
            let _ = stderr.flush();
        });
        Self { stop, handle: Some(handle) }
    }

    /// Stop spinner and print `✓ msg` to stderr.
    pub fn finish(mut self, msg: &str) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() { let _ = h.join(); }
        if use_color() { eprintln!("  \x1b[32m✓\x1b[0m  {msg}"); }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() { let _ = h.join(); }
    }
}
