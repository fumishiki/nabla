#[cfg(feature = "llama")]
fn link_llama() {
    // llama.pc Libs already includes -lggml -lggml-base -lllama
    if let Err(e) = pkg_config::probe_library("llama") {
        eprintln!("error: failed to find libllama via pkg-config: {e}");
        eprintln!();
        eprintln!("  Install llama.cpp:");
        eprintln!("    macOS:  brew install llama.cpp");
        eprintln!("    Linux:  build from source and install to /usr/local");
        eprintln!();
        std::process::exit(1);
    }
}

fn main() {
    #[cfg(feature = "llama")]
    link_llama();
}
