//! `nabla serve` — OpenAI-compatible HTTP inference server.
//!
//! Usage: nabla serve <MODEL_PATH> [--port 8080] [--ctx 2048] [--temp 0.8]
//!                   [--max-tokens 512] [--host 0.0.0.0]
//!
//! Endpoints:
//!   GET  /health                        — liveness probe
//!   GET  /v1/models                     — list loaded model
//!   POST /v1/chat/completions           — OpenAI-compatible chat (non-streaming)
//!   POST /v1/completions                — OpenAI-compatible legacy completions
//!
//! Requires `--features llama`.

#[cfg(feature = "llama")]
pub fn run(args: &[String]) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use nabla_interface::{InferenceConfig, InferenceEngine, SamplingConfig};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::Path;
    use std::sync::Arc;

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: nabla serve <MODEL_PATH> [OPTIONS]\n\n\
            Options:\n  \
            --port       <N>    Listen port (default: 8080)\n  \
            --host       <ADDR> Listen address (default: 0.0.0.0)\n  \
            --ctx        <N>    Context length (default: 2048)\n  \
            --temp       <F>    Sampling temperature (default: 0.8)\n  \
            --max-tokens <N>    Max output tokens (default: 512)\n\n\
            Endpoints:\n  \
            GET  /health\n  \
            GET  /v1/models\n  \
            POST /v1/completions\n  \
            POST /v1/chat/completions"
        );
        return Ok(());
    }

    let model_path = args
        .first()
        .filter(|a| !a.starts_with('-'))
        .ok_or("missing MODEL_PATH — usage: nabla serve <path>")?;

    let port       = flag_usize(args, "--port").unwrap_or(8080);
    let host       = flag_str(args, "--host").unwrap_or("0.0.0.0");
    let ctx        = flag_usize(args, "--ctx").unwrap_or(2048);
    let temp       = flag_f64(args, "--temp").unwrap_or(0.8);
    let max_tokens = flag_usize(args, "--max-tokens").unwrap_or(512);

    let model_name = Path::new(model_path)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    eprintln!("Loading model: {model_path}");
    let engine = Arc::new(InferenceEngine::new(
        model_path,
        InferenceConfig { context_size: ctx, ..Default::default() },
    )?);

    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)?;
    eprintln!("nabla serve  listening on http://{addr}");
    eprintln!("Model: {model_name}  ctx={ctx}  temp={temp}  max_tokens={max_tokens}");

    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let engine = Arc::clone(&engine);

        let sampling = SamplingConfig { temperature: temp, ..Default::default() };
        let max_tok  = max_tokens;
        let mname    = model_name.clone();

        // Handle request in the same thread (single-request-at-a-time; sufficient for local use).
        if let Err(e) = handle_request(&mut stream, &engine, &sampling, max_tok, &mname) {
            eprintln!("serve: request error: {e}");
        }
    }
    Ok(())
}

#[cfg(feature = "llama")]
fn handle_request(
    stream: &mut std::net::TcpStream,
    engine: &nabla_interface::InferenceEngine,
    sampling: &nabla_interface::SamplingConfig,
    max_tokens: usize,
    model_name: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::{BufRead, BufReader, Write};

    let mut reader = BufReader::new(stream.try_clone()?);

    // Read request line.
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 { return Ok(()); }
    let method = parts[0];
    let path   = parts[1];

    // Read headers.
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim();
        if line.is_empty() { break; }
        if let Some(rest) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }

    // Read body.
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    let body_str = String::from_utf8_lossy(&body);

    let response = match (method, path) {
        ("GET", "/health") => {
            json_response(200, r#"{"status":"ok"}"#)
        }
        ("GET", "/v1/models") => {
            let payload = format!(
                r#"{{"object":"list","data":[{{"id":{:?},"object":"model","owned_by":"nabla"}}]}}"#,
                model_name
            );
            json_response(200, &payload)
        }
        ("POST", "/v1/completions") => {
            let prompt = extract_field(&body_str, "prompt").unwrap_or_default();
            let max_tok = extract_usize(&body_str, "max_tokens").unwrap_or(max_tokens);
            match generate_completion(engine, &prompt, max_tok, sampling) {
                Ok(text) => {
                    let payload = format!(
                        r#"{{"id":"cmpl-nabla","object":"text_completion","model":{:?},\
"choices":[{{"text":{:?},"index":0,"finish_reason":"stop"}}]}}"#,
                        model_name, text
                    );
                    json_response(200, &payload)
                }
                Err(e) => json_response(500, &format!(r#"{{"error":{:?}}}"#, e.to_string())),
            }
        }
        ("POST", "/v1/chat/completions") => {
            let prompt = extract_last_user_message(&body_str).unwrap_or_default();
            let max_tok = extract_usize(&body_str, "max_tokens").unwrap_or(max_tokens);
            match generate_completion(engine, &prompt, max_tok, sampling) {
                Ok(text) => {
                    let payload = format!(
                        r#"{{"id":"chatcmpl-nabla","object":"chat.completion","model":{:?},\
"choices":[{{"message":{{"role":"assistant","content":{:?}}},"index":0,"finish_reason":"stop"}}]}}"#,
                        model_name, text
                    );
                    json_response(200, &payload)
                }
                Err(e) => json_response(500, &format!(r#"{{"error":{:?}}}"#, e.to_string())),
            }
        }
        _ => json_response(404, r#"{"error":"not found"}"#),
    };

    stream.write_all(response.as_bytes())?;
    Ok(())
}

#[cfg(feature = "llama")]
fn generate_completion(
    engine: &nabla_interface::InferenceEngine,
    prompt: &str,
    max_tokens: usize,
    sampling: &nabla_interface::SamplingConfig,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let stream = engine.generate_stream(prompt, max_tokens, sampling)?;
    let text: String = stream.collect();
    Ok(text)
}

#[cfg(feature = "llama")]
fn json_response(status: u16, body: &str) -> String {
    let reason = match status {
        200 => "OK", 404 => "Not Found", 500 => "Internal Server Error", _ => "Unknown",
    };
    format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        len = body.len()
    )
}

// Minimal JSON field extractors (avoids adding a JSON dep for the server).
#[cfg(feature = "llama")]
fn extract_field<'a>(json: &'a str, key: &str) -> Option<String> {
    let needle = format!(r#""{key}""#);
    let start = json.find(&needle)?;
    let after_colon = json[start + needle.len()..].trim_start_matches([' ', ':']);
    if after_colon.starts_with('"') {
        let inner = &after_colon[1..];
        let end = find_json_string_end(inner)?;
        Some(unescape_json(&inner[..end]))
    } else {
        None
    }
}

#[cfg(feature = "llama")]
fn find_json_string_end(s: &str) -> Option<usize> {
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if escaped { escaped = false; continue; }
        if c == '\\' { escaped = true; continue; }
        if c == '"' { return Some(i); }
    }
    None
}

#[cfg(feature = "llama")]
fn unescape_json(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"").replace("\\\\", "\\")
}

#[cfg(feature = "llama")]
fn extract_usize(json: &str, key: &str) -> Option<usize> {
    let needle = format!(r#""{key}":"#);
    let start = json.find(&needle)?;
    let rest = json[start + needle.len()..].trim_start();
    rest.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

// Extract the last user message content from a chat completions request body.
#[cfg(feature = "llama")]
fn extract_last_user_message(json: &str) -> Option<String> {
    // Find all "role":"user" occurrences and return the content of the last one.
    let mut last_user_content = None;
    let mut search = json;
    while let Some(role_pos) = search.find(r#""role":"user""#) {
        let after_role = &search[role_pos..];
        if let Some(content) = extract_field(after_role, "content") {
            last_user_content = Some(content);
        }
        search = &search[role_pos + 1..];
    }
    last_user_content
}

// ---------------------------------------------------------------------------
// Stub for non-llama builds (graceful error message)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "llama"))]
pub fn run(_args: &[String]) -> std::result::Result<(), Box<dyn std::error::Error>> {
    Err("`nabla serve` requires `--features llama` (link llama.cpp)".into())
}

// ---------------------------------------------------------------------------
// Arg helpers
// ---------------------------------------------------------------------------

fn flag_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].as_str())
}

fn flag_usize(args: &[String], flag: &str) -> Option<usize> {
    flag_str(args, flag)?.parse().ok()
}

fn flag_f64(args: &[String], flag: &str) -> Option<f64> {
    flag_str(args, flag)?.parse().ok()
}
