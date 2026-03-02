//! `nabla info` — hardware diagnostics.
//!
//! Probes available backends in order: CUDA → HIP → wgpu → CPU.
//! Outputs a human-readable table or `--json`.

use std::error::Error;

pub fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: nabla info [--json]\n\nDetect available backends and display device info.");
        return Ok(());
    }
    let json = args.iter().any(|a| a == "--json");

    let mut entries: Vec<InfoEntry> = Vec::new();

    #[cfg(feature = "cuda")]
    entries.extend(probe_cuda()?);

    #[cfg(feature = "hip")]
    entries.extend(probe_hip()?);

    #[cfg(feature = "wgpu")]
    entries.extend(probe_wgpu());

    entries.push(probe_cpu());

    if json { print_json(&entries); } else { print_table(&entries); }

    // Exit 1 if no GPU found (CLI-INFO-06).
    if !entries.iter().any(|e| e.kind != "CPU") {
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct InfoEntry {
    kind: &'static str,
    name: String,
    mem_total_mib: Option<u64>,
    mem_free_mib: Option<u64>,
    extra: Vec<(&'static str, String)>,
}

// ---------------------------------------------------------------------------
// CUDA probing (only compiled when cuda feature is active)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn probe_cuda() -> Result<Vec<InfoEntry>, Box<dyn Error>> {
    use cudarc::driver::sys::{
        CUdevice_attribute, cuCtxCreate_v2, cuCtxDestroy_v2, cuDeviceGetAttribute,
        cuDeviceGetCount, cuDeviceGetName, cuInit, cuMemGetInfo_v2,
    };

    let mut entries = Vec::new();

    // cuInit is idempotent; ignore error if already initialised.
    unsafe { cuInit(0) };

    let mut count: i32 = 0;
    let rc = unsafe { cuDeviceGetCount(&mut count) };
    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS || count == 0 {
        return Ok(entries);
    }

    for dev in 0..count {
        let mut name_buf = [0u8; 256];
        unsafe { cuDeviceGetName(name_buf.as_mut_ptr() as *mut _, 256, dev) };
        let name = unsafe {
            std::ffi::CStr::from_ptr(name_buf.as_ptr() as *const _)
                .to_string_lossy()
                .into_owned()
        };

        let mut major = 0i32;
        let mut minor = 0i32;
        unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            );
            cuDeviceGetAttribute(
                &mut minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            );
        }

        let mut free: usize = 0;
        let mut total: usize = 0;
        // cuMemGetInfo_v2 requires an active context.
        let mut ctx = std::ptr::null_mut();
        unsafe { cuCtxCreate_v2(&mut ctx, 0, dev) };
        unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        unsafe { cuCtxDestroy_v2(ctx) };

        entries.push(InfoEntry {
            kind: "CUDA",
            name,
            mem_total_mib: Some((total / (1024 * 1024)) as u64),
            mem_free_mib: Some((free / (1024 * 1024)) as u64),
            extra: vec![("Compute", format!("sm_{major}{minor}"))],
        });
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------
// HIP/ROCm probing
// ---------------------------------------------------------------------------

#[cfg(feature = "hip")]
fn probe_hip() -> Result<Vec<InfoEntry>, Box<dyn Error>> {
    use hip_runtime_sys::{hipGetDeviceCount, hipGetDeviceProperties, hipMemGetInfo, hipError_t};

    let mut entries = Vec::new();
    let mut count: i32 = 0;
    let rc = unsafe { hipGetDeviceCount(&mut count) };
    if rc != hipError_t::hipSuccess || count == 0 {
        return Ok(entries);
    }

    for dev in 0..count {
        let mut prop: hip_runtime_sys::hipDeviceProp_t = unsafe { std::mem::zeroed() };
        unsafe { hipGetDeviceProperties(&mut prop, dev) };
        let name = unsafe {
            std::ffi::CStr::from_ptr(prop.name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        let mut free: usize = 0;
        let mut total: usize = 0;
        // hipMemGetInfo requires an active context; best-effort only.
        unsafe { hipMemGetInfo(&mut free, &mut total) };
        let arch = unsafe {
            std::ffi::CStr::from_ptr(prop.gcnArchName.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        entries.push(InfoEntry {
            kind: "HIP",
            name,
            mem_total_mib: Some((total / (1024 * 1024)) as u64),
            mem_free_mib: Some((free / (1024 * 1024)) as u64),
            extra: vec![("Arch", arch)],
        });
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------
// wgpu probing
// ---------------------------------------------------------------------------

#[cfg(feature = "wgpu")]
fn probe_wgpu() -> Vec<InfoEntry> {
    use wgpu::{Backends, Instance, InstanceDescriptor};

    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });
    instance
        .enumerate_adapters(Backends::all())
        .into_iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();
            // Skip software/CPU adapters (e.g. Lavapipe, SwiftShader).
            if info.device_type == wgpu::DeviceType::Cpu {
                return None;
            }
            let backend = format!("{:?}", info.backend);
            Some(InfoEntry {
                kind: "wgpu",
                name: info.name,
                mem_total_mib: None,
                mem_free_mib: None,
                extra: vec![("Backend", backend)],
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CPU probing
// ---------------------------------------------------------------------------

fn probe_cpu() -> InfoEntry {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let ram_mib = sys_total_ram_mib();

    InfoEntry {
        kind: "CPU",
        name: "host CPU".into(),
        mem_total_mib: ram_mib,
        mem_free_mib: None,
        extra: vec![("Cores", cores.to_string())],
    }
}

#[cfg(target_os = "linux")]
fn sys_total_ram_mib() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if line.starts_with("MemTotal:") {
            let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn sys_total_ram_mib() -> Option<u64> {
    // sysctl hw.memsize
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let bytes: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(bytes / (1024 * 1024))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn sys_total_ram_mib() -> Option<u64> { None }

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn print_table(entries: &[InfoEntry]) {
    println!("nabla hardware info");
    println!("{}", "─".repeat(50));
    for e in entries {
        let mem = match (e.mem_total_mib, e.mem_free_mib) {
            (Some(t), Some(f)) => format!("{t} MiB total / {f} MiB free"),
            (Some(t), None)    => format!("{t} MiB total"),
            _                  => "unknown".into(),
        };
        println!("Backend : {}", e.kind);
        println!("Device  : {}", e.name);
        println!("Memory  : {mem}");
        for (k, v) in &e.extra {
            println!("{k:<8}: {v}");
        }
        println!("{}", "─".repeat(50));
    }
}

fn print_json(entries: &[InfoEntry]) {
    print!("[");
    for (i, e) in entries.iter().enumerate() {
        if i > 0 { print!(","); }
        print!(
            "{{\"backend\":{:?},\"device\":{:?},\"mem_total_mib\":{:?},\"mem_free_mib\":{:?}",
            e.kind, e.name, e.mem_total_mib, e.mem_free_mib,
        );
        for (k, v) in &e.extra {
            print!(",{:?}:{:?}", k, v);
        }
        print!("}}");
    }
    println!("]");
}
