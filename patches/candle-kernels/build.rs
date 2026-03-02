use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let ptx_path = out_dir.join("ptx.rs");

    // Detect CUDA toolchain by probing nvcc — avoid calling nvidia-smi which panics on macOS
    let cuda_available = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    // Always emit stub PTX constants.  On systems that have CUDA the full
    // upstream crate should be used; here we just need the workspace to
    // compile and link so that CPU-only tests can run.
    let stub = "\
pub const AFFINE: &str = \"\";\n\
pub const BINARY: &str = \"\";\n\
pub const CAST: &str = \"\";\n\
pub const CONV: &str = \"\";\n\
pub const FILL: &str = \"\";\n\
pub const INDEXING: &str = \"\";\n\
pub const QUANTIZED: &str = \"\";\n\
pub const REDUCE: &str = \"\";\n\
pub const SORT: &str = \"\";\n\
pub const TERNARY: &str = \"\";\n\
pub const UNARY: &str = \"\";\n\
";
    std::fs::write(&ptx_path, stub).expect("Failed to write stub ptx.rs");

    if cuda_available {
        println!(
            "cargo:warning=candle-kernels stub: CUDA (nvcc) detected but this is a no-op stub. \
             Real CUDA kernels are NOT compiled. GPU inference will panic at runtime. \
             Use the upstream candle-kernels for GPU inference."
        );
    } else {
        println!(
            "cargo:warning=candle-kernels stub: CUDA (nvcc) not found — using empty PTX stubs. \
             GPU ops will panic at runtime with a clear message."
        );
    }
}
