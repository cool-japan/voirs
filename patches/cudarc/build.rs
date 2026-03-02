//! Patched build script for cudarc on macOS / platforms without CUDA.
//!
//! On systems where `nvcc` is absent this build script falls back to a fake
//! CUDA version (12.0) and skips all link-library directives so that the
//! workspace compiles for CPU-only use.  Any code path that actually tries to
//! use CUDA at runtime will panic with a clear message because the real shared
//! libraries will not be present.
//!
//! On systems where CUDA is available the original logic is preserved.

use std::path::PathBuf;

const TYPICAL_CUDA_PATH_ENV_VARS: [&str; 5] = [
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
    "CUDNN_LIB",
];

const SUPPORTED_CUDA_VERSIONS: &[((usize, usize), bool)] = &[
    ((13, 1), cfg!(feature = "cuda-13010")),
    ((13, 0), cfg!(feature = "cuda-13000")),
    ((12, 9), cfg!(feature = "cuda-12090")),
    ((12, 8), cfg!(feature = "cuda-12080")),
    ((12, 6), cfg!(feature = "cuda-12060")),
    ((12, 5), cfg!(feature = "cuda-12050")),
    ((12, 4), cfg!(feature = "cuda-12040")),
    ((12, 3), cfg!(feature = "cuda-12030")),
    ((12, 2), cfg!(feature = "cuda-12020")),
    ((12, 1), cfg!(feature = "cuda-12010")),
    ((12, 0), cfg!(feature = "cuda-12000")),
    ((11, 8), cfg!(feature = "cuda-11080")),
    ((11, 7), cfg!(feature = "cuda-11070")),
    ((11, 6), cfg!(feature = "cuda-11060")),
    ((11, 5), cfg!(feature = "cuda-11050")),
    ((11, 4), cfg!(feature = "cuda-11040")),
];

fn detect_version_from_env() -> Option<(usize, usize)> {
    match std::env::var("CUDARC_CUDA_VERSION") {
        Ok(version) => {
            let version = version.as_str();
            for &((major, minor), _) in SUPPORTED_CUDA_VERSIONS.iter() {
                if version == format!("{major}0{minor}0") {
                    return Some((major, minor));
                }
            }
            panic!("Unsupported cuda toolkit version: `$CUDARC_CUDA_VERSION={version}`. Please raise a github issue.")
        }
        _ => None,
    }
}

fn detect_version_from_feature() -> Option<(usize, usize)> {
    for &((major, minor), is_feature_set) in SUPPORTED_CUDA_VERSIONS.iter() {
        if is_feature_set {
            return Some((major, minor));
        }
    }
    None
}

/// Try to determine the CUDA version from `nvcc --version`.
/// Returns `None` when `nvcc` is not found (macOS without CUDA, etc.).
fn try_cuda_version_from_build_system() -> Option<(usize, usize)> {
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().nth(3)?;
    let release_section = version_line.split(", ").nth(1)?;
    let version_number = release_section.split(' ').nth(1)?;

    for &((major, minor), _) in SUPPORTED_CUDA_VERSIONS.iter() {
        if version_number == format!("{major}.{minor}") {
            return Some((major, minor));
        }
    }
    None
}

fn main() {
    // Feature-conflict guards (patched version).
    // Note: In this patched crate, `dynamic-linking` is remapped to
    // `dynamic-loading` so that CUDA symbols are loaded at runtime rather
    // than at link time.  Both features being active simultaneously is
    // therefore intentional and expected — we skip the original conflict panic.
    #[cfg(all(
        not(feature = "dynamic-linking"),
        not(feature = "static-linking"),
        not(feature = "dynamic-loading"),
        not(feature = "fallback-dynamic-loading")
    ))]
    panic!("None between `dynamic-loading`, `fallback-dynamic-loading`, `dynamic-linking` and `static-linking` features are active, this is a bug");
    #[cfg(all(feature = "dynamic-linking", feature = "static-linking"))]
    panic!("Both `dynamic-linking` and `static-linking` features are active, this is a bug");
    #[cfg(all(feature = "dynamic-loading", feature = "static-linking"))]
    panic!("Both `dynamic-loading` and `static-linking` features are active, this is a bug");
    // NOTE: `dynamic-loading` + `dynamic-linking` conflict is intentionally
    // suppressed in this patched crate — `dynamic-linking` activates
    // `dynamic-loading` to avoid link-time CUDA symbol requirements.

    #[cfg(all(
        feature = "fallback-dynamic-loading",
        not(any(
            feature = "dynamic-loading",
            feature = "dynamic-linking",
            feature = "static-linking"
        ))
    ))]
    println!("cargo:rustc-cfg=feature=\"dynamic-loading\"");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");
    TYPICAL_CUDA_PATH_ENV_VARS
        .iter()
        .for_each(|var| println!("cargo:rerun-if-env-changed={var}"));

    // Determine CUDA version, falling back gracefully when CUDA is absent.
    let cuda_version: Option<(usize, usize)> =
        if let Some(v) = detect_version_from_env() {
            println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"", major = v.0, minor = v.1);
            Some(v)
        } else if let Some(v) = detect_version_from_feature() {
            Some(v)
        } else {
            // `cuda-version-from-build-system` or no explicit version — probe nvcc.
            // If nvcc is absent (macOS without CUDA) we fall back to version 12.0
            // so that the `env!("CUDA_MAJOR_VERSION")` / `env!("CUDA_MINOR_VERSION")`
            // invocations inside lib.rs compile successfully.
            let probed = try_cuda_version_from_build_system();
            if let Some(v) = probed {
                println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"", major = v.0, minor = v.1);
                Some(v)
            } else {
                println!(
                    "cargo:warning=cudarc stub: CUDA (nvcc) not found — using placeholder \
                     version 12.0. All GPU operations will fail at runtime with a clear \
                     message. This is expected on macOS without a CUDA toolkit."
                );
                // Emit the cuda-12000 cfg so the conditional compilation in
                // cudarc's own source doesn't hit unexpected paths.
                println!("cargo:rustc-cfg=feature=\"cuda-12000\"");
                Some((12, 0))
            }
        };

    let (major, minor) = cuda_version.unwrap_or((12, 0));
    println!("cargo:rustc-env=CUDA_MAJOR_VERSION={major}");
    println!("cargo:rustc-env=CUDA_MINOR_VERSION={minor}");

    // Only emit link directives when CUDA libraries are actually present on the
    // system.  On macOS without CUDA we skip them entirely — the workspace will
    // still compile; any runtime invocation of a GPU operation will simply fail
    // to load the shared library through the dynamic-loading machinery.
    // Check for a non-empty CUDA installation indicator.
    // Note: `std::env::var().is_ok()` returns true even for empty strings.
    // We must check that the value is non-empty AND points to an existing path.
    let env_var_has_cuda = |var: &str| -> bool {
        std::env::var(var)
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| std::path::Path::new(&v).exists())
            .unwrap_or(false)
    };
    let nvcc_found = try_cuda_version_from_build_system().is_some();
    let cuda_present = nvcc_found
        || env_var_has_cuda("CUDA_HOME")
        || env_var_has_cuda("CUDA_PATH")
        || env_var_has_cuda("CUDA_ROOT")
        || env_var_has_cuda("CUDA_TOOLKIT_ROOT_DIR");

    if !cuda_present {
        // No CUDA toolkit — skip all link-lib directives.
        return;
    }

    #[cfg(feature = "dynamic-linking")]
    dynamic_linking(major, minor);

    #[cfg(feature = "static-linking")]
    static_linking(major, minor);
}

#[allow(unused)]
fn dynamic_linking(major: usize, minor: usize) {
    for path in link_searches(major, minor) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    #[cfg(feature = "driver")]
    println!("cargo:rustc-link-lib=dylib=cuda");
    #[cfg(feature = "nccl")]
    println!("cargo:rustc-link-lib=dylib=nccl");
    #[cfg(feature = "nvrtc")]
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    #[cfg(feature = "curand")]
    println!("cargo:rustc-link-lib=dylib=curand");
    #[cfg(feature = "cublas")]
    println!("cargo:rustc-link-lib=dylib=cublas");
    #[cfg(any(feature = "cublas", feature = "cublaslt"))]
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    #[cfg(feature = "cupti")]
    println!("cargo:rustc-link-lib=dylib=cupti");
    #[cfg(feature = "cusparse")]
    println!("cargo:rustc-link-lib=dylib=cusparse");
    #[cfg(feature = "cusolver")]
    println!("cargo:rustc-link-lib=dylib=cusolver");
    #[cfg(feature = "cusolvermg")]
    println!("cargo:rustc-link-lib=dylib=cusolverMg");
    #[cfg(feature = "cudnn")]
    println!("cargo:rustc-link-lib=dylib=cudnn");
    #[cfg(feature = "runtime")]
    println!("cargo:rustc-link-lib=dylib=cudart");
    #[cfg(feature = "cufile")]
    {
        println!("cargo:rustc-link-lib=dylib=cufile");
        println!("cargo:rustc-link-lib=dylib=cufile_rdma");
    }
    #[cfg(feature = "nvtx")]
    println!("cargo:rustc-link-lib=dylib=nvToolsExt");
    #[cfg(feature = "cutensor")]
    println!("cargo:rustc-link-lib=dylib=cutensor");
}

#[allow(unused)]
fn static_linking(major: usize, minor: usize) {
    for path in link_searches(major, minor) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=static:+whole-archive=stdc++");
    #[cfg(any(feature = "driver", feature = "runtime"))]
    {
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=static:+whole-archive=cudart_static");
    }
    #[cfg(feature = "nccl")]
    println!("cargo:rustc-link-lib=static:+whole-archive=nccl_static");
    #[cfg(feature = "nvrtc")]
    {
        println!("cargo:rustc-link-lib=static:+whole-archive=nvrtc_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=nvptxcompiler_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=nvrtc-builtins_static");
    }
    #[cfg(any(
        feature = "curand",
        feature = "cublas",
        feature = "cublaslt",
        feature = "cusparse",
        feature = "cusolver"
    ))]
    println!("cargo:rustc-link-lib=static:+whole-archive=culibos");
    #[cfg(feature = "curand")]
    println!("cargo:rustc-link-lib=static:+whole-archive=curand_static");
    #[cfg(feature = "cublas")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cublas_static");
    #[cfg(any(feature = "cublas", feature = "cublaslt"))]
    println!("cargo:rustc-link-lib=static:+whole-archive=cublasLt_static");
    #[cfg(feature = "cupti")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cupti_static");
    #[cfg(feature = "cusparse")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cusparse_static");
    #[cfg(feature = "cusolver")]
    {
        println!("cargo:rustc-link-lib=static:+whole-archive=cusolver_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=cusolver_lapack_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=cusolver_metis_static");
    }
    #[cfg(feature = "cusolvermg")]
    println!("cargo:rustc-link-lib=dylib=cusolverMg");
    #[cfg(feature = "cudnn")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cudnn");
    #[cfg(feature = "cufile")]
    {
        println!("cargo:rustc-link-lib=static:+whole-archive=cufile_static");
        println!("cargo:rustc-link-lib=static:+whole-archive=cufile_rdma_static");
    }
    #[cfg(feature = "nvtx")]
    println!("cargo:rustc-link-lib=dylib=nvToolsExt");
    #[cfg(feature = "cutensor")]
    println!("cargo:rustc-link-lib=static:+whole-archive=cutensor_static");
}

#[allow(unused)]
fn link_searches(major: usize, minor: usize) -> Vec<PathBuf> {
    let env_vars = TYPICAL_CUDA_PATH_ENV_VARS
        .iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .collect::<Vec<_>>();

    // When building in a Conda-like environment with dynamic linking, if no
    // CUDA path is supplied, then it is highly likely that, by defaulting our
    // linker search paths to the typical locations below, linker errors will
    // occur. Print a warning with some guidance.
    #[cfg(feature = "dynamic-linking")]
    if env_vars.is_empty() && std::env::var("CONDA_PREFIX").is_ok() {
        println!("cargo::warning=Detected $CONDA_PREFIX, but no CUDA path was set through one of: {TYPICAL_CUDA_PATH_ENV_VARS:?}. Linking to system CUDA libraries; linker errors may occur. To use CUDA installed via conda please ensure the environment contains all required dependencies (e.g. the \"cuda-driver-dev\") and retry building with CUDA_HOME=$CONDA_PREFIX.")
    }

    let typical_locations = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/Program Files/NVIDIA",
        "C:/CUDA",
        "C:/Program Files/NVIDIA/CUDNN/v9.10",
        "C:/Program Files/NVIDIA/CUDNN/v9.9",
        "C:/Program Files/NVIDIA/CUDNN/v9.8",
        "C:/Program Files/NVIDIA/CUDNN/v9.7",
        "C:/Program Files/NVIDIA/CUDNN/v9.6",
        "C:/Program Files/NVIDIA/CUDNN/v9.5",
        "C:/Program Files/NVIDIA/CUDNN/v9.4",
        "C:/Program Files/NVIDIA/CUDNN/v9.3",
        "C:/Program Files/NVIDIA/CUDNN/v9.2",
        "C:/Program Files/NVIDIA/CUDNN/v9.1",
        "C:/Program Files/NVIDIA/CUDNN/v9.0",
    ];

    let possible_locations = if env_vars.is_empty() {
        typical_locations
            .into_iter()
            .map(Into::<String>::into)
            .collect()
    } else {
        env_vars
    };

    let mut candidates = Vec::new();
    for root in possible_locations.into_iter().map(Into::<PathBuf>::into) {
        candidates.extend(
            [
                "lib".into(),
                "lib/stubs".into(),
                "lib/x64".into(),
                "lib/Win32".into(),
                "lib/x86_64".into(),
                "lib/x86_64-linux-gnu".into(),
                "lib64".into(),
                "lib64/stubs".into(),
                "targets/x86_64-linux".into(),
                "targets/x86_64-linux/lib".into(),
                "targets/x86_64-linux/lib/stubs".into(),
                std::format!("lib/{major}.{minor}/x64"),
                std::format!("lib/{major}.{minor}/x86_64"),
            ]
            .iter()
            .map(|p| root.join(p))
            .filter(|p| p.is_dir()),
        )
    }

    candidates
}
