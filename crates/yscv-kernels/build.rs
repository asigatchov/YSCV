fn main() {
    // Link MetalPerformanceShaders.framework when metal-backend is enabled (macOS).
    #[cfg(feature = "metal-backend")]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        }
    }

    // Link Intel MKL VML when the "mkl" feature is enabled (x86/x86_64 only).
    #[cfg(feature = "mkl")]
    {
        // Try MKLROOT environment variable for custom install paths.
        if let Ok(mkl_root) = std::env::var("MKLROOT") {
            println!("cargo:rustc-link-search=native={mkl_root}/lib/intel64");
        }
        // Link MKL libraries (sequential, single-threaded for determinism).
        println!("cargo:rustc-link-lib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=mkl_sequential");
        println!("cargo:rustc-link-lib=mkl_core");
    }

    // Link ARM Performance Libraries when the "armpl" feature is enabled (aarch64 Linux only).
    #[cfg(feature = "armpl")]
    {
        if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
            && std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        {
            if let Ok(armpl_dir) = std::env::var("ARMPL_DIR") {
                println!("cargo:rustc-link-search=native={armpl_dir}/lib");
            }
            println!("cargo:rustc-link-lib=armpl_lp64");
        }
    }

    // Link BLAS when the "blas" feature is enabled.
    #[cfg(feature = "blas")]
    {
        // macOS: Accelerate.framework ships with the OS — zero setup.
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        // Linux: link OpenBLAS (apt install libopenblas-dev / yum install openblas-devel).
        // Also try to find it via pkg-config for non-standard install paths.
        else if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
            // Try pkg-config first for proper link flags
            if let Ok(out) = std::process::Command::new("pkg-config")
                .args(["--libs", "openblas"])
                .output()
            {
                if out.status.success() {
                    let flags = String::from_utf8_lossy(&out.stdout);
                    for flag in flags.split_whitespace() {
                        if let Some(lib) = flag.strip_prefix("-l") {
                            println!("cargo:rustc-link-lib={lib}");
                        } else if let Some(path) = flag.strip_prefix("-L") {
                            println!("cargo:rustc-link-search=native={path}");
                        }
                    }
                } else {
                    // Fallback: assume system-installed openblas
                    println!("cargo:rustc-link-lib=openblas");
                }
            } else {
                println!("cargo:rustc-link-lib=openblas");
            }
        }
        // Windows: link OpenBLAS (install via vcpkg / conda / manual).
        // vcpkg: vcpkg install openblas:x64-windows
        // conda: conda install -c conda-forge openblas
        else if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
            // Try VCPKG_ROOT first
            if let Ok(vcpkg_root) = std::env::var("VCPKG_ROOT") {
                let triplet = if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") {
                    "x64-windows"
                } else {
                    "x86-windows"
                };
                println!(
                    "cargo:rustc-link-search=native={}/installed/{}/lib",
                    vcpkg_root, triplet
                );
            }
            // Also check OPENBLAS_PATH environment variable
            if let Ok(blas_path) = std::env::var("OPENBLAS_PATH") {
                println!("cargo:rustc-link-search=native={blas_path}");
            }
            // Also check conda env
            if let Ok(conda_prefix) = std::env::var("CONDA_PREFIX") {
                println!("cargo:rustc-link-search=native={conda_prefix}/Library/lib");
            }
            println!("cargo:rustc-link-lib=openblas");
        }
    }
}
