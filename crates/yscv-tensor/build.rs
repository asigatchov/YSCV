fn main() {
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

    // macOS: link Accelerate.framework for vDSP vector operations.
    // Use CARGO_CFG_TARGET_OS (not cfg!) to check the TARGET, not the HOST.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
