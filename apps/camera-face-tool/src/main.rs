#![forbid(unsafe_code)]

mod config;
mod diagnostics;
mod error;
mod event_log;
mod pipeline;
mod util;

use config::{FaceAppConfig, FaceAppError, print_usage};
use diagnostics::{print_camera_devices, run_camera_diagnostics};
use error::AppError;
use pipeline::run_pipeline;

fn main() {
    if let Err(err) = run() {
        eprintln!("camera-face-app failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), AppError> {
    let config = match FaceAppConfig::from_env() {
        Ok(config) => config,
        Err(FaceAppError::HelpRequested) => {
            print_usage();
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    if config.list_cameras {
        print_camera_devices(config.device_name_query.as_deref())?;
        return Ok(());
    }
    if config.diagnose_camera {
        run_camera_diagnostics(&config)?;
        return Ok(());
    }

    run_pipeline(&config)
}
