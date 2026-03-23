#![forbid(unsafe_code)]

use yscv_cli::config::{CliConfig, CliError, print_usage};
use yscv_cli::diagnostics::{print_camera_devices, run_camera_diagnostics};
use yscv_cli::error::AppError;
use yscv_cli::evaluation::{run_dataset_evaluation, run_diagnostics_report_validation};
use yscv_cli::pipeline::run_pipeline;

fn main() {
    if let Err(err) = run_app() {
        eprintln!("yscv-cli demo failed: {err}");
        std::process::exit(1);
    }
}

fn run_app() -> Result<(), AppError> {
    let cli = match CliConfig::from_env() {
        Ok(config) => config,
        Err(CliError::HelpRequested) => {
            print_usage();
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    if cli.list_cameras {
        print_camera_devices(cli.device_name_query.as_deref())?;
        return Ok(());
    }
    if cli.diagnose_camera {
        run_camera_diagnostics(&cli)?;
        return Ok(());
    }
    if cli.eval_detection_dataset_path.is_some()
        || cli.eval_detection_coco_gt_path.is_some()
        || cli.eval_detection_openimages_gt_path.is_some()
        || cli.eval_detection_yolo_manifest_path.is_some()
        || cli.eval_detection_voc_manifest_path.is_some()
        || cli.eval_detection_kitti_manifest_path.is_some()
        || cli.eval_detection_widerface_gt_path.is_some()
        || cli.eval_tracking_dataset_path.is_some()
        || cli.eval_tracking_mot_gt_path.is_some()
    {
        run_dataset_evaluation(&cli)?;
        return Ok(());
    }
    if cli.validate_diagnostics_report_path.is_some() {
        run_diagnostics_report_validation(&cli)?;
        return Ok(());
    }

    run_pipeline(&cli)
}
