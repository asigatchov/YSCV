//! Example: run a seq9 grayscale VballNetGrid ONNX model on video or frame folders.
//!
//! Python-compatible flags:
//!   cargo run --example vball_grid_infer -- \
//!     --model_path model.onnx \
//!     --video_path input.mp4 \
//!     --output_dir out \
//!     --confidence_threshold 0.5 \
//!     --track_length 8 \
//!     --only_csv \
//!     --verbose
//!
//! Legacy positional form is also accepted:
//!   cargo run --example vball_grid_infer -- <model.onnx> <input> [output.csv]

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::Instant;

use yscv_imgproc::{ImageU8, imread, resize_bilinear, rgb_to_grayscale};
use yscv_onnx::{load_onnx_model_from_file, run_onnx_model};
use yscv_tensor::Tensor;
use yscv_video::{RawVideoReader, Rgb8Frame};

const GRID_INPUT_WIDTH: usize = 768;
const GRID_INPUT_HEIGHT: usize = 432;
const GRID_COLS: usize = 48;
const GRID_ROWS: usize = 27;
const DEFAULT_SEQ_LEN: usize = 9;
const DEFAULT_THRESHOLD: f32 = 0.5;

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone)]
struct FrameInput {
    index: usize,
    width: usize,
    height: usize,
    rgb: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFamily {
    Grid,
    Heatmap,
}

#[derive(Debug, Clone)]
struct ModelParams {
    family: ModelFamily,
    seq: usize,
    input_seq: usize,
    input_width: usize,
    input_height: usize,
    grid_cols: Option<usize>,
    grid_rows: Option<usize>,
}

#[derive(Debug, Clone)]
struct Config {
    model_path: PathBuf,
    input_path: PathBuf,
    output_dir: Option<PathBuf>,
    explicit_csv_path: Option<PathBuf>,
    confidence_threshold: f32,
    track_length: usize,
    only_csv: bool,
    visualize: bool,
    verbose: bool,
}

enum InputSource {
    Directory { paths: Vec<PathBuf>, cursor: usize },
    Mp4 { reader: FfmpegRgb24Reader, index: usize },
    Raw(RawVideoReader),
}

struct FfmpegRgb24Reader {
    child: Child,
    stdout: ChildStdout,
    width: usize,
    height: usize,
}

impl FfmpegRgb24Reader {
    fn open(path: &Path) -> AppResult<Self> {
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
            ])
            .arg(path)
            .output()?;
        if !probe.status.success() {
            return Err(format!("ffprobe failed for {}", path.display()).into());
        }
        let dims = String::from_utf8(probe.stdout)?;
        let dims = dims.trim();
        let (width, height) = dims
            .split_once('x')
            .ok_or_else(|| format!("unexpected ffprobe output: {dims}"))?;
        let width = width.parse::<usize>()?;
        let height = height.parse::<usize>()?;

        let mut child = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdout = child.stdout.take().ok_or("failed to capture ffmpeg stdout")?;

        Ok(Self {
            child,
            stdout,
            width,
            height,
        })
    }

    fn next_frame(&mut self) -> AppResult<Option<(usize, usize, Vec<u8>)>> {
        let frame_len = self.width * self.height * 3;
        let mut data = vec![0u8; frame_len];
        match self.stdout.read_exact(&mut data) {
            Ok(()) => Ok(Some((self.width, self.height, data))),
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => {
                let _ = self.child.wait();
                Ok(None)
            }
            Err(err) => Err(err.into()),
        }
    }
}

impl InputSource {
    fn open(path: &Path) -> AppResult<Self> {
        if path.is_dir() {
            let mut paths: Vec<PathBuf> = fs::read_dir(path)?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|path| {
                    path.extension()
                        .and_then(OsStr::to_str)
                        .map(|ext| {
                            matches!(
                                ext.to_ascii_lowercase().as_str(),
                                "png" | "jpg" | "jpeg" | "bmp" | "webp"
                            )
                        })
                        .unwrap_or(false)
                })
                .collect();
            paths.sort();
            if paths.is_empty() {
                return Err(format!("no image frames found in {}", path.display()).into());
            }
            return Ok(Self::Directory { paths, cursor: 0 });
        }

        match path
            .extension()
            .and_then(OsStr::to_str)
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref()
        {
            Some("mp4") => Ok(Self::Mp4 {
                reader: FfmpegRgb24Reader::open(path)?,
                index: 0,
            }),
            Some("rcv") => Ok(Self::Raw(RawVideoReader::open(path)?)),
            _ => Err(format!(
                "unsupported input {}: use a frame directory, .mp4, or .rcv",
                path.display()
            )
            .into()),
        }
    }

    fn next_chunk(&mut self, chunk_size: usize) -> AppResult<Vec<FrameInput>> {
        let mut frames = Vec::with_capacity(chunk_size);
        match self {
            Self::Directory { paths, cursor } => {
                for _ in 0..chunk_size {
                    if *cursor >= paths.len() {
                        break;
                    }
                    let path = &paths[*cursor];
                    let rgb = imread(path)?;
                    let shape = rgb.shape();
                    frames.push(FrameInput {
                        index: *cursor,
                        width: shape[1],
                        height: shape[0],
                        rgb,
                    });
                    *cursor += 1;
                }
            }
            Self::Mp4 { reader, index } => {
                for _ in 0..chunk_size {
                    let Some((width, height, rgb8_data)) = reader.next_frame()? else {
                        break;
                    };
                    frames.push(decoded_frame_to_input(*index, width, height, rgb8_data)?);
                    *index += 1;
                }
            }
            Self::Raw(reader) => {
                for _ in 0..chunk_size {
                    let Some(frame) = reader.next_frame() else {
                        break;
                    };
                    frames.push(rgb8_frame_to_input(frame)?);
                }
            }
        }
        Ok(frames)
    }
}

fn rgb8_frame_to_input(frame: Rgb8Frame) -> AppResult<FrameInput> {
    let index = frame.index() as usize;
    let width = frame.width();
    let height = frame.height();
    let image = ImageU8::new(frame.into_data(), height, width, 3)
        .ok_or("invalid RGB8 frame buffer shape")?;
    Ok(FrameInput {
        index,
        width,
        height,
        rgb: image.to_tensor(),
    })
}

fn decoded_frame_to_input(
    index: usize,
    width: usize,
    height: usize,
    rgb8_data: Vec<u8>,
) -> AppResult<FrameInput> {
    let image =
        ImageU8::new(rgb8_data, height, width, 3).ok_or("invalid decoded frame buffer shape")?;
    Ok(FrameInput {
        index,
        width,
        height,
        rgb: image.to_tensor(),
    })
}

fn infer_model_params(model_path: &Path) -> ModelParams {
    let model_name = model_path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_ascii_lowercase();

    if model_name.contains("vballnetgrid") {
        return ModelParams {
            family: ModelFamily::Grid,
            seq: DEFAULT_SEQ_LEN,
            input_seq: DEFAULT_SEQ_LEN,
            input_width: GRID_INPUT_WIDTH,
            input_height: GRID_INPUT_HEIGHT,
            grid_cols: Some(GRID_COLS),
            grid_rows: Some(GRID_ROWS),
        };
    }

    let seq = if model_name.contains("seq15") {
        15
    } else if model_name.contains("seq9") {
        9
    } else {
        3
    };

    ModelParams {
        family: ModelFamily::Heatmap,
        seq,
        input_seq: if model_name.contains("seq15") { 15 } else { 9 },
        input_width: GRID_INPUT_WIDTH,
        input_height: GRID_INPUT_HEIGHT,
        grid_cols: None,
        grid_rows: None,
    }
}

fn preprocess_frame(frame: &Tensor, model_params: &ModelParams) -> AppResult<Tensor> {
    let gray = match frame.shape()[2] {
        1 => frame.clone(),
        3 => rgb_to_grayscale(frame)?,
        channels => {
            return Err(format!("unsupported frame channel count: {channels}").into());
        }
    };
    Ok(resize_bilinear(
        &gray,
        model_params.input_height,
        model_params.input_width,
    )?)
}

fn build_clip_input(frames: &[Tensor], model_params: &ModelParams) -> AppResult<Tensor> {
    if frames.len() != model_params.input_seq {
        return Err(
            format!("expected {} frames, got {}", model_params.input_seq, frames.len()).into(),
        );
    }

    let plane = model_params.input_height * model_params.input_width;
    let mut data = vec![0.0f32; model_params.input_seq * plane];
    for (channel, frame) in frames.iter().enumerate() {
        let expected = [model_params.input_height, model_params.input_width, 1];
        if frame.shape() != expected {
            return Err(format!("unexpected preprocessed frame shape: {:?}", frame.shape()).into());
        }
        let start = channel * plane;
        data[start..start + plane].copy_from_slice(frame.data());
    }

    Ok(Tensor::from_vec(
        vec![
            1,
            model_params.input_seq,
            model_params.input_height,
            model_params.input_width,
        ],
        data,
    )?)
}

fn decode_grid_output(
    output: &Tensor,
    model_params: &ModelParams,
    threshold: f32,
) -> AppResult<Vec<(u8, usize, usize)>> {
    let grid_rows = model_params.grid_rows.ok_or("grid_rows missing")?;
    let grid_cols = model_params.grid_cols.ok_or("grid_cols missing")?;
    let expected = [1, model_params.seq * 3, grid_rows, grid_cols];
    let shape = output.shape();
    if shape != expected {
        return Err(format!("unexpected model output shape {:?}", shape).into());
    }

    let data = output.data();
    let mut results = Vec::with_capacity(model_params.seq);
    let plane = grid_rows * grid_cols;

    for frame_idx in 0..model_params.seq {
        let conf_base = (frame_idx * 3) * plane;
        let x_base = conf_base + plane;
        let y_base = x_base + plane;

        let conf = &data[conf_base..conf_base + plane];
        let (best_idx, &score) = conf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .ok_or("empty confidence map")?;

        if score < threshold {
            results.push((0, 0, 0));
            continue;
        }

        let row = best_idx / grid_cols;
        let col = best_idx % grid_cols;
        let x = (col as f32 + data[x_base + best_idx])
            * (model_params.input_width as f32 / grid_cols as f32);
        let y = (row as f32 + data[y_base + best_idx])
            * (model_params.input_height as f32 / grid_rows as f32);
        let x = x.clamp(0.0, (model_params.input_width - 1) as f32) as usize;
        let y = y.clamp(0.0, (model_params.input_height - 1) as f32) as usize;
        results.push((1, x, y));
    }

    Ok(results)
}

fn decode_predictions(
    output: &Tensor,
    model_params: &ModelParams,
    threshold: f32,
) -> AppResult<Vec<(u8, usize, usize)>> {
    match model_params.family {
        ModelFamily::Grid => decode_grid_output(output, model_params, threshold),
        ModelFamily::Heatmap => Err("heatmap models are not implemented in this Rust example".into()),
    }
}

fn project_prediction_to_frame(
    prediction: (u8, usize, usize),
    frame_width: usize,
    frame_height: usize,
    model_params: &ModelParams,
) -> (u8, isize, isize) {
    let (visible, x, y) = prediction;
    if visible == 0 {
        return (0, -1, -1);
    }
    (
        visible,
        ((x as f32) * frame_width as f32 / model_params.input_width as f32).round() as isize,
        ((y as f32) * frame_height as f32 / model_params.input_height as f32).round() as isize,
    )
}

fn write_csv_row(
    writer: &mut fs::File,
    frame_index: usize,
    visible: u8,
    x: isize,
    y: isize,
) -> AppResult<()> {
    writeln!(writer, "{frame_index},{visible},{x},{y}")?;
    Ok(())
}

fn print_usage_and_exit() -> ! {
    eprintln!("Usage:");
    eprintln!("  vball_grid_infer --model_path <model.onnx> --video_path <input.mp4|input.rcv|frames_dir> [options]");
    eprintln!("  vball_grid_infer <model.onnx> <input> [output.csv]");
    eprintln!("Options:");
    eprintln!("  --output_dir <dir>");
    eprintln!("  --confidence_threshold <float>");
    eprintln!("  --track_length <int>");
    eprintln!("  --only_csv");
    eprintln!("  --visualize");
    eprintln!("  --verbose");
    std::process::exit(1);
}

fn parse_args() -> AppResult<Config> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage_and_exit();
    }

    if !args[0].starts_with('-') {
        if args.len() < 2 || args.len() > 3 {
            print_usage_and_exit();
        }
        return Ok(Config {
            model_path: PathBuf::from(&args[0]),
            input_path: PathBuf::from(&args[1]),
            output_dir: None,
            explicit_csv_path: args.get(2).map(PathBuf::from),
            confidence_threshold: DEFAULT_THRESHOLD,
            track_length: 8,
            only_csv: true,
            visualize: false,
            verbose: false,
        });
    }

    let mut model_path = None;
    let mut input_path = None;
    let mut output_dir = None;
    let mut explicit_csv_path = None;
    let mut confidence_threshold = DEFAULT_THRESHOLD;
    let mut track_length = 8usize;
    let mut only_csv = false;
    let mut visualize = false;
    let mut verbose = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model_path" => {
                i += 1;
                model_path = args.get(i).map(PathBuf::from);
            }
            "--video_path" => {
                i += 1;
                input_path = args.get(i).map(PathBuf::from);
            }
            "--output_dir" => {
                i += 1;
                output_dir = args.get(i).map(PathBuf::from);
            }
            "--output_csv" => {
                i += 1;
                explicit_csv_path = args.get(i).map(PathBuf::from);
            }
            "--confidence_threshold" => {
                i += 1;
                let value = args.get(i).ok_or("missing value for --confidence_threshold")?;
                confidence_threshold = value.parse::<f32>()?;
            }
            "--track_length" => {
                i += 1;
                let value = args.get(i).ok_or("missing value for --track_length")?;
                track_length = value.parse::<usize>()?;
            }
            "--only_csv" => {
                only_csv = true;
            }
            "--visualize" => {
                visualize = true;
            }
            "--verbose" => {
                verbose = true;
            }
            "--help" | "-h" => {
                print_usage_and_exit();
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
        i += 1;
    }

    let model_path = model_path.ok_or("missing --model_path")?;
    let input_path = input_path.ok_or("missing --video_path")?;
    Ok(Config {
        model_path,
        input_path,
        output_dir,
        explicit_csv_path,
        confidence_threshold,
        track_length,
        only_csv,
        visualize,
        verbose,
    })
}

fn csv_output_path(config: &Config) -> Option<PathBuf> {
    if let Some(path) = &config.explicit_csv_path {
        return Some(path.clone());
    }

    let output_dir = config.output_dir.as_ref()?;
    let video_basename = config
        .input_path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("output");
    Some(output_dir.join(video_basename).join("ball.csv"))
}

fn main() -> AppResult<()> {
    let config = parse_args()?;
    let model_params = infer_model_params(&config.model_path);
    if model_params.family != ModelFamily::Grid {
        return Err("this Rust example currently supports only VballNetGrid models".into());
    }

    let model = load_onnx_model_from_file(&config.model_path)?;
    let input_name = model
        .inputs
        .first()
        .cloned()
        .ok_or("model has no declared inputs")?;
    let output_name = model
        .outputs
        .first()
        .cloned()
        .ok_or("model has no declared outputs")?;

    if config.visualize {
        eprintln!("warning: --visualize is accepted for CLI compatibility but is not implemented in Rust");
    }
    if config.track_length != 8 && config.verbose {
        eprintln!(
            "info: --track_length={} is accepted for CLI compatibility but is unused by this detector",
            config.track_length
        );
    }

    let csv_path = csv_output_path(&config);
    let mut writer = if let Some(path) = &csv_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(path)?;
        writeln!(file, "Frame,Visibility,X,Y")?;
        Some(file)
    } else {
        None
    };

    let mut source = InputSource::open(&config.input_path)?;

    println!("Loading model: {}", config.model_path.display());
    println!("Input tensor name: {input_name}");
    println!("Output tensor name: {output_name}");
    println!(
        "Mode: family=grid input_seq={} output_seq={} input={}x{} threshold={:.3}",
        model_params.input_seq,
        model_params.seq,
        model_params.input_width,
        model_params.input_height,
        config.confidence_threshold
    );

    let start = Instant::now();
    let mut processed_frames = 0usize;
    let mut inference_calls = 0usize;

    loop {
        let chunk = source.next_chunk(model_params.input_seq)?;
        if chunk.is_empty() {
            break;
        }

        let processed: Vec<Tensor> = chunk
            .iter()
            .map(|frame| preprocess_frame(&frame.rgb, &model_params))
            .collect::<AppResult<_>>()?;

        let clip = if processed.len() == model_params.input_seq {
            build_clip_input(&processed, &model_params)?
        } else {
            let mut padded = processed.clone();
            let pad_value = if let Some(frame) = processed.last() {
                frame.clone()
            } else {
                Tensor::zeros(vec![model_params.input_height, model_params.input_width, 1])?
            };
            while padded.len() < model_params.input_seq {
                padded.push(pad_value.clone());
            }
            build_clip_input(&padded, &model_params)?
        };

        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), clip);
        let outputs = run_onnx_model(&model, inputs)?;
        let output = outputs
            .get(&output_name)
            .or_else(|| outputs.values().next())
            .ok_or("model produced no outputs")?;
        let predictions =
            decode_predictions(output, &model_params, config.confidence_threshold)?;

        for (frame, prediction) in chunk.iter().zip(predictions.iter()) {
            let (visible, x_out, y_out) =
                project_prediction_to_frame(*prediction, frame.width, frame.height, &model_params);
            if let Some(file) = writer.as_mut() {
                write_csv_row(file, frame.index, visible, x_out, y_out)?;
            }
            if config.verbose {
                println!(
                    "frame {:>6}: visible={} x={} y={}",
                    frame.index, visible, x_out, y_out
                );
            }
            processed_frames += 1;
        }

        inference_calls += 1;
    }

    let elapsed = start.elapsed();
    let fps = if elapsed.as_secs_f64() > 0.0 {
        processed_frames as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "Processed {processed_frames} frames in {:.3}s across {inference_calls} inference calls ({fps:.2} FPS)",
        elapsed.as_secs_f64()
    );
    if let Some(path) = csv_path {
        println!("CSV saved to {}", path.display());
    } else {
        println!("CSV output skipped (no --output_dir / --output_csv provided)");
    }
    if !config.only_csv && config.output_dir.is_some() {
        println!("Video output is not implemented in Rust; only CSV was written");
    }

    Ok(())
}
