# yscv Dataset Adapters

This page defines currently supported training/evaluation dataset formats and their runtime mapping contracts.

## Supported Formats
- Supervised training JSONL (model pipeline):
  - Crate: `yscv-model`
  - API: `SupervisedJsonlConfig`, `parse_supervised_dataset_jsonl`, `load_supervised_dataset_jsonl_file`
  - CLI: not wired yet (library-level training ingestion surface)
- Supervised training CSV (model pipeline):
  - Crate: `yscv-model`
  - API: `SupervisedCsvConfig`, `parse_supervised_dataset_csv`, `load_supervised_dataset_csv_file`
  - CLI: not wired yet (library-level training ingestion surface)
- Supervised training image-manifest CSV (model pipeline):
  - Crate: `yscv-model`
  - API: `SupervisedImageManifestConfig`, `parse_supervised_image_manifest_csv`, `load_supervised_image_manifest_csv_file`
  - CLI: not wired yet (library-level training ingestion surface)
- Supervised training image-folder classification tree (model pipeline):
  - Crate: `yscv-model`
  - API: `SupervisedImageFolderConfig`, `ImageFolderTargetMode`, `SupervisedImageFolderLoadResult`, `load_supervised_image_folder_dataset`, `load_supervised_image_folder_dataset_with_classes`
  - CLI: not wired yet (library-level training ingestion surface)
- Detection JSONL:
  - API: `parse_detection_dataset_jsonl`, `load_detection_dataset_jsonl_file`
  - CLI: `--eval-detection-jsonl <path>`
- Tracking JSONL:
  - API: `parse_tracking_dataset_jsonl`, `load_tracking_dataset_jsonl_file`
  - CLI: `--eval-tracking-jsonl <path>`
- Tracking MOTChallenge TXT pair:
  - API: `parse_tracking_dataset_mot`, `load_tracking_dataset_mot_txt_files`
  - CLI: `--eval-tracking-mot-gt <path>` + `--eval-tracking-mot-pred <path>`
- Detection COCO JSON pair:
  - API: `parse_detection_dataset_coco`, `load_detection_dataset_coco_files`
  - CLI: `--eval-detection-coco-gt <path>` + `--eval-detection-coco-pred <path>`
- Detection OpenImages CSV pair:
  - API: `parse_detection_dataset_openimages_csv`, `load_detection_dataset_openimages_csv_files`
  - CLI: `--eval-detection-openimages-gt <path>` + `--eval-detection-openimages-pred <path>`
- Detection YOLO label directories:
  - API: `load_detection_dataset_yolo_label_dirs`
  - CLI:
    - `--eval-detection-yolo-manifest <path>`
    - `--eval-detection-yolo-gt-dir <path>`
    - `--eval-detection-yolo-pred-dir <path>`
- Detection VOC XML directories:
  - API: `load_detection_dataset_voc_xml_dirs`
  - CLI:
    - `--eval-detection-voc-manifest <path>`
    - `--eval-detection-voc-gt-dir <path>`
    - `--eval-detection-voc-pred-dir <path>`
- Detection KITTI label directories:
  - API: `load_detection_dataset_kitti_label_dirs`
  - CLI:
    - `--eval-detection-kitti-manifest <path>`
    - `--eval-detection-kitti-gt-dir <path>`
    - `--eval-detection-kitti-pred-dir <path>`
- Detection WIDER FACE TXT pair:
  - API: `parse_detection_dataset_widerface`, `load_detection_dataset_widerface_files`
  - CLI: `--eval-detection-widerface-gt <path>` + `--eval-detection-widerface-pred <path>`

## COCO Detection Mapping
- Ground truth input: standard COCO object with `images[]` and `annotations[]`.
- Prediction input: COCO detection results list (`[]`) with `image_id`, `category_id`, `bbox`, `score`.
- `bbox` conversion: COCO `xywh` -> runtime `(x1, y1, x2, y2)`.
- Class mapping: `category_id` -> `class_id` (`usize` conversion with overflow validation).

## COCO Validation Rules
- `images[].id` must be unique.
- Every `annotation.image_id` and prediction `image_id` must exist in `images`.
- All bbox values and scores must be finite.
- Bbox `width` and `height` must be non-negative.

## OpenImages Detection Mapping
- Input pair:
  - ground truth CSV (`--eval-detection-openimages-gt`)
  - prediction CSV (`--eval-detection-openimages-pred`)
- Required columns:
  - `ImageID`, `LabelName`, `XMin`, `XMax`, `YMin`, `YMax`
  - prediction also requires score column: `Score` or `Confidence`.
- Coordinates are normalized `[0, 1]` bbox corners and are used directly in runtime boxes.
- `LabelName` is mapped to runtime `class_id` dynamically and consistently across GT/pred in one load pass.
- Validation enforces non-empty labels/image IDs, finite normalized values in `[0, 1]`, and strict bbox ordering (`XMax > XMin`, `YMax > YMin`).

## Sample Files
- Ground truth: `benchmarks/eval-detection-coco-gt-sample.json`
- Predictions: `benchmarks/eval-detection-coco-pred-sample.json`
- OpenImages GT: `benchmarks/eval-detection-openimages-gt-sample.csv`
- OpenImages predictions: `benchmarks/eval-detection-openimages-pred-sample.csv`
- MOT ground truth: `benchmarks/eval-tracking-mot-gt-sample.txt`
- MOT predictions: `benchmarks/eval-tracking-mot-pred-sample.txt`
- YOLO manifest: `benchmarks/eval-detection-yolo-manifest-sample.txt`
- YOLO GT labels: `benchmarks/eval-detection-yolo-gt/*.txt`
- YOLO prediction labels: `benchmarks/eval-detection-yolo-pred/*.txt`
- VOC manifest: `benchmarks/eval-detection-voc-manifest-sample.txt`
- VOC GT annotations: `benchmarks/eval-detection-voc-gt/*.xml`
- VOC prediction annotations: `benchmarks/eval-detection-voc-pred/*.xml`
- KITTI manifest: `benchmarks/eval-detection-kitti-manifest-sample.txt`
- KITTI GT labels: `benchmarks/eval-detection-kitti-gt/*.txt`
- KITTI prediction labels: `benchmarks/eval-detection-kitti-pred/*.txt`
- WIDER FACE ground truth: `benchmarks/eval-detection-widerface-gt-sample.txt`
- WIDER FACE predictions: `benchmarks/eval-detection-widerface-pred-sample.txt`

## Supervised Training JSONL Mapping (`yscv-model`)
- Record schema (one JSON object per line):
  - `input` (or alias `inputs`/`features`): flat `f32` sample vector
  - `target` (or alias `targets`/`label`/`labels`): flat `f32` target vector
    - scalar shorthand is also accepted when `target_shape` product is `1`
- Parsing contract:
  - `SupervisedJsonlConfig::new(input_shape, target_shape)` defines expected per-sample shapes,
  - each record length must exactly match shape product,
  - values must be finite (`NaN`/`inf` rejected),
  - empty/comment lines (`# ...`) are skipped,
  - at least one valid sample is required.
- Runtime mapping:
  - output `SupervisedDataset.inputs()` shape is `[N, input_shape...]`,
  - output `SupervisedDataset.targets()` shape is `[N, target_shape...]`.

## Supervised Training CSV Mapping (`yscv-model`)
- Row schema (one sample per data line):
  - `[input_flat..., target_flat...]`
  - total column count must equal `prod(input_shape) + prod(target_shape)`.
- Parsing contract:
  - `SupervisedCsvConfig::new(input_shape, target_shape)` defines expected shapes and default parsing options,
  - default delimiter is `,`; custom delimiter is supported via `with_delimiter(...)`,
  - optional header skipping is supported via `with_header(true)`,
  - empty/comment lines (`# ...`) are skipped,
  - values must parse to finite `f32` (`NaN`/`inf` rejected),
  - at least one valid sample row is required.
- Runtime mapping:
  - output `SupervisedDataset.inputs()` shape is `[N, input_shape...]`,
  - output `SupervisedDataset.targets()` shape is `[N, target_shape...]`.

## Supervised Training Image-Manifest CSV Mapping (`yscv-model`)
- Row schema (one sample per data line):
  - `[image_path, target_flat...]`
  - total column count must equal `1 + prod(target_shape)`.
- Parsing contract:
  - `SupervisedImageManifestConfig::new(target_shape, output_height, output_width)` defines expected target shape and output image resolution,
  - image path is loaded from column 1 and resolved against `image_root` unless it is absolute,
  - default delimiter is `,`; custom delimiter is supported via `with_delimiter(...)`,
  - optional header skipping is supported via `with_header(true)`,
  - empty/comment lines (`# ...`) are skipped,
  - image decode/open failures return typed `DatasetImageDecode`,
  - target values must parse to finite `f32` (`NaN`/`inf` rejected),
  - at least one valid sample row is required.
- Runtime mapping:
  - decoded image is converted to RGB and normalized to `[0, 1]`,
  - image tensor is resized (nearest-neighbor) to configured `(output_height, output_width)` when needed,
  - output `SupervisedDataset.inputs()` shape is `[N, output_height, output_width, 3]`,
  - output `SupervisedDataset.targets()` shape is `[N, target_shape...]`.

## Supervised Training Image-Folder Mapping (`yscv-model`)
- Directory schema:
  - `root/<class_name>/<image_file>`
  - each direct child directory under `root` is treated as one class bucket.
- Parsing contract:
  - `SupervisedImageFolderConfig::new(output_height, output_width)` defines output image resolution,
  - target encoding mode is configured via `with_target_mode(...)` (`ClassIndex` default, `OneHot` optional),
  - allowed image extensions can be customized via `with_allowed_extensions(vec![...])`,
  - class ids are assigned by deterministic lexicographic class-directory order (`0..num_classes-1`),
  - default image extensions are `jpg`, `jpeg`, `png`, `bmp`, `webp` (case-insensitive),
  - non-image files are ignored,
  - images are RGB-normalized to `[0, 1]` and resized (nearest-neighbor) to configured output shape,
  - at least one valid image sample is required.
- Runtime mapping:
  - output `SupervisedDataset.inputs()` shape is `[N, output_height, output_width, 3]`,
  - output `SupervisedDataset.targets()` shape depends on target mode:
    - `ClassIndex`: `[N, 1]` with scalar class ids as `f32`,
    - `OneHot`: `[N, class_count]` one-hot class vectors.
  - `load_supervised_image_folder_dataset_with_classes(...)` additionally returns deterministic `class_names` mapping aligned with class indices.

## YOLO Label-Dir Mapping
- Manifest format: one image per line, `<image_id> <width> <height>`.
- Ground-truth label file: `<class_id> <x_center> <y_center> <width> <height>`.
- Prediction label file: `<class_id> <x_center> <y_center> <width> <height> <score>`.
- Coordinates are normalized to `[0, 1]` and converted to runtime absolute boxes.
- Missing `<image_id>.txt` in GT/pred directories is treated as an empty frame side.
- Validation enforces finite values, positive image size, non-zero bbox size, and score in `[0, 1]`.

## VOC XML-Dir Mapping
- Manifest format: one image id per line.
- Ground-truth XML: Pascal VOC-like `<annotation><object><name|class_id>...<bndbox>...</bndbox></object></annotation>`.
- Prediction XML: same object+bbox format, optional `<score>` in each `<object>` (defaults to `1.0` when omitted).
- `name`/`class_id` is parsed as numeric class id (`usize`) in current baseline.
- Missing `<image_id>.xml` in GT/pred directories is treated as an empty frame side.
- Validation enforces finite bbox coordinates and strict bbox ordering (`xmax > xmin`, `ymax > ymin`).

## KITTI Label-Dir Mapping
- Manifest format: one image id per line.
- Label file format: KITTI object lines (space-separated), minimum expected fields:
  - `<type> ... <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> ...`
  - parser requires at least 8 fields and reads bbox from fields `4..7` (0-based indexing).
- Class mapping:
  - `<type>` values are mapped to runtime `class_id` dynamically and consistently across GT/pred files during one load call.
  - `DontCare` labels are skipped.
- Prediction score:
  - if present at KITTI score position (field 16, 1-based), it is used as detection score.
  - otherwise defaults to `1.0`.
- Missing `<image_id>.txt` in GT/pred directories is treated as an empty frame side.
- Validation enforces finite bbox values and strict bbox ordering (`bbox_right > bbox_left`, `bbox_bottom > bbox_top`).

## WIDER FACE TXT Mapping
- Input pair:
  - ground-truth TXT (`--eval-detection-widerface-gt`)
  - prediction TXT (`--eval-detection-widerface-pred`)
- Block format for each image:
  - `<image_id>`
  - `<box_count>`
  - repeated bbox rows
- Ground-truth bbox row:
  - at least 4 fields: `<x> <y> <w> <h>`
  - optional WIDER FACE attributes are accepted; when `invalid` attribute exists and is non-zero, the box is skipped.
- Prediction bbox row:
  - at least 5 fields: `<x> <y> <w> <h> <score>`
- Runtime mapping:
  - bbox conversion: `(x, y, w, h)` -> `(x1=x, y1=y, x2=x+w, y2=y+h)`
  - class mapping: both GT and predictions use face class (`CLASS_ID_FACE`) by contract.
- Validation enforces finite values, positive bbox size (`w > 0`, `h > 0`), non-negative scores, unique image IDs per side, and prediction image IDs present in GT.

## MOTChallenge TXT Mapping
- Input pair:
  - ground truth TXT (`--eval-tracking-mot-gt`)
  - prediction TXT (`--eval-tracking-mot-pred`)
- Expected line shape: CSV with at least first 6 MOT columns:
  - `<frame>,<track_id>,<left>,<top>,<width>,<height>,...`
- Ground-truth semantics:
  - optional confidence field (column 7): rows with `confidence <= 0` are ignored.
  - optional class field (column 8): mapped to `class_id` when positive, otherwise defaults to `0`.
- Prediction semantics:
  - optional score field (column 7): mapped to detection score, defaults to `1.0` when missing.
  - optional class field (column 8): mapped to `class_id` when positive, otherwise defaults to `0`.
- Frame indexing:
  - MOT 1-based frame ids are converted to 0-based frame slots in runtime.
  - missing intermediate frames are represented as empty frames on either side.
- Validation enforces:
  - finite numeric values,
  - `frame > 0`, `track_id > 0`,
  - positive bbox width/height.

## CLI Examples
```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --eval-detection-jsonl benchmarks/eval-detection-sample.jsonl \
  --eval-tracking-jsonl benchmarks/eval-tracking-sample.jsonl \
  --eval-iou 0.5 \
  --eval-score 0.0
```

```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --eval-detection-coco-gt benchmarks/eval-detection-coco-gt-sample.json \
  --eval-detection-coco-pred benchmarks/eval-detection-coco-pred-sample.json \
  --eval-iou 0.5 \
  --eval-score 0.0
```

```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --eval-detection-openimages-gt benchmarks/eval-detection-openimages-gt-sample.csv \
  --eval-detection-openimages-pred benchmarks/eval-detection-openimages-pred-sample.csv \
  --eval-iou 0.5 \
  --eval-score 0.0
```

```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --eval-detection-yolo-manifest benchmarks/eval-detection-yolo-manifest-sample.txt \
  --eval-detection-yolo-gt-dir benchmarks/eval-detection-yolo-gt \
  --eval-detection-yolo-pred-dir benchmarks/eval-detection-yolo-pred \
  --eval-iou 0.5 \
  --eval-score 0.0
```

```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --eval-detection-voc-manifest benchmarks/eval-detection-voc-manifest-sample.txt \
  --eval-detection-voc-gt-dir benchmarks/eval-detection-voc-gt \
  --eval-detection-voc-pred-dir benchmarks/eval-detection-voc-pred \
  --eval-iou 0.5 \
  --eval-score 0.0
```

```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --eval-detection-kitti-manifest benchmarks/eval-detection-kitti-manifest-sample.txt \
  --eval-detection-kitti-gt-dir benchmarks/eval-detection-kitti-gt \
  --eval-detection-kitti-pred-dir benchmarks/eval-detection-kitti-pred \
  --eval-iou 0.5 \
  --eval-score 0.0
```

```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --eval-detection-widerface-gt benchmarks/eval-detection-widerface-gt-sample.txt \
  --eval-detection-widerface-pred benchmarks/eval-detection-widerface-pred-sample.txt \
  --eval-iou 0.5 \
  --eval-score 0.0
```

```bash
cargo run -p yscv-cli --bin yscv-cli -- \
  --eval-tracking-mot-gt benchmarks/eval-tracking-mot-gt-sample.txt \
  --eval-tracking-mot-pred benchmarks/eval-tracking-mot-pred-sample.txt \
  --eval-iou 0.5 \
  --eval-score 0.0
```
