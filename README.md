# DAS Whale Detect

A three-stage pipeline for whale-call detection from Distributed Acoustic Sensing (DAS) data.

## Pipeline overview

1. **Preprocessing**: convert raw DAS HDF5 recordings into FK-domain spectral feature windows.
2. **Training**: train a Random Forest classifier from labeled feature windows.
3. **Detection**: classify new preprocessed DAS windows.

## Repository structure

```text
.
├── configs/
│   ├── preprocessing_folder.yaml
│   ├── preprocessing_list.yaml
│   ├── train.yaml
│   └── detect.yaml
├── daswhaledetect/
│   ├── __init__.py
│   ├── dataload.py
│   ├── features.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── preprocess.py
├── train.py
├── detect.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All paths and processing parameters are stored in YAML files under `configs/`.

Before running the pipeline, update the input and output paths in the relevant YAML configuration file.

Example:

```yaml
paths:
  output_root: "G:/daswhaledetect/output"

folder_mode:
  folder_path: "//server/share/Svalbard/data"
  folder_glob: "*.hdf5"
  run_label: "svalbard_test"
```

## Stage 1: Preprocessing

Preprocessing converts raw DAS HDF5 recordings into compressed NPZ feature files.

For each file, the pipeline:

1. Selects DAS channels within the requested fiber-distance range.
2. Estimates persistently noisy channels.
3. Suppresses bad channels.
4. Resamples data to the configured sampling rate.
5. Splits the fiber into spatial segments.
6. Applies optional common-mode removal, channel-mean removal, and bandpass filtering.
7. Creates overlapping time windows.
8. Calculates FK-domain spectral features.
9. Saves one NPZ file per window and fiber segment.

The primary Random Forest input feature is:

```text
spectrum_after_detrended
```

### Folder mode

Folder mode processes all valid windows from raw HDF5 files in a folder. Use it to prepare unlabeled data for detection.

```bash
python preprocess.py --config configs/preprocessing_folder.yaml
```

### List mode

List mode processes only labeled windows specified in annotation CSV files. Use it to prepare training data.

```bash
python preprocess.py --config configs/preprocessing_list.yaml
```

## Annotation CSV format

Annotation CSV files for list mode should contain these columns:

```csv
npz_path,original_file_path,timestamp,window_index,segment_start_realtime,fiber_segment,CallType
```

Example:

```csv
npz_path,original_file_path,timestamp,window_index,segment_start_realtime,fiber_segment,CallType
,\\server\share\data\090007.hdf5,2022-08-22_09-00-07,0,2022-08-22 09:00:07,050-069km,20hz
,\\server\share\data\090007.hdf5,2022-08-22_09-00-07,1,2022-08-22 09:00:09,050-069km,20hz
,\\server\share\data\090017.hdf5,2022-08-22_09-00-17,0,2022-08-22 09:00:17,030-049km,background noise
```

One raw HDF5 file may appear in multiple rows because it can contain multiple labeled windows and fiber segments.

| Column | Description |
|---|---|
| `original_file_path` | Full path to the source raw HDF5 DAS file. |
| `timestamp` | Timestamp stored in the output metadata. |
| `window_index` | Index of the time window to process. |
| `fiber_segment` | Requested spatial fiber segment, for example `050-069km`. |
| `CallType` | Raw annotation label, for example `20hz`, `bwab`, or `background noise`. |
| `npz_path` | Optional legacy field; it can be blank when raw HDF5 files are used. |

## Preprocessing outputs

Each preprocessing run creates an output folder:

```text
output_root/
└── run_label/
    ├── bad_channels/
    ├── diagnostics/
    ├── npz/
    ├── pngs/
    ├── run_label_summary.txt
    └── preprocess.log
```

### `npz/`

Contains one processed feature file per saved window and spatial segment.

Each NPZ contains the main classifier feature:

```text
spectrum_after_detrended
```

It also includes metadata such as:

```text
freqs
CallType
timestamp
segment_label
window_index
window_start_s
window_end_s
bad_channel_indices
```

### `bad_channels/`

Contains bad-channel information for each file group/day:

```text
bad_channels_<group>.txt
bad_channels_<group>_summary.npz
```

The summary NPZ contains:

```text
bad_channel_indices
bad_channel_dist_m
frac_files_persistent
dist_m
n_sample_files
block_length_s
per_block_z_thresh
per_file_block_fraction_thresh
across_file_fraction_thresh
target_fs
```

### `diagnostics/`

Contains bad-channel diagnostic figures:

- persistent bad-channel score versus fiber distance;
- file-by-channel persistence matrix.

### `pngs/`

Contains optional per-window diagnostic figures:

- FK spectrum before and after background correction;
- detrended spectral feature plots.

Disable PNG generation for large folder-mode runs:

```yaml
output:
  save_pngs: false
```

### Manifest CSV

List-mode preprocessing also creates:

```text
<list_name>_resolved_manifest.csv
```

This records each raw HDF5 file referenced in an annotation list, the number of annotation rows associated with it, and whether the file exists.

## Stage 2: Training

Training loads labeled NPZ files, groups raw labels into model classes, balances classes, trains a Random Forest classifier, and calculates out-of-bag metrics.

```bash
python train.py --config configs/train.yaml
```

Training outputs include the trained model, metadata, confusion matrices, metrics tables, and diagnostic stacked-spectrum plots.

## Stage 3: Detection

Detection loads a trained Random Forest model and classifies preprocessed NPZ files.

```bash
python detect.py --config configs/detect.yaml
```

Detection outputs include window-level predictions, aggregated predictions, confusion matrices, metrics tables, and diagnostic plots.

## Typical workflow

```bash
# 1. Create labeled feature windows for model training.
python preprocess.py --config configs/preprocessing_list.yaml

# 2. Train the Random Forest model.
python train.py --config configs/train.yaml

# 3. Preprocess new unlabeled DAS data.
python preprocess.py --config configs/preprocessing_folder.yaml

# 4. Classify the preprocessed windows.
python detect.py --config configs/detect.yaml
```

## Notes

- Raw DAS HDF5 data are not included in this repository.
- Update all machine-specific paths in YAML configuration files before running.
- Folder-mode preprocessing can generate a large number of NPZ and PNG files.
- Use `save_pngs: false` for large preprocessing runs.
- A pretrained model may be used directly if retraining is not required.

## Citation

Add the relevant paper citation here.

```text
Citation information to be added.
```# DAS Whale Detect

A three-stage pipeline for whale-call detection from Distributed Acoustic Sensing (DAS) data.

## Pipeline overview

1. **Preprocessing**: convert raw DAS HDF5 recordings into FK-domain spectral feature windows.
2. **Training**: train a Random Forest classifier from labeled feature windows.
3. **Detection**: classify new preprocessed DAS windows.

## Repository structure

```text
.
├── configs/
│   ├── preprocessing_folder.yaml
│   ├── preprocessing_list.yaml
│   ├── train.yaml
│   └── detect.yaml
├── daswhaledetect/
│   ├── __init__.py
│   ├── dataload.py
│   ├── features.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── preprocess.py
├── train.py
├── detect.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All paths and processing parameters are stored in YAML files under `configs/`.

Before running the pipeline, update the input and output paths in the relevant YAML configuration file.

Example:

```yaml
paths:
  output_root: "G:/daswhaledetect/output"

folder_mode:
  folder_path: "//server/share/Svalbard/data"
  folder_glob: "*.hdf5"
  run_label: "svalbard_test"
```

## Stage 1: Preprocessing

Preprocessing converts raw DAS HDF5 recordings into compressed NPZ feature files.

For each file, the pipeline:

1. Selects DAS channels within the requested fiber-distance range.
2. Estimates persistently noisy channels.
3. Suppresses bad channels.
4. Resamples data to the configured sampling rate.
5. Splits the fiber into spatial segments.
6. Applies optional common-mode removal, channel-mean removal, and bandpass filtering.
7. Creates overlapping time windows.
8. Calculates FK-domain spectral features.
9. Saves one NPZ file per window and fiber segment.

The primary Random Forest input feature is:

```text
spectrum_after_detrended
```

### Folder mode

Folder mode processes all valid windows from raw HDF5 files in a folder. Use it to prepare unlabeled data for detection.

```bash
python preprocess.py --config configs/preprocessing_folder.yaml
```

### List mode

List mode processes only labeled windows specified in annotation CSV files. Use it to prepare training data.

```bash
python preprocess.py --config configs/preprocessing_list.yaml
```

## Annotation CSV format

Annotation CSV files for list mode should contain these columns:

```csv
npz_path,original_file_path,timestamp,window_index,segment_start_realtime,fiber_segment,CallType
```

Example:

```csv
npz_path,original_file_path,timestamp,window_index,segment_start_realtime,fiber_segment,CallType
,\\server\share\data\090007.hdf5,2022-08-22_09-00-07,0,2022-08-22 09:00:07,050-069km,20hz
,\\server\share\data\090007.hdf5,2022-08-22_09-00-07,1,2022-08-22 09:00:09,050-069km,20hz
,\\server\share\data\090017.hdf5,2022-08-22_09-00-17,0,2022-08-22 09:00:17,030-049km,background noise
```

One raw HDF5 file may appear in multiple rows because it can contain multiple labeled windows and fiber segments.

| Column | Description |
|---|---|
| `original_file_path` | Full path to the source raw HDF5 DAS file. |
| `timestamp` | Timestamp stored in the output metadata. |
| `window_index` | Index of the time window to process. |
| `fiber_segment` | Requested spatial fiber segment, for example `050-069km`. |
| `CallType` | Raw annotation label, for example `20hz`, `bwab`, or `background noise`. |
| `npz_path` | Optional legacy field; it can be blank when raw HDF5 files are used. |

## Preprocessing outputs

Each preprocessing run creates an output folder:

```text
output_root/
└── run_label/
    ├── bad_channels/
    ├── diagnostics/
    ├── npz/
    ├── pngs/
    ├── run_label_summary.txt
    └── preprocess.log
```

### `npz/`

Contains one processed feature file per saved window and spatial segment.

Each NPZ contains the main classifier feature:

```text
spectrum_after_detrended
```

It also includes metadata such as:

```text
freqs
CallType
timestamp
segment_label
window_index
window_start_s
window_end_s
bad_channel_indices
```

### `bad_channels/`

Contains bad-channel information for each file group/day:

```text
bad_channels_<group>.txt
bad_channels_<group>_summary.npz
```

The summary NPZ contains:

```text
bad_channel_indices
bad_channel_dist_m
frac_files_persistent
dist_m
n_sample_files
block_length_s
per_block_z_thresh
per_file_block_fraction_thresh
across_file_fraction_thresh
target_fs
```

### `diagnostics/`

Contains bad-channel diagnostic figures:

- persistent bad-channel score versus fiber distance;
- file-by-channel persistence matrix.

### `pngs/`

Contains optional per-window diagnostic figures:

- FK spectrum before and after background correction;
- detrended spectral feature plots.

Disable PNG generation for large folder-mode runs:

```yaml
output:
  save_pngs: false
```

### Manifest CSV

List-mode preprocessing also creates:

```text
<list_name>_resolved_manifest.csv
```

This records each raw HDF5 file referenced in an annotation list, the number of annotation rows associated with it, and whether the file exists.

## Stage 2: Training

Training loads labeled NPZ files, groups raw labels into model classes, balances classes, trains a Random Forest classifier, and calculates out-of-bag metrics.

```bash
python train.py --config configs/train.yaml
```

Training outputs include the trained model, metadata, confusion matrices, metrics tables, and diagnostic stacked-spectrum plots.

## Stage 3: Detection

Detection loads a trained Random Forest model and classifies preprocessed NPZ files.

```bash
python detect.py --config configs/detect.yaml
```

Detection outputs include window-level predictions, aggregated predictions, confusion matrices, metrics tables, and diagnostic plots.

## Typical workflow

```bash
# 1. Create labeled feature windows for model training.
python preprocess.py --config configs/preprocessing_list.yaml

# 2. Train the Random Forest model.
python train.py --config configs/train.yaml

# 3. Preprocess new unlabeled DAS data.
python preprocess.py --config configs/preprocessing_folder.yaml

# 4. Classify the preprocessed windows.
python detect.py --config configs/detect.yaml
```

## Notes

- Raw DAS HDF5 data are not included in this repository.
- Update all machine-specific paths in YAML configuration files before running.
- Folder-mode preprocessing can generate a large number of NPZ and PNG files.
- Use `save_pngs: false` for large preprocessing runs.
- A pretrained model may be used directly if retraining is not required.

## Citation

Add the relevant paper citation here.

```text
Citation information to be added.
```