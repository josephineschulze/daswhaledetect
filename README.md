# DAS Whale Detect

A three-stage pipeline for whale-call detection from Distributed Acoustic Sensing (DAS) data.

## Usage

First, create and activate a conda environment:

```bash
conda create --name daswhaledetect python=3.10
conda activate daswhaledetect
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```


## Pipeline overview

1. **Preprocessing**: convert raw DAS recordings into FK-domain spectral feature windows.
2. **Training**: train a Random Forest classifier from labeled feature windows.
3. **Detection**: classify new preprocessed DAS windows.


## Files organization

Before running preprocessing, organize input files in one of the following layouts.

### Folder mode: raw DAS files

For folder-mode preprocessing, place raw DAS recording files inside date/day subdirectories:

```text
data_root/
├── 20220822/
│   ├── *.<file_type>
│   ├── *.<file_type>
│   └── *.<file_type>
├── 20220823/
│   ├── *.<file_type>
│   ├── *.<file_type>
│   └── *.<file_type>
└── 20220824/
    └── ...
```

Set `folder_mode.directory` in `configs/preprocessing.yaml` to `data_root`:

### List mode: annotation CSV files

For list-mode preprocessing, place annotation CSV files in one directory:

```text
annotations_root/
├── *.csv
├── *.csv
├── *.csv
└── ...
```

Set `list_mode.list_dir` in `configs/preprocessing.yaml` to `annotations_root`:


Each annotation CSV should follow the format of `example.csv`:

## Stage 1: Preprocessing

Please specify `mode` and corresponding input / output directory at first. Then run:

```bash
python preprocess.py --config configs/preprocessing_folder.yaml
```


## Stage 2: Training

Please specify `npz_dir` and `output_dir` at first and make sure to run training only with npz files extracted with `list` mode. Run:

```bash
python train.py --config configs/train.yaml
```

## Stage 3: Detection

Please specify `model_path`, `npz_dir`, and `output_dir` at first, then run:

```bash
python detect.py --config configs/detect.yaml
```
