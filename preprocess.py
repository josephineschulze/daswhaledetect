""" stage 1 - preprocessing
reads raw DAS hdf5 file, matches windows against annotation list (in list mode, for training data) or 
processes every window directly without requiring annotations (folder mode, for new/unlabeled data headed to the detector stage 3)

extracts the fk based detrended spectrum feature or ach window, saved png diagnostics + npz outputs

multiprocessing pool parallelises across files
"""

import logging
import os
import csv
import traceback
import argparse
from daswhaledetect.dataload import discover_files_by_folder_and_time, discover_list_files, sanitize_filename, load_annotations_from_list_csv
from daswhaledetect.features import estimate_bad_channels, feature_extract
from daswhaledetect.utils import load_config_preprocess

logger = logging.getLogger("preprocess")


def run_files_through_pipeline(valid_files, output_base, run_label, tasks_by_file=None, cfg=None):
    if cfg is None:
        logger.error("cfg must be provided")
        return
    run_mode = cfg["run_mode"]
    png_dir = os.path.join(output_base, "pngs")
    npz_dir = os.path.join(output_base, "npz")
    bad_dir = os.path.join(output_base, "bad_channels")
    diag_dir = os.path.join(output_base, "bad_channels/diagnostics")
    for d in [png_dir, npz_dir, bad_dir, diag_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"{'=' * 16} PROCESSING: {run_label} (mode={run_mode}) {'=' * 17}")
    logger.info("=" * 80)
    logger.info(f"OUTPUT BASE: {output_base}")
    logger.info(f"Found {len(valid_files)} valid files to process.")

    files_by_day, bad_channels_by_group = estimate_bad_channels(valid_files, bad_dir, diag_dir, cfg)
    results = feature_extract(valid_files, tasks_by_file, run_mode, png_dir, npz_dir, bad_channels_by_group, cfg)

    ok = sum(1 for _, s, _ in results if s == "ok")
    err = sum(1 for _, s, _ in results if s == "error")
    skipped = sum(1 for _, s, _ in results if s == "skipped")
    missing = sum(1 for _, s, _ in results if s == "missing")

    summary_txt = os.path.join(output_base, f"{sanitize_filename(run_label)}_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Run label: {run_label}\n")
        f.write(f"Run mode: {run_mode}\n")
        f.write(f"Output base: {output_base}\n")
        f.write(f"OK: {ok}\n")
        f.write(f"Skipped: {skipped}\n")
        f.write(f"Missing: {missing}\n")
        f.write(f"Errors: {err}\n\n")
        for fp, status, msg in results:
            f.write(f"{status}\t{fp}\t{msg}\n")

    logger.info("=" * 80)
    logger.info(f"Run label: {run_label}")
    logger.info(f"OK: {ok}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Missing: {missing}")
    logger.info(f"Errors: {err}")
    logger.info(f"Summary written to: {summary_txt}")


def preprocess_folder(folder_path, folder_glob, output_root, cfg):
    logger.info(f"time range: {cfg['folder_mode'].get('start_dt')} to {cfg['folder_mode'].get('end_dt')}")
    
    files = discover_files_by_folder_and_time(folder_path, folder_glob, start_dt=cfg['folder_mode'].get('start_dt'), end_dt=cfg['folder_mode'].get('end_dt'))
    if not files:
        logger.warning(f"No files found for the requested folder / time range.")
        return
    else:
        logger.info(f"Found {len(files)} files to process.")

    output_base = os.path.join(output_root, sanitize_filename(cfg['folder_mode'].get("run_label")))
    os.makedirs(output_base, exist_ok=True)

    run_files_through_pipeline(files, output_base, run_label=cfg["folder_mode"]["run_label"], tasks_by_file=None, cfg=cfg)

def preprocess_list(list_dir, list_glob, output_root, cfg):
    list_files = discover_list_files(list_dir, list_glob)
    if not list_files:
        logger.warning(f"No list files found in the requested directory.")
        return
    else:
        logger.info(f"Found {len(list_files)} list files to process.")

    for i, p in enumerate(list_files, 1):
        logger.info(f"  {i:02d}. {p}")

    for list_csv_path in list_files:
        try:
            list_tag = sanitize_filename(os.path.splitext(os.path.basename(list_csv_path))[0])
            output_base = os.path.join(output_root, list_tag)
            os.makedirs(output_base, exist_ok=True)

            annotations_by_file = load_annotations_from_list_csv(
                list_csv_path, cfg['list_mode'].get("annotation_columns", ["file_path", "start_time", "end_time", "label"])
            )

            file_paths_list = sorted(annotations_by_file.keys())
            if not file_paths_list:
                logger.warning(f"No valid file paths found in list: {list_csv_path}")
                continue
            logger.info(f"Unique raw file paths in list: {len(file_paths_list)}")
            valid_files_to_process = []
            for fp in file_paths_list:
                if os.path.exists(fp):
                    valid_files_to_process.append(fp)
                else:
                    logger.warning(f"File path from list does not exist: {fp}")

            if not valid_files_to_process:
                logger.warning(f"All file paths from list do not exist")
                return
            os.makedirs(os.path.join(output_base, "summary"), exist_ok=True)
            manifest_csv = os.path.join(output_base, f"summary/{list_tag}_resolved_manifest.csv")
            with open(manifest_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source_list_file", "original_file_path", "n_annotations", "exists"])
                for fp in valid_files_to_process:
                    n_annots = len(annotations_by_file[fp])
                    writer.writerow([list_csv_path, fp, n_annots, int(os.path.exists(fp))])
            logger.info(f"Manifest written to: {manifest_csv}")

            run_files_through_pipeline(valid_files_to_process, output_base, list_tag, annotations_by_file, cfg=cfg)

        except Exception as e:
            logger.error(f"Error processing list file {list_csv_path}: {e}")
            logger.error(traceback.format_exc())

def run_preprocessing(cfg):
    """Run the preprocessing pipeline according to the provided configuration."""

    run_mode = cfg["run_mode"]
    output_root = cfg["output_root"]

    if run_mode == "folder":
        folder_path = cfg["folder_mode"]["folder_path"]
        folder_glob = cfg["folder_mode"]["folder_glob"]
        logger.info(f"Running in folder mode: {folder_path} with file type {folder_glob}")
        preprocess_folder(folder_path, folder_glob, output_root, cfg)

    elif run_mode == "list":
        list_dir = cfg["list_mode"]["list_dir"]
        list_glob = cfg["list_mode"]["list_glob"]
        logger.info(f"Running in list mode: {list_dir} with glob {list_glob}")
        preprocess_list(list_dir, list_glob, output_root, cfg)

    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")



def main():
    parser = argparse.ArgumentParser(description="Run stage 1 (preprocessing) of the DASWhaleDetect pipeline.")
    parser.add_argument("--config", default="configs/preprocessing.yaml", help="Path to a preprocessing YAML config file.")
    args = parser.parse_args()
    cfg = load_config_preprocess(args.config)

    log_path = os.path.join(cfg["output_root"], "preprocess.log")



    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]
        )
    

    run_preprocessing(cfg)


if __name__ == "__main__":
    main()