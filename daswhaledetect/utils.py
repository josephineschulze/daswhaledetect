from collections import Counter
import logging
import numpy as np
from datetime import datetime
import os
import re
import yaml
import pandas as pd


logger = logging.getLogger("daswhaledetect.utils")

def load_config_preprocess(path):
    """Load and validate the preprocessing configuration."""

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required_sections = [
        "run_mode",
        "interrogator", 
        "channel_range",
        "n_workers",
        "output_root",
        "list_mode",
        "folder_mode",
        "features",
        "bad_channel_estimation",
    ]

    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing top-level key: {section}")

    if cfg["run_mode"] not in {"folder", "list"}:
        raise ValueError("run_mode must be either 'folder' or 'list'")

    # ---------- paths ----------
    if not cfg["output_root"]:
        raise ValueError("Missing paths.output_root")

    os.makedirs(cfg["output_root"], exist_ok=True)

    # ---------- input mode ----------
    if cfg["run_mode"] == "folder":
        required = ["folder_path", "folder_glob"]

        for key in required:
            if key not in cfg["folder_mode"]:
                raise ValueError(f"Missing folder_mode.{key}")

        if not os.path.isdir(cfg["folder_mode"]["folder_path"]):
            raise FileNotFoundError(
                f"folder_path does not exist: {cfg['folder_mode']['folder_path']}"
            )

    else:  # list mode
        required = ["list_dir", "list_glob"]

        for key in required:
            if key not in cfg["list_mode"]:
                raise ValueError(f"Missing list_mode.{key}")

        if not os.path.isdir(cfg["list_mode"]["list_dir"]):
            raise FileNotFoundError(
                f"list_dir does not exist: {cfg['list_mode']['list_dir']}"
            )

    return cfg

def load_config_train(path):
    """Load and validate the training configuration."""

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required_sections = [
        "paths",
        "features",
        "sampling",
        "classes",
        "plotting",
    ]

    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")

    required_path_keys = [
        "npz_dir",
        "output_dir",
        "model_name",
    ]

    for key in required_path_keys:
        if key not in cfg["paths"]:
            raise ValueError(f"Missing required config key: paths.{key}")

    required_feature_keys = [
        "feature_key",
        "freq_key",
        "feature_length",
    ]

    for key in required_feature_keys:
        if key not in cfg["features"]:
            raise ValueError(f"Missing required config key: features.{key}")

    return cfg

def load_config_detect(path):
    """Load and validate the detection configuration."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required_sections = [
        "paths", 
        "features", 
        "plotting", 
        "label_mapping"
    ]
    for key in required_sections:
        if key not in cfg:
            raise ValueError(f"Missing top-level key in config: {key}")

    for key in ["model_path", "unknown_npz_dir", "output_dir"]:
        if key not in cfg["paths"]:
            raise ValueError(f"Missing paths.{key} in config")

    if not os.path.exists(cfg["paths"]["model_path"]):
        raise FileNotFoundError(f"model_path does not exist: {cfg['paths']['model_path']}")
    if not os.path.isdir(cfg["paths"]["unknown_npz_dir"]):
        raise FileNotFoundError(f"unknown_npz_dir does not exist: {cfg['paths']['unknown_npz_dir']}")

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    return cfg


def safe_scalar_str(x):
    try:
        arr = np.array(x)
        if arr.shape == ():
            return str(arr.item()).strip()
        return str(x).strip()
    except Exception:
        return str(x).strip()
    
def safe_scalar_int(x, default=-1):
    try:
        arr = np.array(x)
        if arr.shape == ():
            return int(arr.item())
        return int(x)
    except Exception:
        return default
    
def parse_dt(s):
    """Try a list of common datetime string formats until one matches"""
    if s is None:
        return None
    s = str(s).strip().replace("T", " ")
    fmts = [
        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
        "%Y%m%d %H%M%S", "%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S",
        "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H-%M-%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def majority_vote(series): 
    # most common value in round of predicted labels, used for group-level aggregation
    counts = Counter(series)
    return counts.most_common(1)[0][0]


def sanitize_filename(s, max_len=220):
# turns arbitrary string into smoehting usable as a filename
    s = str(s)
    s = s.replace(os.sep, "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    return s[:max_len]

def clean_str(x):
    """Coerce anything, including None, into a stripped string."""
    if x is None:
        return ""
    return str(x).strip()

def parse_int_safe(x, default=None):
    """Parse something into an int, returning `default` instead of raising on failure."""
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x).strip()))
    except Exception:
        return default

def parse_float_safe(x, default=None):
    """Parse something into a float, returning 'default' instead of running into issues"""
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(str(x).strip())
    except Exception:
        return default


def normalize_seg_text(s):
    """Strip spaces/underscores and lowercase for easier comparison"""
    if s is None:
        return ""
    return str(s).strip().replace(" ", "").replace("_", "").lower()


def normalize_label(x):
    """lowercase/strip/remove spaces+underscores, so raw CallType values line up with calltype_to_class keys"""
    x = safe_scalar_str(x).strip().lower()
    x = x.replace("_", "")
    x = x.replace(" ", "")
    return x

