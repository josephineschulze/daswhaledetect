"""
Stage 3 run file (detector)
============================

Loads the saved random forest, predicts on a folder of new NPZ windows,
aggregates predictions to the group level, makes diagnostic stack
spectrogram plots, and (where applicable) evaluates predictions against
ground truth.

Usage:
    python detect.py --config configs/detection.yaml
"""

import argparse
import logging
import os
from glob import glob
import numpy as np
import logging

import joblib
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_fscore_support, classification_report
from daswhaledetect.utils import safe_scalar_str, safe_scalar_int, majority_vote



# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #

