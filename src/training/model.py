"""trains the random forest (RF) classifier and extracts out-of-bag (OOB)
predictions for evaluation without needing a separate held-out test set"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, rf_params=None, random_state=42):
    """
    trains a RandomForestClassifier with defaults tuned from prior
    hyperparameter search (500 trees, depth 10, balanced_subsample
    weighting, OOB scoring on) pass rf_params to override any of these.
    """
    default_params = dict(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=4,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )
    if rf_params:
        default_params.update(rf_params)

    rf = RandomForestClassifier(**default_params)
    rf.fit(X_train, y_train)
    return rf

def get_oob_predictions(rf, y_train):
    """
    extracts out-of-bag predictions from an already-fit RF (must have been
    trained with oob_score=True)
    
    some rows never end up out-of-bag for any
    tree (if their oob_decision_function_ row is all NaN/zero) -> those get
    dropped here rather than treated as a real prediction.

    returns (y_oob_true, y_oob_pred, oob_probs_valid, rf_classes)
    """
    oob_probs = rf.oob_decision_function_
    rf_classes = rf.classes_

    valid_oob_mask = np.isfinite(oob_probs).all(axis=1) & (oob_probs.sum(axis=1) > 0)

    y_oob_true = np.asarray(y_train)[valid_oob_mask]
    oob_probs_valid = oob_probs[valid_oob_mask]
    y_oob_pred = rf_classes[np.argmax(oob_probs_valid, axis=1)]

    return y_oob_true, y_oob_pred, oob_probs_valid, rf_classes