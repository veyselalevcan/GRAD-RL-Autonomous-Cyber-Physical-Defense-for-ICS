"""
train_grfics_hybrid.py — GRAD-RL Framework (v7.1)
GRFICS Digital-Twin Hybrid Training Pipeline

Dataset : grad_rl_final_dataset.csv  (90,728 rows, 13 features)
Labels  : "Normal" (97.4%) + 6 MITRE ATT&CK attack classes (2.6%)

Stage 1-2 │ LSTM-AE   → trained on 88,365 Normal rows → P99 threshold
Stage 3-4 │ XGBoost   → trained on  2,363 Attack rows → 6-class diagnosis

Saved artifacts (_grfics_live suffix):
    models/detective_lstm_grfics_live.keras
    models/threshold_grfics_live.json
    models/detective_classifier_grfics_live.pkl
    models/label_encoder_grfics_live.pkl
    docs/results/lstm_training_loss_grfics_live.png
    docs/results/confusion_matrix_lstm_grfics_live.png
    docs/results/confusion_matrix_rf_grfics_live.png
    docs/results/rf_classification_report_grfics_live.csv
"""

import json
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
from xgboost.callback import EarlyStopping as XGBEarlyStopping

from data_loader_grfics import load_and_preprocess, FEATURE_COLS

# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DS_SUFFIX      = "grfics_live"
TIME_STEPS     = 10          # LSTM sequence length
LSTM_EPOCHS    = 20          # max (EarlyStopping active, patience=3)
NORMAL_HOLDOUT = 5_000       # normal rows held out for balanced Stage 2 eval
MODEL_DIR      = "models"
RESULTS_DIR    = "docs/results"

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LSTM_PATH      = os.path.join(MODEL_DIR, f"detective_lstm_{DS_SUFFIX}.keras")
THRESHOLD_PATH = os.path.join(MODEL_DIR, f"threshold_{DS_SUFFIX}.json")
XGB_PATH       = os.path.join(MODEL_DIR, f"detective_classifier_{DS_SUFFIX}.pkl")
LE_PATH        = os.path.join(MODEL_DIR, f"label_encoder_{DS_SUFFIX}.pkl")


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def create_sequences(values: np.ndarray, time_steps: int = TIME_STEPS) -> np.ndarray:
    """(N, F) → (N - T, T, F) sliding-window sequences."""
    return np.stack([values[i : i + time_steps]
                     for i in range(len(values) - time_steps)])


def save_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.ylabel("Ground Truth"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path); plt.close()


# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────

def train_grfics():
    print(f"\n{'='*65}")
    print(f"  GRAD-RL GRFICS TRAINING PIPELINE")
    print(f"  Dataset  : grad_rl_final_dataset.csv")
    print(f"  Features : {len(FEATURE_COLS)}  → {FEATURE_COLS}")
    print(f"  Suffix   : {DS_SUFFIX}")
    print(f"{'='*65}\n")

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGE 1 — UNSUPERVISED PERCEPTION: LSTM-AE TRAINING
    #  Trained EXCLUSIVELY on Normal rows (label == "Normal").
    # ═══════════════════════════════════════════════════════════════════════
    print("─── Stage 1 │ Unsupervised Perception: LSTM-AE Training ───")

    X_normal, _, _ = load_and_preprocess(mode="train")

    # Reserve last NORMAL_HOLDOUT rows for balanced Stage 2 evaluation.
    # These are excluded from LSTM training to prevent evaluation leakage.
    if len(X_normal) > NORMAL_HOLDOUT:
        X_normal_train   = X_normal[:-NORMAL_HOLDOUT]
        X_normal_holdout = X_normal[-NORMAL_HOLDOUT:]
    else:
        split            = int(len(X_normal) * 0.9)
        X_normal_train   = X_normal[:split]
        X_normal_holdout = X_normal[split:]
        print(f"  ⚠️  Normal rows ({len(X_normal):,}) < holdout — 90/10 split applied.")

    print(f"  Normal train rows : {len(X_normal_train):,}")
    print(f"  Holdout (eval)    : {len(X_normal_holdout):,}")

    X_train_seq = create_sequences(X_normal_train)
    n_features  = X_train_seq.shape[2]   # 13 — inferred, not hardcoded
    print(f"  Sequences shape   : {X_train_seq.shape}")

    model = Sequential([
        LSTM(64, activation="relu",
             input_shape=(TIME_STEPS, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="relu", return_sequences=False),
        RepeatVector(TIME_STEPS),
        LSTM(32, activation="relu", return_sequences=True),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(n_features)),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    checkpoint_cb = ModelCheckpoint(
        LSTM_PATH, monitor="val_loss", save_best_only=True, verbose=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1,
    )

    print("\n  Training on Normal rows only...")
    history = model.fit(
        X_train_seq, X_train_seq,
        epochs=LSTM_EPOCHS, batch_size=64, validation_split=0.1,
        callbacks=[checkpoint_cb, early_stop_cb], verbose=1,
    )

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("LSTM-AE Training Loss [GRFICS]")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"lstm_training_loss_{DS_SUFFIX}.png"))
    plt.close()
    print(f"\n  ✅ Best LSTM-AE → {LSTM_PATH}")

    # ── P99 Anomaly Threshold ──────────────────────────────────────────────
    print("\n  Computing P99 anomaly threshold on normal training data...")
    recons    = model.predict(X_train_seq, batch_size=256, verbose=0)
    mse_norm  = np.mean(np.power(X_train_seq - recons, 2), axis=(1, 2))
    threshold = float(np.percentile(mse_norm, 99))

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold, "dataset": DS_SUFFIX,
                   "method": "percentile_99"}, f, indent=2)
    print(f"  ✅ Threshold (P99) : {threshold:.6f}")
    print(f"     [Ref] Max       : {mse_norm.max():.6f}")
    print(f"     [Ref] Mean+3σ   : {mse_norm.mean() + 3*mse_norm.std():.6f}")

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGE 2 — UNSUPERVISED PERCEPTION: LSTM-AE EVALUATION
    #
    #  Balanced test set = Normal holdout (y=0) + all Attack rows (y=1).
    #  Without the holdout the GRFICS attack file contains ONLY attacks,
    #  making FPR (False Positive Rate) unmeasurable.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n─── Stage 2 │ Unsupervised Perception: LSTM-AE Evaluation ───")

    X_attack_raw, y_attack_mc, y_attack_bin = load_and_preprocess(mode="test")

    y_holdout_bin  = np.zeros(len(X_normal_holdout), dtype=np.int32)
    X_combined     = np.vstack([X_normal_holdout, X_attack_raw])
    y_combined_bin = np.concatenate([y_holdout_bin, y_attack_bin])

    print(f"  Balanced eval — Normal(0): {(y_combined_bin==0).sum():,}, "
          f"Attack(1): {(y_combined_bin==1).sum():,}")

    X_comb_seq  = create_sequences(X_combined)
    y_comb_eval = y_combined_bin[TIME_STEPS - 1 : len(y_combined_bin) - 1]

    recons_comb = model.predict(X_comb_seq, batch_size=256, verbose=0)
    mse_comb    = np.mean(np.power(X_comb_seq - recons_comb, 2), axis=(1, 2))
    y_pred_lstm = (mse_comb > threshold).astype(int)

    print(f"\n  LSTM-AE BINARY ANOMALY DETECTION REPORT [GRFICS]")
    print("  " + "─" * 55)
    print(classification_report(
        y_comb_eval, y_pred_lstm,
        target_names=["Normal", "Anomaly"], digits=4,
    ))
    save_confusion_matrix(
        y_comb_eval, y_pred_lstm,
        labels=["Normal", "Anomaly"],
        title="LSTM-AE Anomaly Detection [GRFICS]",
        path=os.path.join(RESULTS_DIR, f"confusion_matrix_lstm_{DS_SUFFIX}.png"),
    )
    print("  ✅ LSTM-AE confusion matrix saved.")

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGE 3 — SUPERVISED DIAGNOSTIC: XGBOOST TRAINING
    #
    #  Classifies 6 MITRE ATT&CK attack types from the `label` column.
    #  Class-wise temporal split preserves chronological order per class.
    #
    #  IMBALANCE NOTE: Attack_ModbusFlood has only 24 samples (1% of attacks).
    #  compute_sample_weight("balanced") + clip(1.0) upweights it strongly.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n─── Stage 3 │ Supervised Diagnostic: XGBoost Training ───")

    tr_idx, te_idx = [], []
    for cls in np.unique(y_attack_mc):
        idx     = np.where(y_attack_mc == cls)[0]
        split_at = int(len(idx) * 0.7)
        tr_idx.extend(idx[:split_at].tolist())
        te_idx.extend(idx[split_at:].tolist())
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)

    X_rf_train = X_attack_raw[tr_idx];  y_rf_train = y_attack_mc[tr_idx]
    X_rf_test  = X_attack_raw[te_idx];  y_rf_test  = y_attack_mc[te_idx]

    print(f"  Class-wise temporal split — Train: {len(X_rf_train)}, Test: {len(X_rf_test)}")
    for split_name, y_split in [("train", y_rf_train), ("test", y_rf_test)]:
        u, c = np.unique(y_split, return_counts=True)
        print(f"  {split_name:5s}: " + "  ".join(f"c{cls}={cnt}" for cls, cnt in zip(u, c)))

    # Unified imbalance weighting — label-index agnostic
    sample_weights = np.clip(
        compute_sample_weight("balanced", y_rf_train), 1.0, None
    )
    for cls in np.unique(y_rf_train):
        w = sample_weights[y_rf_train == cls][0]
        n = int((y_rf_train == cls).sum())
        print(f"  Class {cls} (n={n:4d}): weight={w:.4f}"
              + (" ← upweighted" if w > 1.0 else " ← baseline"))

    n_classes   = len(np.unique(y_rf_train))
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    print(f"\n  n_classes={n_classes} → eval_metric='{eval_metric}'")

    clf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.1, 
        eval_metric=eval_metric,
        use_label_encoder=False,
        verbosity=0,
        random_state=SEED,
        callbacks=[XGBEarlyStopping(rounds=50, save_best=True)],
    )
    clf.fit(
        X_rf_train, y_rf_train,
        sample_weight=sample_weights,
        eval_set=[(X_rf_test, y_rf_test)],
        verbose=False,
    )
    print(f"  Best iteration : {clf.best_iteration}")
    joblib.dump(clf, XGB_PATH)
    print(f"  ✅ XGBoost saved → {XGB_PATH}")

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGE 4 — SUPERVISED DIAGNOSTIC: XGBOOST EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n─── Stage 4 │ Supervised Diagnostic: XGBoost Evaluation ───")

    y_pred_proba   = clf.predict_proba(X_rf_test)
    n_classes_test = y_pred_proba.shape[1]

    if n_classes_test == 2:
        # Binary threshold sweep — maximises F1 for minority attack class
        minority_class = int(y_rf_train[np.argmin(np.bincount(y_rf_train.astype(int)))])
        proba_min = y_pred_proba[:, minority_class]
        best_thr, best_f1 = 0.5, 0.0
        from sklearn.metrics import f1_score as _f1
        for thr in np.arange(0.05, 0.95, 0.01):
            y_tmp = ((proba_min >= thr).astype(int) * minority_class +
                     (proba_min <  thr).astype(int) * (1 - minority_class))
            if len(np.unique(y_tmp)) < 2:
                continue
            score = _f1(y_rf_test, y_tmp, pos_label=minority_class, zero_division=0)
            if score > best_f1:
                best_f1, best_thr = score, float(thr)
        y_pred_xgb = ((proba_min >= best_thr).astype(int) * minority_class +
                      (proba_min <  best_thr).astype(int) * (1 - minority_class))
        print(f"  Binary sweep → best_threshold={best_thr:.2f}, F1={best_f1:.4f}")
    else:
        y_pred_xgb = np.argmax(y_pred_proba, axis=1)
        print("  Multiclass mode: argmax prediction")

    le = joblib.load(LE_PATH) if os.path.exists(LE_PATH) else None
    observed_codes = sorted(set(np.unique(y_rf_test)) | set(np.unique(y_pred_xgb)))
    target_names   = (
        [str(le.classes_[c]) for c in observed_codes] if le else
        [f"class_{c}" for c in observed_codes]
    )
    print(f"  Classes in test  : {len(observed_codes)}/{n_classes}")
    for c, name in zip(observed_codes, target_names):
        n_gt = int((y_rf_test == c).sum()); n_pred = int((y_pred_xgb == c).sum())
        print(f"    [{c}] {name:<48s}  GT={n_gt:3d}  pred={n_pred:3d}")

    print(f"\n  XGBOOST ATTACK CLASSIFICATION REPORT [GRFICS]")
    print("  " + "─" * 55)
    print(classification_report(
        y_rf_test, y_pred_xgb,
        labels=observed_codes, target_names=target_names,
        digits=4, zero_division=0,
    ))
    pd.DataFrame(
        classification_report(y_rf_test, y_pred_xgb,
                              labels=observed_codes, target_names=target_names,
                              output_dict=True, zero_division=0)
    ).transpose().to_csv(
        os.path.join(RESULTS_DIR, f"rf_classification_report_{DS_SUFFIX}.csv")
    )
    save_confusion_matrix(
        y_rf_test, y_pred_xgb, labels=target_names,
        title="XGBoost Attack Classification [GRFICS]",
        path=os.path.join(RESULTS_DIR, f"confusion_matrix_rf_{DS_SUFFIX}.png"),
    )
    print("  ✅ Report and confusion matrix saved.")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE [GRFICS]")
    print(f"{'='*65}")
    print(f"\n  [Perception]  LSTM-AE")
    print(f"    {LSTM_PATH}")
    print(f"    {THRESHOLD_PATH}  (P99 = {threshold:.6f})")
    print(f"\n  [Diagnostic]  XGBoost  (best_iter={clf.best_iteration})")
    print(f"    {XGB_PATH}")
    print(f"\n  [Figures]  {RESULTS_DIR}/\n")


if __name__ == "__main__":
    train_grfics()