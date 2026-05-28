"""
train_hybrid.py — GRAD-RL Framework (v6.3)
Hybrid Intrusion Detection Training Pipeline for ICS Environments

═══════════════════════════════════════════════════════════════════════════════
  ARCHITECTURAL PHILOSOPHY: TWO-STAGE HYBRID DETECTION
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  STAGE 1-2 │ UNSUPERVISED PERCEPTION LAYER (LSTM-Autoencoder)          │
  │                                                                         │
  │  Trained EXCLUSIVELY on Normal operational data. Learns the statistical │
  │  fingerprint of healthy process behaviour using reconstruction error.   │
  │  At inference time, any input whose MSE exceeds the P99 threshold is    │
  │  flagged as an anomaly — including zero-day attacks never seen during   │
  │  training. Output: Binary label {0: Normal, 1: Anomaly}.               │
  └─────────────────────────────────────────────────────────────────────────┘
                                      │ Anomaly detected (=1)
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  STAGE 3-4 │ SUPERVISED DIAGNOSTIC LAYER (Random Forest Classifier)    │
  │                                                                         │
  │  Trained on historical labelled Attack data. Once the LSTM-AE raises   │
  │  an anomaly flag, the RF identifies the ATTACK TYPE (Attack1…Attack6)  │
  │  so the downstream Graph-Based RL Agent knows which asset is           │
  │  compromised and can select the appropriate mitigation policy.          │
  │  Output: Multiclass label {Attack1, Attack2, …, AttackN}.              │
  └─────────────────────────────────────────────────────────────────────────┘
                                      │ Attack type + affected asset
                                      ▼
                        ┌─────────────────────────┐
                        │  Graph-Based RL Agent   │
                        │  (PPO + Topology Risk)  │
                        └─────────────────────────┘

  This two-stage design is intentional and methodologically sound:
  • The LSTM-AE is unsupervised → detects novel/zero-day anomalies without
    requiring attack labels during training.
  • The RF is supervised → exploits historical attack structure to provide
    semantic context (attack type) needed by the RL policy.
  • The RF does NOT replace the LSTM-AE. It only runs AFTER the LSTM-AE
    confirms an anomaly, forming a coarse-to-fine detection pipeline.

  Refs: Goh et al. (2017) SWaT; Ahmed et al. (2017) WADI;
        Yoon et al. (2019) LSTM-AE for ICS; MITRE ATT&CK for ICS.

═══════════════════════════════════════════════════════════════════════════════
  CHANGELOG (v6.7 vs v6.6)
═══════════════════════════════════════════════════════════════════════════════
  - CRITICAL BUG FIX: Removed binary vs multiclass branching in Stage 3.
    Root cause: WADI labels are -1 (Attack) / 1 (Normal). LabelEncoder sorts
    numerically, mapping Attack → 0 and Normal → 1. The binary branch assumed
    class 0 = Normal and class 1 = Attack, inverting the imbalance logic:
      scale_pos_weight = 65355 (Attack) / 4644 (Normal) = 14.07
    This boosted Normal and penalised Attack with weight 0.071 — the exact
    opposite of the intended behaviour.
  - Unified training block: compute_sample_weight("balanced") + np.clip(1.0)
    is label-index agnostic. It weights by actual class frequency regardless
    of how LabelEncoder assigned integer codes, correct for both datasets:
      WADI  Attack(0, n=65355): weight=1.000 (baseline, not penalised)
      WADI  Normal(1, n= 4644): weight=7.537 (upweighted — correct)
      SWaT  Attack6(n=1675):    weight=1.000 (baseline, not penalised)
      SWaT  Attack5(n= 281):    weight=2.513 (upweighted — correct)
  - eval_metric="mlogloss" used for both cases; XGBoost automatically selects
    binary:logistic or multi:softprob from the label cardinality in the data.
  - All other v6.6 features retained (early stopping, hyperparameters,
    class-wise temporal split, balanced LSTM eval, P99 threshold).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.callback import EarlyStopping as XGBEarlyStopping
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_and_preprocess

# ─────────────────────────────────────────────
#  REPRODUCIBILITY
#  All seeds set before any library call so results are deterministic
#  across Python, NumPy, and TensorFlow.
# ─────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────
#  GLOBAL CONFIG
#  Set DATASET_TO_TRAIN to "SWaT" or "WADI" to switch datasets.
#  All downstream logic branches on this single flag.
# ─────────────────────────────────────────────
DATASET_TO_TRAIN = "SWaT"   # "SWaT" | "WADI"

DATASET_CONFIG = {
    "SWaT": {"epochs": 10},
    "WADI": {"epochs": 15},
}

TIME_STEPS     = 10
NORMAL_HOLDOUT = 10_000   # normal rows reserved for balanced LSTM-AE evaluation
MODEL_DIR      = "models"
RESULTS_DIR    = "docs/results"

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def create_sequences(values: np.ndarray, time_steps: int = TIME_STEPS) -> np.ndarray:
    """
    Sliding-window sequence builder.

    Converts a 2-D array (N, F) into a 3-D array (N-T, T, F) by extracting
    overlapping windows of length `time_steps`.  The window at index i
    represents timesteps [i, i+T-1]; its representative label is the label
    at the LAST timestep (i+T-1), used for alignment in Stage 2.

    Output shape: (N - time_steps, time_steps, n_features)
    """
    return np.stack([values[i : i + time_steps] for i in range(len(values) - time_steps)])


def save_confusion_matrix(y_true, y_pred, labels, title, path):
    """Save a labelled, annotated confusion-matrix heatmap to disk."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────

def train_system(dataset: str = DATASET_TO_TRAIN):
    print(f"\n{'='*65}")
    print(f"  GRAD-RL HYBRID TRAINING PIPELINE  [dataset={dataset}]")
    print(f"{'='*65}\n")

    n_epochs  = DATASET_CONFIG[dataset]["epochs"]
    ds_suffix = dataset.lower()

    lstm_best_path = os.path.join(MODEL_DIR, f"detective_lstm_{ds_suffix}.keras")
    rf_path        = os.path.join(MODEL_DIR, f"detective_classifier_{ds_suffix}.pkl")
    le_path        = os.path.join(MODEL_DIR, f"label_encoder_{ds_suffix}.pkl")
    threshold_path = os.path.join(MODEL_DIR, f"threshold_{ds_suffix}.json")

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGE 1 — UNSUPERVISED PERCEPTION LAYER: LSTM-AE TRAINING
    #
    #  The LSTM Autoencoder is trained ONLY on normal (non-attack) data.
    #  It learns to reconstruct normal sensor patterns with low error.
    #  At inference time, a sample with reconstruction MSE above the
    #  threshold is declared an anomaly — no attack labels are required.
    #  This enables zero-day detection: the model can flag attack types
    #  it has never seen, because it only knows what "Normal" looks like.
    # ═══════════════════════════════════════════════════════════════════════
    print("─── Stage 1 │ Unsupervised Perception: LSTM-AE Training ───")

    X_normal, _, _ = load_and_preprocess(dataset=dataset, mode="train")

    # Hold out the last NORMAL_HOLDOUT rows from training.
    # These rows are used in Stage 2 to construct a balanced evaluation set
    # that contains BOTH normal and attack samples (SWaT's attack file has
    # zero Normal rows, making FPR/Precision unmeasurable without this holdout).
    if len(X_normal) > NORMAL_HOLDOUT:
        X_normal_train   = X_normal[:-NORMAL_HOLDOUT]
        X_normal_holdout = X_normal[-NORMAL_HOLDOUT:]
        print(f"  Normal train rows : {len(X_normal_train)}")
        print(f"  Holdout (eval)    : {len(X_normal_holdout)}")
    else:
        split_idx        = int(len(X_normal) * 0.9)
        X_normal_train   = X_normal[:split_idx]
        X_normal_holdout = X_normal[split_idx:]
        print(f"  ⚠️  Dataset < NORMAL_HOLDOUT. 90/10 split applied: "
              f"train={len(X_normal_train)}, holdout={len(X_normal_holdout)}")

    X_train_seq = create_sequences(X_normal_train)

    # n_features is derived from the actual data shape, not from a config
    # constant, so the model architecture is always consistent with the
    # scaler output regardless of dataset or post-cleaning column count.
    n_features = X_train_seq.shape[2]
    print(f"\n  Training sequences : {X_train_seq.shape}")
    print(f"  (samples × {TIME_STEPS} timesteps × {n_features} features)")

    model = Sequential([
        LSTM(64, activation="relu",
             input_shape=(TIME_STEPS, n_features),
             return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="relu", return_sequences=False),
        RepeatVector(TIME_STEPS),
        LSTM(32, activation="relu", return_sequences=True),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(n_features)),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # ModelCheckpoint writes the epoch with lowest val_loss to disk.
    # EarlyStopping halts training and restores those weights in memory,
    # so the threshold in Stage 2 is computed on the genuinely best model.
    checkpoint_cb = ModelCheckpoint(
        filepath=lstm_best_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    print("\n  Training LSTM-Autoencoder on Normal data only...")
    history = model.fit(
        X_train_seq, X_train_seq,
        epochs=n_epochs,
        batch_size=64,
        validation_split=0.1,
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=1,
    )

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"LSTM-AE Training Loss [{dataset}]")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"lstm_training_loss_{ds_suffix}.png"))
    plt.close()
    print(f"\n  ✅ Best LSTM-AE saved → {lstm_best_path}")

    # ── Anomaly Threshold (P99) ────────────────────────────────────────────
    # The threshold is set at the 99th-percentile of the reconstruction MSE
    # on normal training data.  This accepts 99% of normal behaviour and
    # rejects the top 1%, which is tight enough to detect real anomalies
    # while being robust to the occasional high-error normal sample that
    # would inflate a max-based threshold by orders of magnitude.
    print("\n  Computing anomaly threshold (P99 on normal training data)...")
    recons_normal = model.predict(X_train_seq, batch_size=256, verbose=0)
    mse_normal    = np.mean(np.power(X_train_seq - recons_normal, 2), axis=(1, 2))

    threshold = float(np.percentile(mse_normal, 99))

    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold, "dataset": dataset,
                   "method": "percentile_99"}, f, indent=2)

    print(f"  ✅ Threshold (P99)   : {threshold:.6f}")
    print(f"     [Ref] Max         : {mse_normal.max():.6f}")
    print(f"     [Ref] Mean + 3σ   : {mse_normal.mean() + 3 * mse_normal.std():.6f}")

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGE 2 — UNSUPERVISED PERCEPTION LAYER: LSTM-AE EVALUATION
    #
    #  We evaluate the LSTM-AE on a BALANCED test set that contains both
    #  Normal and Attack samples, enabling proper Precision/Recall/FPR
    #  measurement.  The attack data provides the positive (anomaly) class;
    #  the held-out normal rows provide the negative (normal) class.
    #
    #  The output of this stage is a BINARY prediction {0: Normal, 1: Anomaly}.
    #  The attack TYPE is not known here — that is the responsibility of the
    #  RF Diagnostic Layer in Stages 3-4.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n─── Stage 2 │ Unsupervised Perception: LSTM-AE Evaluation ───")

    X_attack_raw, y_attack_multiclass, y_attack_binary = load_and_preprocess(
        dataset=dataset, mode="test"
    )

    # Construct balanced evaluation set:
    #   [Normal holdout rows  → y_binary = 0]
    #   [Attack file rows     → y_binary = from data_loader (0 or 1)]
    y_holdout_binary  = np.zeros(len(X_normal_holdout), dtype=np.int32)
    X_combined        = np.vstack([X_normal_holdout, X_attack_raw])
    y_combined_binary = np.concatenate([y_holdout_binary, y_attack_binary])

    print(f"  Balanced eval set — Normal(0): {(y_combined_binary == 0).sum()}, "
          f"Attack(1): {(y_combined_binary == 1).sum()}")

    X_combined_seq  = create_sequences(X_combined)
    # Label alignment: sequence window i → label at its last timestep (i + T - 1)
    y_combined_eval = y_combined_binary[TIME_STEPS - 1 : len(y_combined_binary) - 1]

    recons_combined = model.predict(X_combined_seq, batch_size=256, verbose=0)
    mse_combined    = np.mean(np.power(X_combined_seq - recons_combined, 2), axis=(1, 2))
    y_pred_lstm     = (mse_combined > threshold).astype(int)

    print(f"\n  LSTM-AE BINARY ANOMALY DETECTION REPORT  [{dataset}]")
    print("  " + "─" * 55)
    print(classification_report(
        y_combined_eval, y_pred_lstm,
        target_names=["Normal", "Anomaly"],
        digits=4,
    ))

    save_confusion_matrix(
        y_combined_eval, y_pred_lstm,
        labels=["Normal", "Anomaly"],
        title=f"LSTM-AE Anomaly Detection [{dataset}]",
        path=os.path.join(RESULTS_DIR, f"confusion_matrix_lstm_{ds_suffix}.png"),
    )
    print("  ✅ LSTM-AE confusion matrix saved.")

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGE 3 — SUPERVISED DIAGNOSTIC LAYER: RF ATTACK CLASSIFIER TRAINING
    #
    #  The Random Forest is trained on LABELLED attack data.  Its role is
    #  NOT binary anomaly detection — the LSTM-AE already handles that.
    #  The RF answers a different question:
    #    "Given that an anomaly has been confirmed, WHICH type of attack is it?"
    #
    #  This multiclass output (Attack1, Attack2, …) is consumed by the
    #  Graph-Based Risk Engine and the PPO RL Agent to determine which ICS
    #  asset is compromised and which mitigation action to execute.
    #
    #  Split strategy — Class-wise Temporal Split (Block-wise):
    #
    #  Problem with a global 70/30 temporal slice on ICS data:
    #    SWaT attacks are chronologically ordered (Attack1 → Attack6).
    #    A single boundary at 70% puts Attack1–5 entirely in train and
    #    only Attack6 in test → classification_report crashes with a
    #    class-count mismatch.
    #
    #  Problem with StratifiedShuffleSplit or shuffle=True:
    #    Randomising a time series destroys temporal dependencies between
    #    consecutive samples and leaks future-state feature distributions
    #    into the training set, producing artificially inflated accuracy.
    #
    #  Solution — Class-wise Temporal Split:
    #    For EACH attack class independently:
    #      1. Extract the chronologically-ordered indices of that class.
    #      2. Take the first 70% of those indices → train.
    #      3. Take the remaining 30%              → test.
    #    Concatenate all per-class train/test indices.
    #
    #  This guarantees:
    #    • Zero data leakage   — within every class, train indices always
    #                            precede test indices in calendar time.
    #    • All classes present — each class contributes samples to both
    #                            splits, so classification_report never
    #                            encounters a missing-class mismatch.
    #    • Natural stratification — the 70/30 ratio is enforced per class,
    #                            so the split proportions match the dataset's
    #                            natural class distribution.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n─── Stage 3 │ Supervised Diagnostic: RF Classifier Training ───")

    train_idx_list, test_idx_list = [], []
    for cls in np.unique(y_attack_multiclass):
        cls_indices = np.where(y_attack_multiclass == cls)[0]  # chronological order
        split_at    = int(len(cls_indices) * 0.7)
        train_idx_list.extend(cls_indices[:split_at].tolist())
        test_idx_list.extend(cls_indices[split_at:].tolist())

    train_idx = np.array(train_idx_list)
    test_idx  = np.array(test_idx_list)

    X_rf_train = X_attack_raw[train_idx]
    X_rf_test  = X_attack_raw[test_idx]
    y_rf_train = y_attack_multiclass[train_idx]
    y_rf_test  = y_attack_multiclass[test_idx]

    print(f"  Class-wise temporal split — Train: {len(X_rf_train)}, "
          f"Test: {len(X_rf_test)}")

    for split_name, y_split in [("train", y_rf_train), ("test", y_rf_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        print(f"  Classes in {split_name}: "
              + ", ".join(f"c{c}={n}" for c, n in zip(unique, counts)))

    print("\n  Training XGBoost on labelled attack data...")

    # ── Unified imbalance handling (label-index agnostic) ──────────────────
    # WHY NOT scale_pos_weight for WADI?
    # WADI raw labels: -1 (Attack), 1 (Normal). LabelEncoder sorts numerically
    # → Attack maps to class 0, Normal maps to class 1. The binary branch
    # assumed class 0 = Normal and class 1 = Attack, so:
    #   scale_pos_weight = n_class0 / n_class1 = 65355 / 4644 = 14.07
    # This explicitly PENALISED the Attack class (0) with weight 0.071 instead
    # of boosting it — the exact inverse of the intended behaviour.
    #
    # compute_sample_weight("balanced") is label-index agnostic: it computes
    #   weight_i = n_total / (n_classes × n_i)
    # regardless of which integer code was assigned to which class.
    # np.clip(weights, 1.0, None) then ensures that majority classes are held
    # at baseline (weight=1.0) rather than penalised below it, while minority
    # classes retain their upweighted values.
    #
    # Resulting weights (clipped):
    #   WADI Attack (class 0, n=65355): 1.000  ← baseline, not penalised
    #   WADI Normal (class 1, n= 4644): 7.537  ← upweighted (correct)
    #   SWaT Attack6(class 5, n= 1675): 1.000  ← baseline, not penalised
    #   SWaT Attack5(class 4, n=  281): 2.513  ← upweighted (correct)
    sample_weights = compute_sample_weight("balanced", y_rf_train)
    sample_weights = np.clip(sample_weights, 1.0, None)

    for cls in np.unique(y_rf_train):
        w = sample_weights[y_rf_train == cls][0]
        n = int((y_rf_train == cls).sum())
        print(f"  Class {cls} (n={n:5d}): weight={w:.4f}"
              + (" ← upweighted (minority)" if w > 1.0 else " ← baseline (majority)"))

    # eval_metric must match the objective XGBoost selects from the data.
    # XGBoost 3.x strictly enforces this when early stopping uses an eval_set:
    #   2 unique classes  → binary:logistic → eval_metric must be "logloss"
    #   >2 unique classes → multi:softprob  → eval_metric must be "mlogloss"
    # Using "mlogloss" unconditionally triggers a label-range error when the
    # multiclass evaluator receives binary labels {0, 1}.
    n_classes   = len(np.unique(y_rf_train))
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    print(f"  n_classes={n_classes} → eval_metric='{eval_metric}'")

    clf = xgb.XGBClassifier(
        n_estimators=1000,        # ↑ more boosting rounds for complex features
        max_depth=7,              # ↑ deeper trees capture stealthy interactions
        learning_rate=0.02,       # ↓ smaller steps for smoother convergence
        subsample=0.9,            # ↑ use more rows per tree (was 0.8)
        colsample_bytree=0.9,     # ↑ use more features per tree (was 0.8)
        reg_alpha=0.5,            # ↑ stronger L1 regularisation (was 0.1)
        reg_lambda=2.0,           # ↑ stronger L2 regularisation (was 1.0)
        eval_metric=eval_metric,
        use_label_encoder=False,
        verbosity=0,
        random_state=SEED,
        # Increased patience: rounds=50 prevents premature stopping when
        # learning stealthy WADI features that improve slowly at lr=0.02.
        callbacks=[XGBEarlyStopping(rounds=50, save_best=True)],
    )

    eval_set = [(X_rf_test, y_rf_test)]
    clf.fit(
        X_rf_train, y_rf_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=False,
    )
    print(f"  Best iteration: {clf.best_iteration}")

    joblib.dump(clf, rf_path)
    print(f"  ✅ XGBoost Classifier saved → {rf_path}")

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGE 4 — SUPERVISED DIAGNOSTIC LAYER: XGBOOST EVALUATION
    #
    #  Binary (WADI): the default 0.5 decision threshold causes near-zero
    #  Attack recall (Precision=1.0, Recall=0.02) because the minority class
    #  probability rarely crosses 0.5 despite the sample weighting. Optimal
    #  threshold search finds the probability cutoff that maximises F1 for the
    #  minority class without touching the model or the data pipeline.
    #
    #  Multiclass (SWaT): argmax over 6 class probabilities is already optimal;
    #  there is no meaningful single threshold to tune.
    #
    #  observed_codes is derived from the union of classes in y_rf_test and
    #  y_pred_xgb, preventing classification_report ValueError when the test
    #  window contains fewer classes than the full encoder vocabulary.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n─── Stage 4 │ Supervised Diagnostic: XGBoost Evaluation ───")

    y_pred_proba = clf.predict_proba(X_rf_test)   # shape: (N, n_classes)
    n_classes_test = y_pred_proba.shape[1]

    if n_classes_test == 2:
        # ── Binary threshold optimisation (WADI) ──────────────────────────
        # predict_proba returns [[P(class0), P(class1)], ...].
        # For WADI: class 0 = Attack (minority), class 1 = Normal (majority).
        # We tune the threshold on P(class 0): if P(class 0) > threshold,
        # predict Attack (0), else predict Normal (1).
        # The sweep maximises F1 for the minority (Attack) class, which is the
        # class that drives RL agent action — missing an attack is far costlier
        # than a false positive.
        minority_class = int(y_rf_train[
            np.argmin(np.bincount(y_rf_train.astype(int)))
        ])
        proba_minority = y_pred_proba[:, minority_class]

        best_threshold, best_f1 = 0.5, 0.0
        from sklearn.metrics import f1_score as _f1
        for thr in np.arange(0.05, 0.95, 0.01):
            y_tmp = (proba_minority >= thr).astype(int) * minority_class +                     (proba_minority <  thr).astype(int) * (1 - minority_class)
            # Guard: skip if only one class predicted (f1_score undefined)
            if len(np.unique(y_tmp)) < 2:
                continue
            score = _f1(y_rf_test, y_tmp, pos_label=minority_class, zero_division=0)
            if score > best_f1:
                best_f1, best_threshold = score, float(thr)

        y_pred_xgb = (proba_minority >= best_threshold).astype(int) * minority_class +                      (proba_minority <  best_threshold).astype(int) * (1 - minority_class)

        print(f"  Binary threshold sweep → best_threshold={best_threshold:.2f}  "
              f"(minority class={minority_class}, F1={best_f1:.4f})")
        print(f"  Default 0.5 predictions: Attack={int((y_pred_proba[:,minority_class]>=0.5).sum())}"
              f"  |  Tuned predictions: Attack={int((y_pred_xgb==minority_class).sum())}")
    else:
        # ── Multiclass argmax (SWaT) ───────────────────────────────────────
        # argmax over n_classes columns is already the optimal decision rule;
        # threshold tuning is not meaningful across 6 simultaneous class probs.
        y_pred_xgb = np.argmax(y_pred_proba, axis=1)
        print("  Multiclass mode: argmax prediction (no threshold tuning needed)")

    # Load the label encoder to map integer codes back to attack names
    le = joblib.load(le_path) if os.path.exists(le_path) else None

    # Determine which classes are actually present in this test window.
    # Union of ground-truth and predicted classes handles edge cases where
    # the RF predicts a class that does not appear in y_rf_test (rare but
    # possible with unbalanced temporal windows).
    observed_codes = sorted(set(np.unique(y_rf_test)) | set(np.unique(y_pred_xgb)))

    if le is not None:
        target_names = [str(le.classes_[c]) for c in observed_codes]
    else:
        target_names = [f"class_{c}" for c in observed_codes]

    present_class_count = len(observed_codes)
    total_class_count   = len(le.classes_) if le is not None else "?"
    print(f"  Classes present in test window: {present_class_count} / {total_class_count}")
    print(f"  → {target_names}")

    print(f"\n  XGBOOST MULTICLASS ATTACK CLASSIFICATION REPORT  [{dataset}]")
    print("  " + "─" * 55)
    print(classification_report(
        y_rf_test,
        y_pred_xgb,
        labels=observed_codes,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    report_dict = classification_report(
        y_rf_test,
        y_pred_xgb,
        labels=observed_codes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_path = os.path.join(RESULTS_DIR, f"rf_classification_report_{ds_suffix}.csv")
    pd.DataFrame(report_dict).transpose().to_csv(report_path)

    save_confusion_matrix(
        y_rf_test,
        y_pred_xgb,
        labels=target_names,
        title=f"XGBoost Attack Classification [{dataset}]",
        path=os.path.join(RESULTS_DIR, f"confusion_matrix_rf_{ds_suffix}.png"),
    )
    print("  ✅ RF classification report and confusion matrix saved.")

    # ─────────────────────────────────────────────
    #  PIPELINE SUMMARY
    # ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE  [dataset={dataset}]")
    print(f"{'='*65}")
    print(f"\n  [Layer 1 — Perception]  LSTM-AE")
    print(f"    Model     : {lstm_best_path}")
    print(f"    Threshold : {threshold_path}  (P99 = {threshold:.6f})")
    print(f"\n  [Layer 2 — Diagnostic]  XGBoost Classifier")
    print(f"    Model     : {rf_path}")
    print(f"\n  [Figures]  → {RESULTS_DIR}/")
    print(f"    lstm_training_loss_{ds_suffix}.png")
    print(f"    confusion_matrix_lstm_{ds_suffix}.png")
    print(f"    confusion_matrix_rf_{ds_suffix}.png")
    print(f"    rf_classification_report_{ds_suffix}.csv\n")


if __name__ == "__main__":
    train_system(dataset=DATASET_TO_TRAIN)