import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    recall_score, precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, plot_importance
import optuna
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
DATA_PATH        = 'data/cumulative_2026.02.27_03.57.44.csv'
MODEL_DIR        = 'models'
RANDOM_SEED      = 42
TEST_SIZE        = 0.2
N_TRIALS         = 200     
N_TRIALS_REFINE  = 80     
N_CV_FOLDS       = 7      
MIN_PRECISION    = 0.95    
FORCE_THRESHOLD  = None   

# ── Features หลัก (Base) ──────────────────────────────────────────────────────
BASE_FEATURES = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_model_snr', 'koi_teq', 'koi_srad',
    'koi_steff', 'koi_slogg', 'koi_insol', 'koi_kepmag'
]

# ── Features ใหม่จาก dataset ชุดใหม่ ─────────────────────────────────────────
FP_FLAGS = [
    'koi_fpflag_nt',   
    'koi_fpflag_ss',   
    'koi_fpflag_co',   
    'koi_fpflag_ec',   
]

EXTRA_PHYSICS = [
    'koi_impact',  
    'koi_ror',      
    'koi_srho',         
    'koi_sma',      
    'koi_dor',      
    'koi_incl',     
]

ERR_PAIRS = [
    ('koi_period',   'koi_period_err1',   'koi_period_err2'),
    ('koi_depth',    'koi_depth_err1',    'koi_depth_err2'),
    ('koi_prad',     'koi_prad_err1',     'koi_prad_err2'),
    ('koi_duration', 'koi_duration_err1', 'koi_duration_err2'),
    ('koi_ror',      'koi_ror_err1',      'koi_ror_err2'),
    ('koi_srho',     'koi_srho_err1',     'koi_srho_err2'),
    ('koi_impact',   'koi_impact_err1',   'koi_impact_err2'),
]

TARGET = 'koi_disposition'

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()

    # ── A. Log-scale ─────────────────────────────────────────────────────────
    log_cols = ['koi_depth', 'koi_prad', 'koi_insol', 'koi_period',
                'koi_model_snr', 'koi_duration', 'koi_srho', 'koi_dor']
    for c in log_cols:
        if c in d.columns:
            d[f'log_{c}'] = np.log1p(d[c].clip(lower=0))

    # ── B. Original Physics Features ─────────────────────────────────────────
    d['log_transit_shape'] = np.log1p(
        (d['koi_depth'] / (d['koi_duration'] ** 2 + 1e-6)).clip(lower=0))
    d['prad_srad_ratio']   = d['koi_prad']  / (d['koi_srad']   + 1e-6)
    d['log_prad_srad']     = np.log1p(d['prad_srad_ratio'].clip(lower=0))
    d['teq_steff_ratio']   = d['koi_teq']   / (d['koi_steff']  + 1e-6)
    d['log_snr_per_depth'] = np.log1p(
        (d['koi_model_snr'] / (d['koi_depth'] + 1e-6)).clip(lower=0))
    d['log_slogg']         = np.log1p(d['koi_slogg'].clip(lower=0))

    # ── C. Interaction Terms ──────────────────────────────────────────────────
    d['snr_x_depth']    = d['log_koi_model_snr'] * d['log_koi_depth']
    d['insol_x_prad']   = d['log_koi_insol']     * d['log_koi_prad']
    d['period_x_depth'] = d['log_koi_period']    * d['log_koi_depth']
    d['prad_x_snr']     = d['log_koi_prad']      * d['log_koi_model_snr']

    # ── D. ★ NEW: FP Flag Features ★ ─────────────────────────────────────────
    for f in FP_FLAGS:
        d[f] = d[f].fillna(0).astype(int)
    # รวม flag → ยิ่งมาก flag ยิ่งน่าสงสัย
    d['fpflag_sum']  = d[FP_FLAGS].sum(axis=1)
    d['fpflag_any']  = (d['fpflag_sum'] > 0).astype(int)

    # ── E. ★ NEW: Extra Physics Columns ★ ────────────────────────────────────
    for c in EXTRA_PHYSICS:
        if c in d.columns:
            d[c] = d[c].fillna(d[c].median())

    # koi_ror เป็น radius ratio โดยตรงจาก fit (แม่นกว่า prad/srad ที่คำนวณเอง)
    d['log_ror']       = np.log1p(d['koi_ror'].clip(lower=0))

    # Impact parameter: ใกล้ 0 = transit ผ่านกลางดาว (ดีกว่า)
    d['impact_sq']     = d['koi_impact'] ** 2

    # Stellar density: ดาวแน่นกว่า = ดาวฤกษ์เล็กกว่า = transit ชัดกว่า
    d['srho_x_ror']    = d['log_koi_srho'] * d['log_ror']

    # ── F. ★ NEW: Uncertainty Features (Error Bars) ★ ────────────────────────
    for val_col, err1_col, err2_col in ERR_PAIRS:
        if err1_col in d.columns and err2_col in d.columns:
            avg_err = (d[err1_col].abs() + d[err2_col].abs()) / 2.0
            val_abs = d[val_col].abs() + 1e-8
            feat_name = f'uncert_{val_col}'
            d[feat_name] = np.log1p((avg_err / val_abs).clip(lower=0))

    # ── G. ★ NEW: koi_score (Kepler pipeline's own probability) ★ ────────────
    if 'koi_score' in d.columns:
        # Fill missing ด้วย 0.5 (neutral)
        d['koi_score'] = d['koi_score'].fillna(0.5)
        d['koi_score_logit'] = np.log(
            d['koi_score'].clip(1e-4, 1 - 1e-4) /
            (1 - d['koi_score'].clip(1e-4, 1 - 1e-4))
        )

    # ── H. Bins ───────────────────────────────────────────────────────────────
    d['period_bin'] = pd.cut(
        d['koi_period'], bins=[0, 5, 15, 50, 200, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    d['size_cat'] = pd.cut(
        d['koi_prad'], bins=[0, 1.25, 2, 6, 15, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    d['snr_bucket'] = pd.cut(
        d['koi_model_snr'], bins=[0, 10, 30, 100, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # ── รวม Feature List ──────────────────────────────────────────────────────
    engineered = (
        # Log-scaled base
        [f'log_{c}' for c in log_cols if c in df.columns] +
        # Physics ratios
        ['log_transit_shape', 'log_prad_srad', 'teq_steff_ratio',
         'log_snr_per_depth', 'log_slogg'] +
        # Interaction terms
        ['snr_x_depth', 'insol_x_prad', 'period_x_depth', 'prad_x_snr'] +
        # FP Flags (★ new ★)
        FP_FLAGS + ['fpflag_sum', 'fpflag_any'] +
        # Extra physics (★ new ★)
        ['log_ror', 'impact_sq', 'srho_x_ror',
         'koi_impact', 'koi_sma', 'koi_incl'] +
        # Uncertainty features (★ new ★)
        [f'uncert_{v}' for v, e1, e2 in ERR_PAIRS if e1 in df.columns] +
        # Kepler score (★ new ★)
        (['koi_score', 'koi_score_logit'] if 'koi_score' in df.columns else []) +
        # Bins & categoricals
        ['period_bin', 'size_cat', 'snr_bucket'] +
        # Raw originals ที่ยังมีประโยชน์
        ['koi_teq', 'koi_srad', 'koi_steff', 'koi_slogg', 'koi_kepmag']
    )
    return d, engineered


# ─────────────────────────────────────────────
# 3. OPTUNA — Maximize Recall @ Precision ≥ MIN_PRECISION
# ─────────────────────────────────────────────
def build_objective(X_tr, y_tr, base_ratio, min_precision):
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        spw = trial.suggest_float('spw_mult', 0.7, 2.5) * base_ratio
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 400, 1500),
            'learning_rate':     trial.suggest_float('learning_rate', 0.003, 0.08, log=True),
            'max_depth':         trial.suggest_int('max_depth', 3, 9),
            'min_child_weight':  trial.suggest_int('min_child_weight', 1, 15),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.4, 1.0),
            'gamma':             trial.suggest_float('gamma', 0, 8),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-5, 20.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-5, 20.0, log=True),
            'scale_pos_weight':  spw,
            'random_state':      RANDOM_SEED,
            'n_jobs':            -1,
            'eval_metric':       'logloss',
        }

        fold_recalls = []
        for tr_idx, val_idx in cv.split(X_tr, y_tr):
            m = XGBClassifier(**params)
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx], verbose=False)
            prob = m.predict_proba(X_tr.iloc[val_idx])[:, 1]
            prec_arr, rec_arr, _ = precision_recall_curve(y_tr.iloc[val_idx], prob)
            mask = prec_arr[:-1] >= min_precision
            fold_recalls.append(
                rec_arr[:-1][mask].max() if mask.any() else rec_arr[:-1].max() * 0.4
            )
        return float(np.mean(fold_recalls))

    return objective


# ─────────────────────────────────────────────
# 4. THRESHOLD SELECTION
# ─────────────────────────────────────────────
def find_threshold(y_prob, y_true, min_precision, force=None):
    if force is not None:
        return force, "Forced"
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    mask = prec_arr[:-1] >= min_precision
    if mask.any():
        best_i = rec_arr[:-1][mask].argmax()
        return thr_arr[mask][best_i], f"Max Recall @ Precision≥{min_precision}"
    fscore = (2 * prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-8)
    return thr_arr[np.argmax(fscore[:-1])], "Best F1 (fallback)"


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
def plot_dashboard(model, X_test, y_test, results, save_dir, best_thresh, min_precision):
    os.makedirs(save_dir, exist_ok=True)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, y_prob)

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # (A) Learning Curve
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(results['validation_0']['logloss'], label='Train Loss', lw=1.5, color='royalblue')
    ax0.plot(results['validation_1']['logloss'], label='Val Loss',   lw=1.5, color='darkorange')
    ax0.set_title('Learning Curve (Log Loss)', fontweight='bold')
    ax0.set_xlabel('Boosting Round'); ax0.legend(fontsize=8)

    # (B) Feature Importance
    ax1 = fig.add_subplot(gs[0, 1:])
    base = model.calibrated_classifiers_[0].estimator if hasattr(model, 'calibrated_classifiers_') else model
    plot_importance(base, ax=ax1, max_num_features=16, importance_type='gain',
                    title='Feature Importance (Gain)', show_values=False)

    # (C) ROC Curve
    ax2 = fig.add_subplot(gs[1, 0])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax2.fill_between(fpr, tpr, alpha=0.15, color='royalblue')
    ax2.plot(fpr, tpr, lw=2, color='royalblue', label=f'AUC = {auc_val:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_title('ROC Curve', fontweight='bold')
    ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR'); ax2.legend()

    # (D) PR Curve
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(rec_arr, prec_arr, lw=2, color='teal')
    ax3.axhline(min_precision, color='gray', ls=':', lw=1.5,
                label=f'Min Precision = {min_precision}')
    thr_idx = min(np.argmin(np.abs(thr_arr - best_thresh)), len(rec_arr) - 2)
    ax3.axvline(rec_arr[thr_idx], color='red', ls='-', lw=2,
                label=f'Used thr = {best_thresh:.3f}')
    ax3.set_title('Precision-Recall Curve', fontweight='bold')
    ax3.set_xlabel('Recall'); ax3.set_ylabel('Precision'); ax3.legend(fontsize=8)

    # (E) Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['False Positive', 'Confirmed']).plot(
        ax=ax4, colorbar=False, cmap='Blues')
    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    ax4.set_title(f'Confusion Matrix\nPrecision={p:.3f}  Recall={r:.3f}',
                  fontweight='bold', fontsize=10)

    fig.suptitle('Kepler XGBoost — Recall-Optimized Dashboard (New Dataset)',
                 fontsize=15, fontweight='bold')
    path = os.path.join(save_dir, 'full_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Dashboard → {path}")


def plot_feature_importance_all(base_model, save_dir):
    print("[PLOT] Feature importance (3 types)...")
    importance_types = [
        ('weight', 'Weight\n(# splits used)',          'steelblue'),
        ('gain',   'Gain\n(avg. improvement/split)',   'darkorange'),
        ('cover',  'Cover\n(avg. samples/split)',      'seagreen'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax, (itype, title, color) in zip(axes, importance_types):
        scores = base_model.get_booster().get_score(importance_type=itype)
        if not scores:
            continue
        items = sorted(scores.items(), key=lambda x: x[1])[-18:]
        names, vals = zip(*items)
        bars = ax.barh(names, vals, color=color, alpha=0.85)
        ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Score')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Feature Importance — Weight / Gain / Cover', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'feature_importance_full.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Feature importance → {path}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────
def main():
    print("[INFO] Loading Kepler Dataset (new 141-col version)...")
    try:
        df = pd.read_csv(DATA_PATH, comment='#')
    except FileNotFoundError:
        print(f"[ERROR] File not found: {DATA_PATH}"); return

    print(f"[INFO] Raw shape: {df.shape}")

    # dropna เฉพาะ BASE_FEATURES (feature ใหม่จะ fill median แทน)
    df = df.dropna(subset=BASE_FEATURES + [TARGET])
    df = df[df[TARGET] != 'CANDIDATE']
    df['is_planet'] = (df[TARGET] == 'CONFIRMED').astype(int)
    print(f"[INFO] CONFIRMED: {df['is_planet'].sum()} | FALSE POSITIVE: {(df['is_planet']==0).sum()}")

    df, final_features = engineer_features(df)

    # ตรวจสอบ features ที่ยังมี null → fill 0
    X = df[final_features].astype(float).fillna(0)
    y = df['is_planet']

    print(f"[INFO] Total features used: {len(final_features)}")
    print(f"       → {final_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    ratio = float((y_train == 0).sum()) / (y_train == 1).sum()
    print(f"[INFO] Class ratio: {ratio:.2f}")
    print(f"[INFO] Objective : Maximize Recall @ Precision ≥ {MIN_PRECISION}")

    # ── Optuna Phase 1: Broad Search ────────────────────────────────────────
    print(f"\n[OPTUNA] Phase 1 — Broad search: {N_TRIALS} trials × {N_CV_FOLDS}-fold CV...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=25),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=9, reduction_factor=3)
    )
    objective_fn = build_objective(X_train, y_train, ratio, MIN_PRECISION)
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"[OPTUNA P1] Best CV Recall: {study.best_value:.4f}")

    # ── Optuna Phase 2: Refine รอบ best params ───────────────────────────────
    print(f"\n[OPTUNA] Phase 2 — Refine search: {N_TRIALS_REFINE} trials...")
    best_p = study.best_params

    def narrow_suggest(trial, key, lo, hi, is_log=False):
        center = best_p[key]
        width  = (hi - lo) * 0.15        # แคบลง 85%
        new_lo = max(lo, center - width)
        new_hi = min(hi, center + width)
        if is_log:
            return trial.suggest_float(key, new_lo, new_hi, log=False)
        return trial.suggest_float(key, new_lo, new_hi)

    def build_refine_objective(X_tr, y_tr, base_ratio, min_precision):
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        def objective(trial):
            spw = trial.suggest_float('spw_mult', max(0.7, best_p['spw_mult']-0.4),
                                       min(2.5, best_p['spw_mult']+0.4)) * base_ratio
            params = {
                'n_estimators':      trial.suggest_int('n_estimators',
                                         max(400, best_p['n_estimators']-200),
                                         min(2000, best_p['n_estimators']+200)),
                'learning_rate':     trial.suggest_float('learning_rate',
                                         max(0.001, best_p['learning_rate']*0.5),
                                         min(0.1,   best_p['learning_rate']*2.0), log=True),
                'max_depth':         trial.suggest_int('max_depth',
                                         max(3, best_p['max_depth']-1),
                                         min(10, best_p['max_depth']+1)),
                'min_child_weight':  trial.suggest_int('min_child_weight',
                                         max(1, best_p['min_child_weight']-3),
                                         min(20, best_p['min_child_weight']+3)),
                'subsample':         trial.suggest_float('subsample',
                                         max(0.5, best_p['subsample']-0.15),
                                         min(1.0, best_p['subsample']+0.15)),
                'colsample_bytree':  trial.suggest_float('colsample_bytree',
                                         max(0.4, best_p['colsample_bytree']-0.15),
                                         min(1.0, best_p['colsample_bytree']+0.15)),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel',
                                         max(0.4, best_p['colsample_bylevel']-0.15),
                                         min(1.0, best_p['colsample_bylevel']+0.15)),
                'colsample_bynode':  trial.suggest_float('colsample_bynode',
                                         max(0.4, best_p['colsample_bynode']-0.15),
                                         min(1.0, best_p['colsample_bynode']+0.15)),
                'gamma':             trial.suggest_float('gamma',
                                         max(0, best_p['gamma']-2),
                                         min(10, best_p['gamma']+2)),
                'reg_alpha':         trial.suggest_float('reg_alpha',
                                         max(1e-6, best_p['reg_alpha']*0.1),
                                         min(30.0,  best_p['reg_alpha']*10), log=True),
                'reg_lambda':        trial.suggest_float('reg_lambda',
                                         max(1e-6, best_p['reg_lambda']*0.1),
                                         min(30.0,  best_p['reg_lambda']*10), log=True),
                'scale_pos_weight': spw,
                'random_state':      RANDOM_SEED,
                'n_jobs':            -1,
                'eval_metric':       'logloss',
            }
            fold_recalls = []
            for tr_idx, val_idx in cv.split(X_tr, y_tr):
                m = XGBClassifier(**params)
                m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx], verbose=False)
                prob = m.predict_proba(X_tr.iloc[val_idx])[:, 1]
                prec_arr, rec_arr, _ = precision_recall_curve(y_tr.iloc[val_idx], prob)
                mask = prec_arr[:-1] >= min_precision
                fold_recalls.append(
                    rec_arr[:-1][mask].max() if mask.any() else rec_arr[:-1].max() * 0.4
                )
            return float(np.mean(fold_recalls))
        return objective

    refine_study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED + 1, n_startup_trials=10),
    )
    # warm-start: เอา best trial จาก phase 1 มาใส่ก่อน
    refine_study.enqueue_trial(best_p)
    refine_study.optimize(
        build_refine_objective(X_train, y_train, ratio, MIN_PRECISION),
        n_trials=N_TRIALS_REFINE, show_progress_bar=True
    )

    # เลือก study ที่ได้ผลดีกว่า
    if refine_study.best_value >= study.best_value:
        final_study = refine_study
        print(f"[OPTUNA P2] Improved! {study.best_value:.4f} → {refine_study.best_value:.4f}")
    else:
        final_study = study
        print(f"[OPTUNA P2] Phase 1 still best: {study.best_value:.4f}")

    study = final_study
    print(f"[OPTUNA] Final Best CV Recall: {study.best_value:.4f}")
    print(f"[OPTUNA] Final Best params   : {study.best_params}")

    # ── Final Model ─────────────────────────────────────────────────────────
    bp    = study.best_params.copy()
    spw   = bp.pop('spw_mult') * ratio
    n_est = bp.pop('n_estimators')

    final_model = XGBClassifier(
        **bp, n_estimators=min(n_est + 300, 2500),  # ให้ ceiling สูงขึ้น early-stop จะหยุดเอง
        scale_pos_weight=spw,
        random_state=RANDOM_SEED, n_jobs=-1,
        eval_metric='logloss', early_stopping_rounds=50,
    )
    final_model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=False)
    print(f"[INFO] Best boosting round: {final_model.best_iteration}")

    # ── Calibration ──────────────────────────────────────────────────────────
    # CalibratedClassifierCV จะ retrain model ใหม่เอง
    # → ต้อง clone แล้วเอา early_stopping_rounds ออก ไม่งั้น error เพราะไม่มี eval_set
    print("[INFO] Calibrating probabilities (Isotonic, cv=5)...")
    from sklearn.base import clone
    cal_base = clone(final_model)
    cal_base.set_params(early_stopping_rounds=None)
    calibrated = CalibratedClassifierCV(estimator=cal_base, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)

    # ── Threshold ────────────────────────────────────────────────────────────
    y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
    best_thresh, mode = find_threshold(y_prob_cal, y_test, MIN_PRECISION, FORCE_THRESHOLD)
    print(f"[THRESHOLD] {mode} → {best_thresh:.4f}")
    y_pred = (y_prob_cal >= best_thresh).astype(int)

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_dashboard(calibrated, X_test, y_test,
                   final_model.evals_result(), MODEL_DIR, best_thresh, MIN_PRECISION)
    plot_feature_importance_all(final_model, MODEL_DIR)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n╔════════════════════════════════════════════════════╗")
    print("║   Recall-Optimized Kepler Model — Final Report     ║")
    print("╚════════════════════════════════════════════════════╝")
    print(f"  Dataset         : {DATA_PATH}")
    print(f"  Features used   : {len(final_features)}")
    print(f"  Threshold Mode  : {mode}")
    print(f"  Threshold       : {best_thresh:.4f}")
    print(f"  ROC-AUC         : {roc_auc_score(y_test, y_prob_cal):.4f}")
    print(f"  Precision       : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall          : {recall_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred,
          target_names=['False Positive (0)', 'Confirmed Planet (1)']))

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump({
        'model':          calibrated,
        'base_model':     final_model,
        'features':       final_features,
        'best_threshold': best_thresh,
        'min_precision':  MIN_PRECISION,
        'optuna_params':  study.best_params,
    }, os.path.join(MODEL_DIR, 'kepler_finetuned.pkl'))
    print(f"[SUCCESS] Saved → {MODEL_DIR}/kepler_finetuned.pkl")


if __name__ == '__main__':
    main()