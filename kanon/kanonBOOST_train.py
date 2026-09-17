"""Training + evaluation script for kanonBOOST (photometry-only redshifts,
native multi-output latent bridge).

Builds SDSS+WISE magnitudes from fluxes, filters to a "golden sample" of
objects whose SDSS and DESI redshifts agree within class-dependent
tolerances, trains the ugrizW configuration, and evaluates it in FORCED mode
(regression quality in isolation) and end-to-end AUTOMATIC mode (classifier +
expert) -- strictly on the held-out 10% test split in both cases.

If MODEL_FILE already exists, training is skipped and the cached model is
loaded directly; this is the file that should reproduce the manuscript's
"native multi-output" latent-bridge numbers (FORCED 0.80%/12.26%,
AUTOMATIC 1.14%/12.36% outlier rates for GALAXY/QSO) -- update MODEL_FILE
below if your cached run used a different filename (e.g. 'kanonBOOST_fixed.pkl'
from an earlier notebook run).
"""

import os
import sys
import h5py
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

from kanonBOOST import kanonBOOST

# --- CONFIGURATION ---
DATA_FILE = '../merged_filtered_sample.hdf5'   # <-- UPDATE: catalog HDF5
EMB_FILE = '../embeddings.h5'                  # <-- UPDATE: spectral embeddings HDF5

# The cached checkpoint that should reproduce the manuscript's reported
# latent-bridge numbers. Point this at whichever file you actually have on
# disk; common names from this project's history are 'kanonBOOST.pkl' and
# 'kanonBOOST_fixed.pkl'.
MODEL_FILE = 'kanonBOOST_fixed.pkl'

OUT_DIR = "../DESI_Flow_Figs_and_Data"
os.makedirs(OUT_DIR, exist_ok=True)

# Concordance thresholds for the golden sample: |z_sdss - z_desi| / (1 + z_desi)
TOL_GAL = 0.0033
TOL_QSO = 0.0100

plt.rcParams.update({
    'font.size': 18, 'axes.titlesize': 22, 'axes.labelsize': 20,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 18,
    'figure.titlesize': 24,
})


# ==========================================
#   DATA
# ==========================================

def load_and_prep_data():
    """Load fluxes -> extinction-corrected magnitudes, apply the golden-sample filter.

    Returns (X_7band, z_desi, embeddings, spectype) restricted to objects with
    finite data and SDSS/DESI redshift concordance within the class tolerance.
    """
    print(f"Loading HDF5 data from {DATA_FILE}...")
    with h5py.File(DATA_FILE, 'r') as f:
        z_desi = f['desi_z'][:]
        z_sdss = f['sdss_z'][:] if 'sdss_z' in f else z_desi
        raw_type = f['spectype'][:]
        spec_type = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in raw_type])

        def get_mag(k_flux, k_ext):
            if k_flux not in f:
                return np.zeros_like(z_desi)
            fl = f[k_flux][:]
            ex = f[k_ext][:] if k_ext in f else np.zeros_like(fl)
            f_corr = fl * (10 ** (0.4 * ex))
            m = np.full_like(fl, np.nan)
            valid = (f_corr > 1e-9) & np.isfinite(f_corr)
            m[valid] = -2.5 * np.log10(f_corr[valid])
            return m

        mags = [get_mag(f'sdss_flux_{b}', f'sdss_ext_{b}') for b in ['u', 'g', 'r', 'i', 'z']]
        mags.append(get_mag('sdss_flux_w1', 'sdss_ext_w1'))
        mags.append(get_mag('sdss_flux_w2', 'sdss_ext_w2'))
        X_7band = np.column_stack(mags)

    print(f"Loading Embeddings from {EMB_FILE}...")
    with h5py.File(EMB_FILE, 'r') as f:
        embeddings = f['plato'][:]

    print("Applying Golden Sample Concordance Filters...")
    valid_data = np.isfinite(z_desi) & np.isfinite(z_sdss) & np.isfinite(embeddings).all(axis=1)
    dz_norm = np.abs(z_sdss - z_desi) / (1 + z_desi)

    mask_gal = (spec_type == 'GALAXY') & (dz_norm < TOL_GAL)
    mask_qso = (spec_type == 'QSO') & (dz_norm < TOL_QSO)
    golden_mask = valid_data & (mask_gal | mask_qso)

    print(f" -> Filtered Count: {golden_mask.sum()} / {len(z_desi)}")
    return X_7band[golden_mask], z_desi[golden_mask], embeddings[golden_mask], spec_type[golden_mask]


# ==========================================
#   METRICS & PLOTTING
# ==========================================

def calculate_extended_metrics(z_true, z_pred):
    """R2, MSE, sigma_NMAD, bias, MAE, and outlier rate eta (%)."""
    denom = 1 + z_true
    mask = denom > 0
    delta = np.zeros_like(z_true)
    delta[mask] = (z_pred[mask] - z_true[mask]) / denom[mask]

    r2 = r2_score(z_true, z_pred)
    mse = mean_squared_error(z_true, z_pred)
    mae = np.mean(np.abs(z_pred - z_true))
    bias = np.median(delta[mask])
    sigma_nmad = 1.4826 * np.median(np.abs(delta[mask] - bias))
    eta = np.mean(np.abs(delta[mask]) > 0.15) * 100

    return r2, mse, sigma_nmad, bias, mae, eta


def plot_standardized_redshift_panel(fig, ax, y_t, y_p, title_str):
    """Hexbin predicted-vs-true panel with a stats box (matches Fig. 9 style)."""
    r2, mse, sigma, bias, mae, eta = calculate_extended_metrics(y_t, y_p)

    MIN_BIN_COUNT = 3
    dummy_hb = ax.hexbin(y_t, y_p, gridsize=100, mincnt=1, visible=False)
    counts = dummy_hb.get_array()
    verts = dummy_hb.get_offsets()

    sparse_bin_mask = (counts > 0) & (counts < MIN_BIN_COUNT)
    sparse_verts = verts[sparse_bin_mask]
    if len(sparse_verts) > 0:
        ax.scatter(sparse_verts[:, 0], sparse_verts[:, 1], s=5, alpha=0.3, color='gray', zorder=1)

    # A slice with too few points to fill any bin at MIN_BIN_COUNT (small
    # test-set classes, e.g. QSO) leaves the dense hexbin empty, which
    # crashes fig.colorbar on an all-empty mappable -- fall back to a plain
    # scatter with no colorbar rather than erroring out.
    dense_bin_mask = counts >= MIN_BIN_COUNT
    if dense_bin_mask.any():
        hb = ax.hexbin(y_t, y_p, gridsize=100, cmap='viridis', bins='log', mincnt=MIN_BIN_COUNT, zorder=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label('Log Density')
    else:
        ax.scatter(y_t, y_p, s=5, alpha=0.5, color='gray', zorder=2)

    dmin, dmax = 0, min(max(y_t.max(), y_p.max()), 6.0)
    ax.plot([dmin, dmax], [dmin, dmax], color='black', linestyle='-', linewidth=1.5, zorder=3)
    ax.plot([dmin, dmax], [dmin, dmax], color='white', linestyle='--', linewidth=1.5, zorder=4)

    ax.set_title(title_str)
    ax.set_xlabel('True Redshift (DESI)')
    ax.set_ylabel('Predicted Redshift')
    ax.set_xlim(dmin, dmax)
    ax.set_ylim(dmin, dmax)

    stats_text = (f"$R^2$: {r2:.4f}\nBias: {bias:.4f}\n"
                  f"$\\sigma_{{NMAD}}$: {sigma:.4f}\n$\\eta_{{>0.15}}$: {eta:.2f}%")
    proxy = mlines.Line2D([], [], color='none', label=stats_text)
    ax.legend(handles=[proxy], loc='upper left', frameon=True,
              handlelength=0, handletextpad=0, borderaxespad=0.5)


def _load_kanonboost_compat(path):
    """joblib.load with aliases for kanonBOOST's renaming history.

    This class has gone through three names over the project's history:
    KanonBOOST/KanonPipeline (original) -> KaNoNboost/KaNoNPipeline (first
    refactor) -> kanonBOOST/kanonPipeline (current). A cached .pkl may have
    been saved under any of these; try a plain load first, and only register
    the legacy aliases if that fails.
    """
    try:
        return joblib.load(path)
    except (AttributeError, ModuleNotFoundError):
        pass

    import kanonBOOST as _mod
    legacy = [('KanonBOOST', 'KanonBOOST', 'KanonPipeline'),
              ('KaNoNboost', 'KaNoNboost', 'KaNoNPipeline')]
    for module_name, cls_name, pipe_name in legacy:
        alias = sys.modules.setdefault(module_name, _mod)
        if not hasattr(alias, cls_name):
            setattr(alias, cls_name, _mod.kanonBOOST)
        if not hasattr(alias, pipe_name):
            setattr(alias, pipe_name, _mod.kanonPipeline)
    return joblib.load(path)


# ==========================================
#   EXECUTION
# ==========================================

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # Load golden-sample data
    X, z, emb, labels_raw = load_and_prep_data()

    print("\nSplitting Train/Test (90/10)...")
    X_tr, X_te, z_tr, z_te, emb_tr, emb_te, lab_tr, lab_te = train_test_split(
        X, z, emb, labels_raw, test_size=0.1, random_state=42,
    )

    # "Ground truth" types from the embedding manifold, test-set only
    # fair implementation since we've shown the kNN classifications to be more accurate than pipeline
    print("Training Ground Truth 15-NN Classifier (Embeddings)...")
    knn_gold = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)
    knn_gold.fit(emb_tr, lab_tr)
    lab_gold_te = knn_gold.predict(emb_te)  # test-set embeddings only

    # Train, or load the cached model
    if os.path.exists(MODEL_FILE):
        print(f"\nFound cached model at {MODEL_FILE} -- loading (skipping training).")
        model = _load_kanonboost_compat(MODEL_FILE)
    else:
        print("\nNo cached model found. Initializing kanonBOOST Training...")
        model = kanonBOOST()
        model.train_all(X_tr, z_tr, emb_tr, lab_tr)
        model.save(MODEL_FILE)

    # Evaluation: FORCED and AUTOMATIC
    print("\n" + "=" * 115)
    print(f"{'MODE':<12} | {'TYPE':<8} | {'R2':<8} | {'MSE':<8} | {'NMAD':<8} | {'BIAS':<9} | {'MAE':<8} | {'OUTLIER%':<8}")
    print("=" * 115)

    indices = [0, 1, 2, 3, 4, 5, 6]  # ugrizW
    X_te_slice = X_te[:, indices]

    z_pred_forced_all = np.zeros_like(z_te)

    # FORCED MODE (isolates regression from classification)
    for obj_type in ['GALAXY', 'QSO']:
        mask_type = (lab_gold_te == obj_type)
        if np.sum(mask_type) == 0:
            continue

        z_t = z_te[mask_type]
        z_p = model.predict(X_te_slice[mask_type], spectype=obj_type)
        z_pred_forced_all[mask_type] = z_p

        r2, mse, nmad, bias, mae, eta = calculate_extended_metrics(z_t, z_p)
        print(f"{'FORCED':<12} | {obj_type:<8} | {r2:8.4f} | {mse:8.4f} | {nmad:8.4f} | {bias:9.4f} | {mae:8.4f} | {eta:8.2f}%")

    print("-" * 115)

    # AUTOMATIC MODE (end-to-end pipeline)
    z_pred_auto = model.predict(X_te_slice, spectype=None)

    for obj_type in ['GALAXY', 'QSO']:
        mask_type = (lab_gold_te == obj_type)
        if np.sum(mask_type) == 0:
            continue

        z_t = z_te[mask_type]
        z_p = z_pred_auto[mask_type]

        r2, mse, nmad, bias, mae, eta = calculate_extended_metrics(z_t, z_p)
        print(f"{'AUTOMATIC':<12} | {obj_type:<8} | {r2:8.4f} | {mse:8.4f} | {nmad:8.4f} | {bias:9.4f} | {mae:8.4f} | {eta:8.2f}%")

    print("=" * 115)

    # Plots (FORCED and AUTOMATIC), test set only
    gal_mask = lab_gold_te == 'GALAXY'
    qso_mask = lab_gold_te == 'QSO'

    for mode_name, z_pred_mode in [('Forced', z_pred_forced_all), ('Automatic', z_pred_auto)]:
        g_r2, _, g_nmad, _, _, _ = calculate_extended_metrics(z_te, z_pred_mode)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        plot_standardized_redshift_panel(fig, axes[0], z_te[gal_mask], z_pred_mode[gal_mask], f'GALAXY ({mode_name})')
        plot_standardized_redshift_panel(fig, axes[1], z_te[qso_mask], z_pred_mode[qso_mask], f'QSO ({mode_name})')
        fig.suptitle(f"kanonBOOST Photometric Redshift (ugrizW) - {mode_name.upper()} | "
                     f"$R^2$: {g_r2:.4f}, $\\sigma_{{NMAD}}$: {g_nmad:.4f}", y=0.9)

        out_name = os.path.join(OUT_DIR, f"kanonBOOST_{mode_name}_Photometric_Redshift.png")
        plt.tight_layout(w_pad=4.0, rect=[0, 0, 1, 0.95])
        plt.savefig(out_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Plot saved to {out_name}")
