"""Training + evaluation script for KaNoNboost (photometry-only redshifts).

Builds SDSS+WISE magnitudes from fluxes, filters to a "golden sample" of
objects whose SDSS and DESI redshifts agree within class-dependent
tolerances, trains all four band configurations, and evaluates each expert
both in forced mode (regression quality in isolation) and end-to-end
automatic mode (classifier + expert), saving scatter plots per configuration.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from KaNoNboost import KaNoNboost

# --- CONFIGURATION ---
DATA_FILE = 'merged_filtered_sample.hdf5'   # <-- UPDATE: catalog HDF5
EMB_FILE = 'embeddings.h5'                  # <-- UPDATE: spectral embeddings HDF5
OUTPUT_MODEL = 'KaNoNboost.pkl'

# Concordance thresholds for the golden sample: |z_sdss - z_desi| / (1 + z_desi)
TOL_GAL = 0.0033
TOL_QSO = 0.0100


def calculate_metrics(z_true, z_pred):
    """MAE, outlier rate eta (|dz|/(1+z) > 0.15, in %), and sigma_NMAD."""
    denom = 1 + z_true
    mask = denom > 0

    delta = np.zeros_like(z_true)
    delta[mask] = (z_pred[mask] - z_true[mask]) / denom[mask]

    mae = np.mean(np.abs(z_pred - z_true))
    eta = np.mean(np.abs(delta) > 0.15) * 100
    sigma_nmad = 1.4826 * np.median(np.abs(delta))

    return mae, eta, sigma_nmad


def plot_performance(z_true, z_pred, obj_type, config_name, ax, is_auto=False):
    """Predicted-vs-true scatter with the 0.15(1+z) outlier envelope + metrics box."""
    mae, eta, nmad = calculate_metrics(z_true, z_pred)

    ax.scatter(z_true, z_pred, s=1, alpha=0.05, c='black', rasterized=True)

    # Identity line and outlier envelope.
    max_z = np.max(z_true)
    ax.plot([0, max_z], [0, max_z], 'r--', lw=1.5, alpha=0.7)
    x_line = np.linspace(0, max_z, 100)
    ax.plot(x_line, x_line + 0.15 * (1 + x_line), 'r:', lw=1, alpha=0.5)
    ax.plot(x_line, x_line - 0.15 * (1 + x_line), 'r:', lw=1, alpha=0.5)

    mode_str = "AUTOMATIC" if is_auto else "FORCED EXPERT"
    ax.set_title(f"{obj_type} ({config_name})\n[{mode_str}]")
    ax.set_xlabel("Spectroscopic Z")
    ax.set_ylabel("Predicted Z")
    ax.set_xlim(0, max_z)
    ax.set_ylim(0, max_z)
    ax.set_aspect('equal')

    text_str = (
        f"MAE: {mae:.4f}\n"
        rf"$\sigma_{{NMAD}}$: {nmad:.4f}" "\n"
        rf"Outlier $\eta$: {eta:.2f}%"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)


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
            """Flux (+extinction) -> magnitude; NaN where the flux is unusable."""
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


if __name__ == "__main__":
    # 1. --- Load golden-sample data ---
    X, z, emb, labels_raw = load_and_prep_data()

    # 2. --- Train/test split ---
    print("\nSplitting Train/Test (90/10)...")
    X_tr, X_te, z_tr, z_te, emb_tr, emb_te, lab_tr, lab_te = train_test_split(
        X, z, emb, labels_raw, test_size=0.1, random_state=42,
    )

    # 3. --- "Ground truth" types from the embedding manifold ---
    # A 15-NN classifier on the spectral embeddings defines the reference
    # class labels for evaluation (manifold class, not raw pipeline labels).
    print("Training Ground Truth 15-NN Classifier (Embeddings)...")
    knn_gold = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)
    knn_gold.fit(emb_tr, lab_tr)
    lab_gold_full = knn_gold.predict(emb)

    # 4. --- Train KaNoNboost ---
    print("\nInitializing KaNoNboost Training...")
    model = KaNoNboost()
    model.train_all(X_tr, z_tr, emb_tr, lab_tr)
    model.save(OUTPUT_MODEL)

    # 5. --- Evaluation ---
    print("\n" + "=" * 110)
    print(f"{'CONFIG':<8} | {'TYPE':<7} | {'MODE':<12} | {'ACC (Col vs Emb)':<16} | "
          f"{'MAE':<8} | {'NMAD':<8} | {'OUTLIER%':<8}")
    print("=" * 110)

    config_slices = {
        'griz': [1, 2, 3, 4],
        'ugriz': [0, 1, 2, 3, 4],
        'grizW': [1, 2, 3, 4, 5, 6],
        'ugrizW': [0, 1, 2, 3, 4, 5, 6],
    }

    # --- Part A: forced-expert metrics (isolates the latent bridge) ---
    for name, indices in config_slices.items():
        X_full_slice = X[:, indices]

        # Classifier accuracy on the held-out test set (for reporting).
        X_te_slice = X_te[:, indices]
        lab_gold_te = lab_gold_full[len(X_tr):]

        internal_clf = model.registry[name]['classifier']
        X_clf_feats = np.nan_to_num(
            model._get_colors_for_clf(X_te_slice, model.registry[name]['bands']), nan=30.0)
        pred_ints = internal_clf.predict(X_clf_feats)
        pred_strings = np.where(pred_ints == 1, 'QSO', 'GALAXY')
        acc_te = np.mean(pred_strings == lab_gold_te) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, obj_type in enumerate(['GALAXY', 'QSO']):
            # Force the specific expert to isolate regression performance.
            mask_type = (lab_gold_full == obj_type)
            z_pred_forced = model.predict(X_full_slice[mask_type], spectype=obj_type)
            z_target = z[mask_type]

            mae, eta, nmad = calculate_metrics(z_target, z_pred_forced)
            print(f"{name:<8} | {obj_type:<7} | {'FORCED':<12} | {acc_te:.2f}%           | "
                  f"{mae:.4f}   | {nmad:.4f}   | {eta:.2f}")

            plot_performance(z_target, z_pred_forced, obj_type, name, axes[i], is_auto=False)

        plt.tight_layout()
        plot_filename = f"kanonboost_{name}_forced.png"
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        print(f" -> Plot saved to {plot_filename}")

    # --- Part B: end-to-end automatic mode on the flagship config (ugrizW) ---
    print("-" * 110)
    name = 'ugrizW'
    indices = config_slices[name]
    X_full_slice = X[:, indices]

    z_pred_auto = model.predict(X_full_slice, spectype=None)  # classifier routes

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, obj_type in enumerate(['GALAXY', 'QSO']):
        # Filter by manifold-truth type to see how well true populations recover
        # (metrics now include the cost of classification errors).
        mask_type = (lab_gold_full == obj_type)
        z_p = z_pred_auto[mask_type]
        z_t = z[mask_type]

        mae, eta, nmad = calculate_metrics(z_t, z_p)
        acc_disp = "N/A (Auto)"
        print(f"{name:<8} | {obj_type:<7} | {'AUTOMATIC':<12} | {acc_disp:<16} | "
              f"{mae:.4f}   | {nmad:.4f}   | {eta:.2f}")

        plot_performance(z_t, z_p, obj_type, "ugrizW", axes[i], is_auto=True)

    plt.tight_layout()
    plot_filename = f"kanonboost_{name}_AUTOMATIC.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f" -> Final Automatic Plot saved to {plot_filename}")

    print("=" * 110)
