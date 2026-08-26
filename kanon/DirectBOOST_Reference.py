import numpy as np
import h5py
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
#   CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_FILE = 'merged_filtered_sample.hdf5'
EMB_FILE = 'embeddings.h5'
OUTPUT_MODEL = 'DirectBOOST.pkl'

# Concordance Thresholds (Golden Sample)
TOL_GAL = 0.0033
TOL_QSO = 0.0100

CONFIG_GALAXY_PROXY = {
    'n_estimators': 801,
    'max_depth': 10,
    'learning_rate': 0.0593,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'missing': np.nan,
    'tree_method': 'hist'
}

CONFIG_QSO_PROXY = {
    'n_estimators': 148,
    'max_depth': 10,
    'learning_rate': 0.0766,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'missing': np.nan,
    'tree_method': 'hist'
}

CONFIG_CLF = {
    'n_estimators': 390,
    'max_depth': 8,
    'learning_rate': 0.101,
    'subsample': 0.944,
    'colsample_bytree': 0.649,
    'min_child_weight': 1,
    'gamma': 1.66,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'n_jobs': -1,
    'missing': np.nan
}

# ==========================================
#   DIRECT PIPELINE ARCHITECTURE
# ==========================================

class DirectPipeline:
    """A pure Phot -> Z direct regression expert."""
    def __init__(self, obj_type, bands, params):
        self.obj_type = obj_type
        self.bands = bands
        self.model = xgb.XGBRegressor(**params)
        self.is_fitted = False

    def _extract_features(self, X_mags):
        X_mags = np.array(X_mags, dtype=np.float32)
        feats = [X_mags[:, i] for i in range(X_mags.shape[1])]
        for i in range(len(self.bands) - 1):
            feats.append(X_mags[:, i] - X_mags[:, i + 1])
        X_out = np.column_stack(feats)
        X_out[~np.isfinite(X_out)] = np.nan
        return X_out

    def train(self, X_mags, z_true):
        X_feats = self._extract_features(X_mags)
        self.model.fit(X_feats, z_true)
        self.is_fitted = True

    def predict_z(self, X_mags):
        if not self.is_fitted: raise RuntimeError("Pipeline not trained.")
        X_feats = self._extract_features(X_mags)
        return self.model.predict(X_feats)


class DirectBOOST:
    """
    Direct Photometry -> Redshift Baseline.
    Mirrors KanonBOOST routing, but skips the latent space entirely.
    """
    def __init__(self):
        self.registry = {}
        self.configs = {
            4: ('griz', ['g', 'r', 'i', 'z']),
            5: ('ugriz', ['u', 'g', 'r', 'i', 'z']),
            6: ('grizW', ['g', 'r', 'i', 'z', 'w1', 'w2']),
            7: ('ugrizW', ['u', 'g', 'r', 'i', 'z', 'w1', 'w2'])
        }

        for n_cols, (name, bands) in self.configs.items():
            self.registry[name] = {
                'bands': bands,
                'classifier': XGBClassifier(**CONFIG_CLF),
                'GALAXY': DirectPipeline('GALAXY', bands, CONFIG_GALAXY_PROXY),
                'QSO': DirectPipeline('QSO', bands, CONFIG_QSO_PROXY)
            }

    def train_all(self, X_full_7band, z_true, labels):
        X_full = np.array(X_full_7band)
        labels = np.array(labels)
        y_encoded = (labels == 'QSO').astype(int) 

        idx_map = {
            'griz': [1, 2, 3, 4], 'ugriz': [0, 1, 2, 3, 4],
            'grizW': [1, 2, 3, 4, 5, 6], 'ugrizW': [0, 1, 2, 3, 4, 5, 6]
        }

        print(f"{'=' * 60}\n  DIRECTBOOST BASELINE TRAINING\n{'=' * 60}")

        for name, config_dict in self.registry.items():
            indices = idx_map[name]
            bands = config_dict['bands']
            print(f"\n>>> Configuration: {name}")

            X_slice = X_full[:, indices]

            # Train Gatekeeper
            print("   [Gatekeeper] Training Classifier...")
            X_clf = self._get_colors_for_clf(X_slice, bands)
            config_dict['classifier'].fit(X_clf, y_encoded)

            # Train Direct Experts
            mask_gal = (labels == 'GALAXY')
            print(f"   [Expert: GAL] Training Direct Z on {mask_gal.sum()} objects...")
            config_dict['GALAXY'].train(X_slice[mask_gal], z_true[mask_gal])

            mask_qso = (labels == 'QSO')
            print(f"   [Expert: QSO] Training Direct Z on {mask_qso.sum()} objects...")
            config_dict['QSO'].train(X_slice[mask_qso], z_true[mask_qso])

    def predict(self, X_input, spectype=None):
        X_input = np.array(X_input)
        if X_input.ndim == 1: X_input = X_input.reshape(1, -1)

        n_cols = X_input.shape[1]
        if n_cols not in self.configs:
            raise ValueError(f"Input has {n_cols} columns. Expected 4, 5, 6, or 7.")

        config_name = self.configs[n_cols][0]
        config_dict = self.registry[config_name]
        n_objs = len(X_input)

        # Routing
        if spectype:
            preds_type = np.full(n_objs, spectype)
        else:
            X_clf = self._get_colors_for_clf(X_input, config_dict['bands'])
            preds_encoded = config_dict['classifier'].predict(X_clf)
            preds_type = np.where(preds_encoded == 1, 'QSO', 'GALAXY')

        # Direct Z Prediction
        z_final = np.zeros(n_objs)
        mask_gal = (preds_type == 'GALAXY')
        mask_qso = (preds_type == 'QSO')

        if mask_gal.sum() > 0:
            z_final[mask_gal] = config_dict['GALAXY'].predict_z(X_input[mask_gal])
        if mask_qso.sum() > 0:
            z_final[mask_qso] = config_dict['QSO'].predict_z(X_input[mask_qso])

        return z_final

    def _get_colors_for_clf(self, X_mags, bands):
        feats = [X_mags[:, i] for i in range(X_mags.shape[1])]
        for i in range(len(bands) - 1):
            feats.append(X_mags[:, i] - X_mags[:, i + 1])
        return np.column_stack(feats)

    def save(self, filepath):
        joblib.dump(self, filepath, compress=3)
        print(f"DirectBOOST saved to {filepath}")


# ==========================================
#   EVALUATION & PLOTTING FUNCTIONS
# ==========================================

def calculate_metrics(z_true, z_pred):
    denom = 1 + z_true
    mask = denom > 0
    delta = np.zeros_like(z_true)
    delta[mask] = (z_pred[mask] - z_true[mask]) / denom[mask]
    
    abs_diff = np.abs(z_pred - z_true)
    mae = np.mean(abs_diff)
    
    outliers = np.abs(delta) > 0.15
    eta = np.mean(outliers) * 100
    
    mad = np.median(np.abs(delta))
    sigma_nmad = 1.4826 * mad
    
    return mae, eta, sigma_nmad

def plot_performance(z_true, z_pred, obj_type, config_name, ax, is_auto=False):
    mae, eta, nmad = calculate_metrics(z_true, z_pred)
    ax.scatter(z_true, z_pred, s=1, alpha=0.05, c='black', rasterized=True)
    
    max_z = np.max(z_true)
    ax.plot([0, max_z], [0, max_z], 'r--', lw=1.5, alpha=0.7)
    x_line = np.linspace(0, max_z, 100)
    ax.plot(x_line, x_line + 0.15 * (1 + x_line), 'r:', lw=1, alpha=0.5)
    ax.plot(x_line, x_line - 0.15 * (1 + x_line), 'r:', lw=1, alpha=0.5)
    
    mode_str = "AUTOMATIC" if is_auto else "FORCED EXPERT"
    ax.set_title(f"DIRECT: {obj_type} ({config_name})\n[{mode_str}]")
    ax.set_xlabel("Spectroscopic Z")
    ax.set_ylabel("Predicted Z")
    ax.set_xlim(0, max_z)
    ax.set_ylim(0, max_z)
    ax.set_aspect('equal')
    
    text_str = f"MAE: {mae:.4f}\n$\sigma_{{NMAD}}$: {nmad:.4f}\nOutlier $\eta$: {eta:.2f}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)


def load_and_prep_data():
    print(f"Loading HDF5 data from {DATA_FILE}...")
    with h5py.File(DATA_FILE, 'r') as f:
        z_desi = f['desi_z'][:]
        z_sdss = f['sdss_z'][:] if 'sdss_z' in f else z_desi
        raw_type = f['spectype'][:]
        spec_type = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in raw_type])

        def get_mag(k_flux, k_ext):
            if k_flux not in f: return np.zeros_like(z_desi)
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

    is_gal = (spec_type == 'GALAXY')
    mask_gal = is_gal & (dz_norm < TOL_GAL)

    is_qso = (spec_type == 'QSO')
    mask_qso = is_qso & (dz_norm < TOL_QSO)

    golden_mask = valid_data & (mask_gal | mask_qso)

    print(f" -> Filtered Count: {golden_mask.sum()} / {len(z_desi)}")
    return X_7band[golden_mask], z_desi[golden_mask], embeddings[golden_mask], spec_type[golden_mask]

# ==========================================
#   MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    X, z, emb, labels_raw = load_and_prep_data()

    print("\nSplitting Train/Test (90/10)...")
    X_tr, X_te, z_tr, z_te, emb_tr, emb_te, lab_tr, lab_te = train_test_split(
        X, z, emb, labels_raw, test_size=0.1, random_state=42
    )

    print("Training Ground Truth 15-NN Classifier (Embeddings)...")
    knn_gold = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)
    knn_gold.fit(emb_tr, lab_tr)
    # Define the ground truth strictly for the test set
    lab_gold_te = knn_gold.predict(emb_te)

    print("\nInitializing DirectBOOST Training...")
    model = DirectBOOST()
    # Train only on the training split
    model.train_all(X_tr, z_tr, lab_tr)
    model.save(OUTPUT_MODEL)

    print("\n" + "=" * 110)
    print(f"{'CONFIG':<8} | {'TYPE':<7} | {'MODE':<12} | {'ACC (Col vs Emb)':<16} | {'MAE':<8} | {'NMAD':<8} | {'OUTLIER%':<8}")
    print("=" * 110)

    config_slices = {
        'griz': [1, 2, 3, 4],
        'ugriz': [0, 1, 2, 3, 4],
        'grizW': [1, 2, 3, 4, 5, 6],
        'ugrizW': [0, 1, 2, 3, 4, 5, 6]
    }

    # PART A: Forced Modes
    for name, indices in config_slices.items():
        # Evaluate strictly on the test set slice
        X_te_slice = X_te[:, indices]

        internal_clf = model.registry[name]['classifier']
        X_clf_feats = np.nan_to_num(model._get_colors_for_clf(X_te_slice, model.registry[name]['bands']), nan=30.0)
        pred_ints = internal_clf.predict(X_clf_feats)
        pred_strings = np.where(pred_ints == 1, 'QSO', 'GALAXY')
        acc_te = np.mean(pred_strings == lab_gold_te) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, obj_type in enumerate(['GALAXY', 'QSO']):
            # Filter the test set by ground-truth type
            mask_type = (lab_gold_te == obj_type)
            if np.sum(mask_type) == 0:
                continue

            z_pred_forced = model.predict(X_te_slice[mask_type], spectype=obj_type)
            z_target = z_te[mask_type]

            mae, eta, nmad = calculate_metrics(z_target, z_pred_forced)
            print(f"{name:<8} | {obj_type:<7} | {'FORCED':<12} | {acc_te:.2f}%             | {mae:.4f}   | {nmad:.4f}   | {eta:.2f}")
            plot_performance(z_target, z_pred_forced, obj_type, name, axes[i], is_auto=False)

        plt.tight_layout()
        plot_filename = f"direct_res_{name}_forced.png"
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        print(f" -> Plot saved to {plot_filename}")

    # PART B: Automatic Mode
    print("-" * 110)
    name = 'ugrizW'
    indices = config_slices[name]
    X_te_slice = X_te[:, indices]

    z_pred_auto = model.predict(X_te_slice, spectype=None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, obj_type in enumerate(['GALAXY', 'QSO']):
        mask_type = (lab_gold_te == obj_type)
        if np.sum(mask_type) == 0:
            continue
            
        z_p = z_pred_auto[mask_type]
        z_t = z_te[mask_type]

        mae, eta, nmad = calculate_metrics(z_t, z_p)
        acc_disp = "N/A (Auto)"
        print(f"{name:<8} | {obj_type:<7} | {'AUTOMATIC':<12} | {acc_disp:<16} | {mae:.4f}   | {nmad:.4f}   | {eta:.2f}")
        plot_performance(z_t, z_p, obj_type, "ugrizW", axes[i], is_auto=True)

    plt.tight_layout()
    plot_filename = f"direct_res_{name}_AUTOMATIC.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f" -> Final Automatic Plot saved to {plot_filename}")
    print("=" * 110)
