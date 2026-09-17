"""DirectBOOST: the direct photometry -> redshift baseline.

Mirrors kanonBOOST's routing (a gatekeeper GALAXY/QSO classifier followed by
per-class XGBoost experts) but skips the latent space entirely: each expert
regresses redshift directly from magnitudes and adjacent colors, with no
32-dim embedding target and no k-NN manifold lookup. This is the baseline
kanonBOOST is compared against.

Supports the ugrizW photometric configuration only (u, g, r, i, z, W1, W2).

Class and module names are held fixed (``DirectBOOST`` / ``DirectPipeline``
in ``DirectBOOST.py``) for the same reason as ``kanonBOOST.py``: cached
``.pkl`` files depend on being able to resolve these names at unpickling
time.
"""

import numpy as np
import joblib
import xgboost as xgb
from xgboost import XGBClassifier

# --- GALAXY / QSO DIRECT REGRESSORS ---
CONFIG_GALAXY_PROXY = {
    'n_estimators': 801,
    'max_depth': 10,
    'learning_rate': 0.0593,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'missing': np.nan,
    'tree_method': 'hist',
    'device': 'cuda',
}

CONFIG_QSO_PROXY = {
    'n_estimators': 148,
    'max_depth': 10,
    'learning_rate': 0.0766,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'missing': np.nan,
    'tree_method': 'hist',
    'device': 'cuda',
}

# --- GATEKEEPER CLASSIFIER ---
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
    'device': 'cuda',
    'n_jobs': -1,
    'missing': np.nan,
}


# ==========================================
#   PIPELINE ARCHITECTURE
# ==========================================

class DirectPipeline:
    """A pure photometry -> redshift direct regression expert (one class)."""

    def __init__(self, obj_type, bands, params):
        self.obj_type = obj_type
        self.bands = bands
        self.model = xgb.XGBRegressor(**params)
        self.is_fitted = False

    def _extract_features(self, X_mags):
        """Magnitudes -> [magnitudes, adjacent colors], NaN for non-finite."""
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
        if not self.is_fitted:
            raise RuntimeError("Pipeline not trained.")
        X_feats = self._extract_features(X_mags)
        return self.model.predict(X_feats)


class DirectBOOST:
    """Master direct-regression pipeline: magnitudes -> class -> z.

    Holds one gatekeeper classifier plus one GALAXY and one QSO
    ``DirectPipeline`` for the ugrizW configuration. ``predict`` expects
    exactly 7 columns (u, g, r, i, z, W1, W2); any other width raises
    ``ValueError``.
    """

    def __init__(self):
        self.registry = {}
        self.configs = {
            7: ('ugrizW', ['u', 'g', 'r', 'i', 'z', 'w1', 'w2']),
        }

        for n_cols, (name, bands) in self.configs.items():
            self.registry[name] = {
                'bands': bands,
                'classifier': XGBClassifier(**CONFIG_CLF),
                'GALAXY': DirectPipeline('GALAXY', bands, CONFIG_GALAXY_PROXY),
                'QSO': DirectPipeline('QSO', bands, CONFIG_QSO_PROXY),
            }

    def train_all(self, X_full_7band, z_true, labels):
        """Train the ugrizW configuration from the full 7-band training matrix.

        Args:
            X_full_7band: (N, 7) magnitudes [u, g, r, i, z, w1, w2].
            z_true: (N,) spectroscopic redshifts.
            labels: (N,) string labels, 'GALAXY' or 'QSO'.
        """
        X_full = np.array(X_full_7band)
        labels = np.array(labels)
        y_encoded = (labels == 'QSO').astype(int)

        idx_map = {
            'ugrizW': [0, 1, 2, 3, 4, 5, 6],
        }

        print(f"{'=' * 60}\n  DIRECTBOOST BASELINE TRAINING\n{'=' * 60}")

        for name, config_dict in self.registry.items():
            indices = idx_map[name]
            bands = config_dict['bands']
            print(f"\n>>> Configuration: {name}")

            X_slice = X_full[:, indices]

            print("   [Gatekeeper] Training Classifier...")
            X_clf = self._get_colors_for_clf(X_slice, bands)
            config_dict['classifier'].fit(X_clf, y_encoded)

            mask_gal = (labels == 'GALAXY')
            print(f"   [Expert: GAL] Training Direct Z on {mask_gal.sum()} objects...")
            config_dict['GALAXY'].train(X_slice[mask_gal], z_true[mask_gal])

            mask_qso = (labels == 'QSO')
            print(f"   [Expert: QSO] Training Direct Z on {mask_qso.sum()} objects...")
            config_dict['QSO'].train(X_slice[mask_qso], z_true[mask_qso])

    def predict(self, X_input, spectype=None):
        """Magnitudes -> redshift.

        Args:
            X_input: (N, 7) ugrizW magnitudes.
            spectype: 'GALAXY' or 'QSO' to force a specific expert (bypassing
                the gatekeeper), or None for automatic routing.
        """
        X_input = np.array(X_input)
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)

        n_cols = X_input.shape[1]
        if n_cols not in self.configs:
            raise ValueError(f"Input has {n_cols} columns. Expected 7 (ugrizW).")

        config_name = self.configs[n_cols][0]
        config_dict = self.registry[config_name]
        n_objs = len(X_input)

        # --- Routing ---
        if spectype:
            preds_type = np.full(n_objs, spectype)
        else:
            X_clf = self._get_colors_for_clf(X_input, config_dict['bands'])
            preds_encoded = config_dict['classifier'].predict(X_clf)
            preds_type = np.where(preds_encoded == 1, 'QSO', 'GALAXY')

        # --- Direct z prediction per class ---
        z_final = np.zeros(n_objs)
        mask_gal = (preds_type == 'GALAXY')
        mask_qso = (preds_type == 'QSO')

        if mask_gal.sum() > 0:
            z_final[mask_gal] = config_dict['GALAXY'].predict_z(X_input[mask_gal])
        if mask_qso.sum() > 0:
            z_final[mask_qso] = config_dict['QSO'].predict_z(X_input[mask_qso])

        return z_final

    def _get_colors_for_clf(self, X_mags, bands):
        """Classifier features: [magnitudes, adjacent colors]."""
        feats = [X_mags[:, i] for i in range(X_mags.shape[1])]
        for i in range(len(bands) - 1):
            feats.append(X_mags[:, i] - X_mags[:, i + 1])
        return np.column_stack(feats)

    def save(self, filepath):
        joblib.dump(self, filepath, compress=3)
        print(f"DirectBOOST saved to {filepath}")

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)
