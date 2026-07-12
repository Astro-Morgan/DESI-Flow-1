"""KaNoNboost: photometry-only redshift estimation via the latent bridge.

Where KaNoN needs a spectrum-derived embedding, KaNoNboost works from
broadband magnitudes alone. The trick is the "latent bridge": instead of
regressing z from colors directly, gradient-boosted trees map photometry into
the *same 32-dim embedding space* the spectral models produce, and the final
redshift comes from a k-NN lookup in that space (the "manifold memory" fitted
on spectral embeddings + true redshifts).

Per photometric configuration (griz / ugriz / grizW / ugrizW):
    1. A gatekeeper XGBoost classifier routes each object to GALAXY or QSO.
    2. A per-class ``KaNoNPipeline`` runs an iteratively-stacked proxy-z
       chain (each stage sees the previous stage's prediction as a feature),
       then maps [features, proxy_z] -> 32-dim embedding with a
       multi-output XGBoost regressor trained on out-of-fold proxy
       predictions (to avoid leakage).
    3. The shared k-NN manifold memory converts the predicted embedding to z.

Hyperparameters below were tuned per class; the commented scores record the
outlier rates achieved during that tuning.
"""

import numpy as np
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

# ==========================================
#   OPTIMIZED HYPERPARAMETERS
# ==========================================

# --- GALAXY PIPELINE ---
CONFIG_GALAXY = {
    'proxy': {
        'n_estimators': 801,
        'max_depth': 10,
        'learning_rate': 0.0593,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'missing': np.nan,
        'tree_method': 'hist',
    },
    # Score: 0.684% Outlier Rate
    'latent': {
        'n_estimators': 704,
        'max_depth': 15,
        'learning_rate': 0.0537,
        'min_child_weight': 7,
        'subsample': 0.727,
        'colsample_bytree': 0.85,
        'gamma': 0.936,
        'reg_alpha': 2.56,
        'grow_policy': 'lossguide',
        'tree_method': 'hist',
        'n_jobs': -1,
        'missing': np.nan,
    },
}

# --- QUASAR PIPELINE ---
CONFIG_QSO = {
    'proxy': {
        'n_estimators': 148,
        'max_depth': 10,
        'learning_rate': 0.0766,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'missing': np.nan,
        'tree_method': 'hist',
    },
    # Score: 12.07% Outlier Rate
    'latent': {
        'n_estimators': 3400,
        'max_depth': 14,
        'learning_rate': 0.0073,
        'min_child_weight': 9,
        'subsample': 0.807,
        'colsample_bytree': 0.796,
        'gamma': 0.0008,  # near-zero gamma allows jagged manifold cuts
        'reg_alpha': 2.88,
        'grow_policy': 'lossguide',
        'tree_method': 'hist',
        'n_jobs': -1,
        'missing': np.nan,
    },
}

# --- GATEKEEPER CLASSIFIER ---
# 98.19% Accuracy
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
    'missing': np.nan,
}


# ==========================================
#   PIPELINE ARCHITECTURE
# ==========================================

class KaNoNPipeline:
    """Per-class (GALAXY or QSO) photometry -> latent-embedding pipeline.

    Two stages:
        proxy_chain: ``iterations`` stacked XGBoost regressors predicting a
            proxy redshift; stage i > 0 receives [features, z_{i-1}].
        latent_mapper: multi-output XGBoost mapping
            [features, proxy_z] -> 32-dim embedding. Trained on
            *out-of-fold* proxy predictions so it never sees a proxy value
            the chain produced for its own training row.
    """

    def __init__(self, obj_type, bands, proxy_params, latent_params):
        self.obj_type = obj_type
        self.bands = bands
        self.proxy_params = proxy_params
        self.latent_params = latent_params

        self.proxy_chain = []
        self.latent_mapper = None
        self.is_fitted = False

    def _extract_features(self, X_mags):
        """Magnitudes -> [magnitudes, adjacent colors], NaN for non-finite."""
        X_mags = np.array(X_mags, dtype=np.float32)
        feats = [X_mags[:, i] for i in range(X_mags.shape[1])]
        for i in range(len(self.bands) - 1):
            feats.append(X_mags[:, i] - X_mags[:, i + 1])
        X_out = np.column_stack(feats)
        X_out[~np.isfinite(X_out)] = np.nan  # XGBoost handles NaN natively
        return X_out

    def train(self, X_mags, z_true, embeddings, iterations=5, n_folds=5):
        """Fit the proxy chain and latent mapper.

        Args:
            X_mags: (N, n_bands) magnitudes for this object class.
            z_true: (N,) spectroscopic redshifts.
            embeddings: (N, 32) target spectral embeddings.
            iterations: number of stacking stages in the proxy chain.
            n_folds: folds for the out-of-fold proxy predictions.
        """
        X_feats = self._extract_features(X_mags)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        z_proxy_oof = np.zeros(len(X_feats))

        # --- Full chain (used at inference) ---
        z_curr_full = np.zeros(len(X_feats))
        self.proxy_chain = []
        for itr in range(iterations):
            X_in = X_feats if itr == 0 else np.column_stack([X_feats, z_curr_full])
            m = xgb.XGBRegressor(**self.proxy_params)
            m.fit(X_in, z_true)
            self.proxy_chain.append(m)
            z_curr_full = m.predict(X_in)

        # --- Out-of-fold chain (only its val-fold predictions are kept,
        #     and only as *inputs* to the latent mapper) ---
        for tr_idx, val_idx in kf.split(X_feats):
            z_curr_fold_tr = np.zeros(len(tr_idx))
            z_curr_fold_val = np.zeros(len(val_idx))
            for itr in range(iterations):
                X_tr = X_feats[tr_idx] if itr == 0 else np.column_stack([X_feats[tr_idx], z_curr_fold_tr])
                X_val = X_feats[val_idx] if itr == 0 else np.column_stack([X_feats[val_idx], z_curr_fold_val])
                m = xgb.XGBRegressor(**self.proxy_params)
                m.fit(X_tr, z_true[tr_idx])
                z_curr_fold_tr = m.predict(X_tr)
                z_curr_fold_val = m.predict(X_val)
            z_proxy_oof[val_idx] = z_curr_fold_val

        # --- Latent mapper: [features, oof proxy z] -> embedding ---
        X_final_train = np.column_stack([X_feats, z_proxy_oof])
        self.latent_mapper = MultiOutputRegressor(xgb.XGBRegressor(**self.latent_params))
        self.latent_mapper.fit(X_final_train, embeddings)
        self.is_fitted = True

    def predict_latent(self, X_mags):
        """Magnitudes -> predicted 32-dim embedding."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not trained.")
        X_feats = self._extract_features(X_mags)

        # Replay the proxy chain: it has exactly `iterations` models, so this
        # loop refines z_curr the same number of times as during training.
        z_curr = np.zeros(len(X_feats))
        for m in self.proxy_chain:
            X_in = X_feats if np.all(z_curr == 0) else np.column_stack([X_feats, z_curr])
            z_curr = m.predict(X_in)

        X_final = np.column_stack([X_feats, z_curr])
        return self.latent_mapper.predict(X_final)


class KaNoNboost:
    """Master photometric pipeline: magnitudes -> class -> embedding -> z.

    Holds one gatekeeper classifier plus one GALAXY and one QSO
    ``KaNoNPipeline`` per band configuration, and a single shared k-NN
    "manifold memory" (embeddings -> z) fitted on the spectral training set.
    The band configuration is inferred at predict time from the number of
    input columns: 4=griz, 5=ugriz, 6=grizW, 7=ugrizW.
    """

    def __init__(self):
        self.registry = {}
        self.manifold_memory = KNeighborsRegressor(n_neighbors=10, weights='distance',
                                                   n_jobs=-1, algorithm='kd_tree')

        self.configs = {
            4: ('griz', ['g', 'r', 'i', 'z']),
            5: ('ugriz', ['u', 'g', 'r', 'i', 'z']),
            6: ('grizW', ['g', 'r', 'i', 'z', 'w1', 'w2']),
            7: ('ugrizW', ['u', 'g', 'r', 'i', 'z', 'w1', 'w2']),
        }

        for n_cols, (name, bands) in self.configs.items():
            self.registry[name] = {
                'bands': bands,
                'classifier': XGBClassifier(**CONFIG_CLF),
                'GALAXY': KaNoNPipeline('GALAXY', bands, CONFIG_GALAXY['proxy'], CONFIG_GALAXY['latent']),
                'QSO': KaNoNPipeline('QSO', bands, CONFIG_QSO['proxy'], CONFIG_QSO['latent']),
            }

    def train_all(self, X_full_7band, z_true, embeddings, labels):
        """Train every band configuration from the full 7-band training matrix.

        Args:
            X_full_7band: (N, 7) magnitudes [u, g, r, i, z, w1, w2].
            z_true: (N,) spectroscopic redshifts.
            embeddings: (N, 32) spectral embeddings (the latent targets).
            labels: (N,) string labels, 'GALAXY' or 'QSO'.
        """
        X_full = np.array(X_full_7band)
        labels = np.array(labels)
        y_encoded = (labels == 'QSO').astype(int)  # 0=GAL, 1=QSO

        print("Training Global Manifold Lookup (Singleton)...")
        self.manifold_memory.fit(embeddings, z_true)

        idx_map = {
            'griz': [1, 2, 3, 4], 'ugriz': [0, 1, 2, 3, 4],
            'grizW': [1, 2, 3, 4, 5, 6], 'ugrizW': [0, 1, 2, 3, 4, 5, 6],
        }

        print(f"{'=' * 60}\n  KANONBOOST PRODUCTION TRAINING (Optimized)\n{'=' * 60}")

        for name, config_dict in self.registry.items():
            indices = idx_map[name]
            bands = config_dict['bands']
            print(f"\n>>> Configuration: {name}")

            X_slice = X_full[:, indices]

            print("   [Gatekeeper] Training Classifier...")
            X_clf = self._get_colors_for_clf(X_slice, bands)
            config_dict['classifier'].fit(X_clf, y_encoded)

            mask_gal = (labels == 'GALAXY')
            print(f"   [Expert: GAL] Training on {mask_gal.sum()} objects...")
            config_dict['GALAXY'].train(X_slice[mask_gal], z_true[mask_gal], embeddings[mask_gal])

            mask_qso = (labels == 'QSO')
            print(f"   [Expert: QSO] Training on {mask_qso.sum()} objects...")
            config_dict['QSO'].train(X_slice[mask_qso], z_true[mask_qso], embeddings[mask_qso])

    def predict(self, X_input, spectype=None):
        """Magnitudes -> redshift.

        Args:
            X_input: (N, 4|5|6|7) magnitudes; the column count selects the
                band configuration.
            spectype: 'GALAXY' or 'QSO' to force a specific expert (bypassing
                the gatekeeper), or None for automatic routing.
        """
        X_input = np.array(X_input)
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)

        n_cols = X_input.shape[1]
        if n_cols not in self.configs:
            raise ValueError(f"Input has {n_cols} columns. Expected 4, 5, 6, or 7.")

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

        # --- Latent projection per class ---
        latent_vectors = np.zeros((n_objs, 32))
        mask_gal = (preds_type == 'GALAXY')
        mask_qso = (preds_type == 'QSO')

        if mask_gal.sum() > 0:
            latent_vectors[mask_gal] = config_dict['GALAXY'].predict_latent(X_input[mask_gal])
        if mask_qso.sum() > 0:
            latent_vectors[mask_qso] = config_dict['QSO'].predict_latent(X_input[mask_qso])

        # --- Final z via k-NN in embedding space ---
        return self.manifold_memory.predict(latent_vectors)

    def _get_colors_for_clf(self, X_mags, bands):
        """Classifier features: [magnitudes, adjacent colors]."""
        feats = [X_mags[:, i] for i in range(X_mags.shape[1])]
        for i in range(len(bands) - 1):
            feats.append(X_mags[:, i] - X_mags[:, i + 1])
        return np.column_stack(feats)

    def save(self, filepath):
        joblib.dump(self, filepath, compress=3)
        print(f"KaNoNboost saved to {filepath}")

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)
