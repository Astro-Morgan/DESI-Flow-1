# DESI-Flow

Reference implementation of a multi-stage machine-learning pipeline for
spectroscopic redshift estimation on DESI/SDSS spectra. The pipeline distills
a large multimodal foundation model (AION-1) into progressively lighter,
deployable models, ending in full redshift posteriors with out-of-distribution
detection -- and a photometry-only branch for objects without spectra.

This repository is published as a code reference, not an installable package.
Paths in the training scripts and notebook are placeholders (marked
`<-- UPDATE`) that must be pointed at your own data.

## The pipeline

```
AION-1 embeddings + spectra                     photometry (ugriz + WISE)
        |                                                |
        v                                                v
   [1] Plato  (teacher) ---(distill)---> [2] Aristotle   [4] KaNoNboost
        \                                     |            (latent bridge:
         \--> 32-dim embedding space  <-------/             mags -> embedding
                        |                                   -> kNN -> z)
                        v
              [3] KaNoN (normalizing flows)
                  p(z | phi) posteriors  +  p(phi) OOD scores
```

**1. Plato** (`plato/`) -- the teacher. Attentively pools AION-1 embedding
sequences against the object's own spectrum: 12 fusion layers interleave
self-attention over the query tokens with cross-attention into a 1D-CNN
projection of the spectrum, then pool a [CLS] token through a deep residual
projection head to a 32-dim embedding. Trained with a triplet loss whose
ground truth is the normalized cross-correlation (CCF) peak between
preprocessed spectra, using on-the-fly hard-triplet mining.

**2. Aristotle** (`aristotle/`) -- the student. A spectrum-only CNN +
Transformer encoder distilled from Plato via MSE on the 32-dim embeddings.
It loads Plato's trained projection head and freezes it, so student and
teacher share one embedding space. No AION model is needed at inference.

**3. KaNoN** (`kanon/KaNoN.py`, trained in `kanon/KaNoN_train.ipynb`) --
conditional normalizing flows (zuko neural spline flows) on the embeddings:
a manifold flow p(phi) for OOD detection, and a physics flow p(z | phi)
parameterized as a residual around a k-NN anchor prediction with
redshift-binned residual scaling. Produces full redshift posteriors, not just
point estimates.

**4. KaNoNboost** (`kanon/KaNoNboost.py`) -- the photometric branch. XGBoost
models map broadband magnitudes into the same 32-dim embedding space (the
"latent bridge") via a gatekeeper GALAXY/QSO classifier and per-class stacked
proxy-redshift chains; the final redshift is a k-NN lookup in embedding space.
Supports griz / ugriz / grizW / ugrizW band configurations.

## Repository layout

```
preprocessing.py                     Shared spectrum preprocessing (see below)
plato/
    Plato.py                         Teacher model + Muon optimizer factory
    Plato_training_helpers.py        DDP utils, CCF similarity, triplet loss, dataset
    Plato_train.py                   DDP training script
aristotle/
    Aristotle.py                     Student model + Muon optimizer factory
    Aristotle_training_helpers.py    DDP utils, distillation dataset
    Aristotle_train.py               DDP distillation script
kanon/
    KaNoN.py                         Flow model (anchor, flows, losses, inference)
    KaNoN_train.ipynb                Precompute -> train -> full-catalog inference
    KaNoNboost.py                    Photometric pipeline
    KaNoNboost_train.py              Training + evaluation script with plots
LICENSE                              MIT
```

The `ConvNetTeacher` / attention / projection-head blocks are intentionally
duplicated between `plato/Plato.py` and `aristotle/Aristotle.py` so each model
file reads standalone; the projection heads must (and do) match exactly.

## Spectrum preprocessing

Every model consumes the same representation, produced by
`preprocessing.preprocess_for_model(flux, ivar, mask)`: an
inverse-variance-weighted Gaussian smoothing (11-px kernel, sigma = 11/6) of
the flux over valid pixels (`mask == 0`), standardized to zero mean / unit
variance over the valid pixels, with masked regions zeroed. `GPUPreprocessor`
in the same module is the mathematically equivalent batched PyTorch version,
used by the Aristotle training loop to whiten raw batches on the GPU.

## Data expectations

Spectra HDF5 (`merged_filtered_sample.hdf5` in the placeholders): `FLUX`,
`IVAR`, `MASK` arrays of shape (N, 7781), plus for the KaNoN/KaNoNboost
stages `desi_z`, `sdss_z`, `spectype` ('GALAXY'/'QSO'), and for KaNoNboost
`sdss_flux_{u,g,r,i,z,w1,w2}` / `sdss_ext_*` fluxes and extinctions.

Embeddings HDF5 (`embeddings.h5`): per-object embedding arrays keyed by stage
-- `raw_aion` (AION-1 sequences, Plato's queries), then the outputs written by
each trained model (the placeholders use `plato` / `plato_deep_1c` for teacher
embeddings and `aristotle` for student embeddings).

KaNoN and KaNoNboost train on a "golden sample": objects whose SDSS and DESI
redshifts agree within class-dependent tolerances (GALAXY 0.33%, QSO 1%).

## Training workflow

1. **Plato**: edit `Config` in `plato/Plato_train.py`, then run it. Written
   for SLURM + multi-GPU DDP (`srun python plato/Plato_train.py` with
   MASTER_ADDR/MASTER_PORT set by the submit script); falls back to a single
   process without SLURM/DDP env vars. Embed the catalog with the best
   checkpoint and store the embeddings in `embeddings.h5`.
2. **Aristotle**: place the Plato checkpoint as `plato.pth` anywhere under
   the working directory (it is auto-discovered), edit `Config` in
   `aristotle/Aristotle_train.py`, and run the same way. Then embed the
   catalog with Aristotle.
3. **KaNoN**: run `kanon/KaNoN_train.ipynb` top to bottom -- it precomputes
   the FAISS local-scale map, local redshift gradients, and cross-validated
   k-NN anchors, trains both flows with independent early stopping and LR
   annealing, and writes a full-catalog posterior HDF5.
4. **KaNoNboost**: run `kanon/KaNoNboost_train.py` to train all four band
   configurations against the spectral embeddings and produce evaluation
   plots (`kanonboost_*.png`).

## Dependencies

Python 3.10+, PyTorch (CUDA for training), and: `numpy`, `scipy`, `h5py`,
`zuko`, `faiss` (faiss-gpu or faiss-cpu), `xgboost`, `scikit-learn`,
`joblib`, `matplotlib`, plus `livelossplot` and `tqdm` for the notebook.
Plato and Aristotle use the Muon optimizer
(`pip install git+https://github.com/KellerJordan/Muon`).

## License

MIT -- see [LICENSE](LICENSE).
