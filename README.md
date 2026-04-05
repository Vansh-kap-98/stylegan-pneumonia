# Synthetic Chest X-ray Image Generation (256x256) Using Conditional Generative Adversarial Networks

This repository implements a class-conditional StyleGAN2-ADA model for synthesizing high-fidelity chest X-ray images at 256x256 resolution, addressing scarcity and privacy concerns in medical imaging datasets.

## 1. Project Overview

This research extends prior cGAN work on chest X-ray synthesis by leveraging **StyleGAN2-ADA**, a state-of-the-art architecture with adaptive discriminator augmentation (ADA). 

**Key Contributions:**
- Improved resolution from 64×64 to 256×256 for enhanced diagnostic detail.
- Class-conditional generation: Normal (0) vs Pneumonia (1).
- Adaptive augmentation strategy (ADA target = 0.6) tailored for limited medical datasets.
- Windows-compatible implementation with fallback ops for broader reproducibility.

**Previous Work Context:**
Prior 64×64 cGAN studies achieved 78% visual Turing test indistinguishability. This 256×256 implementation provides a stronger baseline for downstream clinical validation.

## 2. Technical Implementation

### Core Architecture
- **Model**: StyleGAN2-ADA (PyTorch)
- **Resolution**: 256×256 pixels
- **Conditioning**: Class labels (0 = Normal, 1 = Pneumonia)
- **Augmentation**: Adaptive Discriminator Augmentation (ADA) with target 0.6
- **Optimizer**: Adam (NVIDIA defaults)
- **Loss**: Logistic + R1 regularization

### Environment & Compatibility

**Hardware Requirements:**
- Single GPU with minimum 12 GB VRAM (tested on RTX 86xx series)
- Python 3.8+
- PyTorch 1.8.1 + CUDA 11.1

**Windows Custom-Op Fallback:**
Due to CUDA compilation constraints on this platform, training uses fallback ops:

```powershell
$env:STYLEGAN_DISABLE_CUSTOM_OPS="1"
```

**Local Patches Applied:**
- `third_party/stylegan2-ada-pytorch/torch_utils/ops/bias_act.py`: Fallback to PyTorch reference implementation
- `third_party/stylegan2-ada-pytorch/torch_utils/ops/upfirdn2d.py`: Fixed infinite retry bug + fallback ops

These patches enable stable training without custom CUDA plugin builds, trading ~3× throughput for platform compatibility.

## Current Project Status (April 4, 2026)

- Historical baseline run to **kimg 400** completed (run 00014).
- Updated retraining run 00017 (gamma 8) reached **kimg 300** with:
  - FID (snapshot 300): **51.7074**
  - KID (snapshot 300): **0.06121**
- Latest tuning run 00018 (gamma 6, ADA target 0.65) produced the best checkpoint so far at **kimg 240**:
  - Checkpoint: `training-runs/00018-pneumonia_256_conditional-cond-auto1-gamma6-kimg300-batch4-ada-target0.65/network-snapshot-000240.pkl`
  - FID: **25.3038**
  - KID: **0.02246**

Final selected research checkpoint for downstream comparison is run 00018 snapshot 240.

## 3. Training Configuration & Progress

### Training Hyperparameters

Main flags used for kimg 100 run:

- `--cfg=auto`: Auto-select config based on resolution
- `--cond=1`: Enable class-conditional generation
- `--mirror=0`: Disable horizontal flips (clinically inappropriate for X-rays)
- `--aug=ada --target=0.6`: Adaptive augmentation targeting 60% augmentation probability
- `--gamma=2.0`: R1 regularization gamma
- `--kimg=100`: Total training images in thousands
- `--snap=5`: Save snapshots every 5 ticks
- `--workers=2`: DataLoader workers
- `--metrics=none`: Disable FID during training for speed
- `--batch=4`: Batch size per GPU

### Observed Performance

- **Steady-state throughput**: ~200 sec/kimg with fallback ops (vs. ~50-100 sec/kimg with compiled ops on fast hardware)
- **GPU memory**: 2.5–2.7 GB sustained
- **Total training time to kimg 100**: ~5.8 hours
- **Architecture stability**: No NaN/inf, no mode collapse detected

### Run History

| Run ID | Target kimg | Status | Notes |
|--------|-------------|--------|-------|
| 00007 | 2 | Completed | Smoke test; all pipelines validated |
| 00008 | 50 | Completed | Resume from 00007; quality improving |
| 00009 | 100 | Interrupted | Aborted at tick 11 (kimg 44) by user |
| 00010 | 100 | Aborted | Test resume; created new run 00011 instead |
| 00011 | 100 | **Completed** | First production milestone |
| 00013 | 200 | **Completed** | Best checkpoint identified here |
| 00014 | 400 | **Completed** | Used for post-200 degradation analysis |
| 00017 | 300 | **Completed** | Gamma 8 retuning; FID 51.7, KID 0.061 at snapshot 300 |
| 00018 | 300 | **Completed** | Gamma 6 + ADA 0.65; final selected snapshot 240 (FID 25.3, KID 0.022) |

### Dataset and Labels

- **Dataset source**: Mendeley chest X-ray repository
- **Dataset zip**: `datasets/pneumonia_256_conditional.zip`
- **Resolution**: 256×256 pixels
- **Total samples**: 5,232 images
- **Class distribution**: 0 = Normal, 1 = Pneumonia
- **Preprocessing**: Lanczos resampling, no horizontal flip

## 4. Evaluation Outcomes

### Metrics Used and Extended Metrics Set

This project currently uses:

- **FID (`fid50k_full`)**: Primary distribution-level quality metric between generated and real images (lower is better).
- **KID (`kid50k_full`)**: Kernel-based distance alternative to FID, often more stable with smaller datasets (lower is better).

For publication-strength evaluation, include the following additional metrics:

- **Precision / Recall for GANs**: Separates fidelity (precision) from diversity (recall).
- **Density / Coverage**: Complements precision/recall for manifold overlap analysis.
- **Class-conditional consistency**: External pneumonia classifier agreement with conditioning label.
- **Diversity score (LPIPS-based intra-class diversity)**: Detects mode collapse within each class.
- **Nearest-neighbor leakage check**: Confirms generated images are not near-duplicates of training images.
- **Downstream utility delta**: Change in classifier performance when training with real-only vs real+synthetic data.

Recommended reporting convention:

- Report mean and standard deviation across at least 3 runs/seeds where possible.
- Keep sample counts and class balance fixed across checkpoints/models for fair comparison.
- Treat FID/KID as necessary but not sufficient; pair them with diversity and leakage analyses.

### First Quantitative Results (Snapshot 000100)

**Test date**: March 30, 2026

Checkpoint evaluated:
- `training-runs/00011-pneumonia_256_conditional-cond-auto1-gamma2-kimg100-batch4-ada-target0.6-resumecustom/network-snapshot-000100.pkl`

Metrics computation (required custom-op fallback):
```powershell
$env:STYLEGAN_DISABLE_CUSTOM_OPS="1"
python calc_metrics.py --metrics=fid50k_full,kid50k_full --data=datasets/pneumonia_256_conditional.zip --network=[snapshot path]
```

**Observed Results:**
- **FID (Frechet Inception Distance)**: 133.3156
- **KID (Kernel Inception Distance)**: 0.1831
- **Evaluation time**: ~24 minutes (GPU, single pass)

**Preliminary Interpretation:**
- These values indicate a valid but weak baseline; the generated distribution shows a significant gap to real X-ray distribution.
- FID/KID alone do not indicate overfitting—further analysis (leakage checks, visual review, downstream task utility) is required.
- Next step: Direct comparison against snapshots 000080 and 000060 using identical evaluation protocol.

### Latest Quantitative Results (Run 00018, Snapshot 000240)

**Test date**: April 4, 2026

Checkpoint evaluated:
- `training-runs/00018-pneumonia_256_conditional-cond-auto1-gamma6-kimg300-batch4-ada-target0.65/network-snapshot-000240.pkl`

**Observed Results:**
- **FID (Frechet Inception Distance)**: 25.3038
- **KID (Kernel Inception Distance)**: 0.02246
- **Precision (`pr50k3_full_precision`)**: 0.48944
- **Recall (`pr50k3_full_recall`)**: 0.00765
- **SSIM (Structural Similarity Index)**:
  - Normal class: 0.3121 ± 0.0483
  - Pneumonia class: 0.3713 ± 0.0721
  - Overall: 0.3417 ± 0.0681
- **VGG16 Downstream Test Accuracy (50/50 real-synthetic train mix)**: 97.99%
  - Normal (precision/recall): 0.9698 / 0.9519
  - Pneumonia (precision/recall): 0.9834 / 0.9897

**Interpretation:**
- This is a major improvement over run 00017 snapshot 300 (FID 51.7074, KID 0.06121).
- Snapshot 000240 is the strongest checkpoint observed so far in this repository.
- Precision improved over earlier baselines, while recall remains low and should be treated as a key limitation.

### Convergence Behavior

Observed during training:
- **Loss/G/loss**: Decreases from ~19.6 (initial) to ~2.1 (final), indicating generator learning.
- **Loss/D/loss**: Oscillates ~0–1.3, normal discriminator dynamics.
- **Augmentation (ADA)**: Increases from 0.0 to ~0.18 over first 6 ticks, stabilizes ~0.11–0.18 range.
- **Stability**: No divergence, no NaN/inf throughout 100 kimg run.

## 5. Repository Layout

- `configs/train_256_conditional.yaml`: Baseline training configuration
- `scripts/prepare_mendeley_dataset.py`: Dataset preparation and resizing to 256×256
- `scripts/create_stylegan_zip.sh`: Wraps NVIDIA `dataset_tool.py` to create StyleGAN-compatible zip
- `scripts/launch_stylegan_train.py`: Training launcher wrapper with config integration
- `scripts/bootstrap_stylegan_repo.sh`: Clones official StyleGAN2-ADA repo
- `third_party/stylegan2-ada-pytorch/`: StyleGAN2-ADA source code (submodule or cloned)
- `.gitignore`: Excludes large artifacts (training-runs/, datasets/, .venv/)
- `requirements.txt`: Python dependencies
- `README.md`: This file

## 6. Usage Instructions

### Environment Setup

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Clone StyleGAN2-ADA (one-time setup)
bash scripts/bootstrap_stylegan_repo.sh
```

### Dataset Preparation

Prepare your dataset with class folders named containing "NORMAL" and "PNEUMONIA":

```powershell
python scripts/prepare_mendeley_dataset.py `
  --input-root data/raw/chest_xray `
  --output-root data/processed/mendeley_256 `
  --image-size 256
```

Output structure:
```
data/processed/mendeley_256/
├── 0_normal/
└── 1_pneumonia/
```

Build StyleGAN-compatible zip:

```powershell
bash scripts/create_stylegan_zip.sh `
  third_party/stylegan2-ada-pytorch `
  data/processed/mendeley_256 `
  datasets/pneumonia_256_conditional.zip
```

### Training

```powershell
# Set fallback ops for Windows
$env:STYLEGAN_DISABLE_CUSTOM_OPS="1"

# Run training
.\.venv\Scripts\python.exe third_party\stylegan2-ada-pytorch\train.py `
  --outdir=training-runs `
  --cfg=auto `
  --data=datasets/pneumonia_256_conditional.zip `
  --gpus=1 --batch=4 --gamma=2.0 --kimg=100 --snap=5 --workers=2 --seed=42 `
  --aug=ada --target=0.6 --metrics=none --cond=1 --mirror=0
```

## 7. Evaluation & Testing Protocol

### Generate Synthetic Samples

Generate fixed-seed image sets per class for reproducible comparison:

```powershell
$env:STYLEGAN_DISABLE_CUSTOM_OPS="1"

# Class 0 (Normal)
.\.venv\Scripts\python.exe third_party\stylegan2-ada-pytorch\generate.py `
  --network=training-runs/00011-pneumonia_256_conditional-cond-auto1-gamma2-kimg100-batch4-ada-target0.6-resumecustom/network-snapshot-000100.pkl `
  --seeds=0-63 `
  --class=0 `
  --outdir=training-runs/00011-pneumonia_256_conditional-cond-auto1-gamma2-kimg100-batch4-ada-target0.6-resumecustom/eval_samples/class0_seed0-63

# Class 1 (Pneumonia)
.\.venv\Scripts\python.exe third_party\stylegan2-ada-pytorch\generate.py `
  --network=training-runs/00011-pneumonia_256_conditional-cond-auto1-gamma2-kimg100-batch4-ada-target0.6-resumecustom/network-snapshot-000100.pkl `
  --seeds=0-63 `
  --class=1 `
  --outdir=training-runs/00011-pneumonia_256_conditional-cond-auto1-gamma2-kimg100-batch4-ada-target0.6-resumecustom/eval_samples/class1_seed0-63
```

Review for:
- Anatomical realism
- Artifact rates
- Diversity
- Class-consistent visual patterns (e.g., presence/absence of opacities)

### Compute Quantitative Metrics

Evaluate FID and KID for checkpoint comparison:

```powershell
$env:STYLEGAN_DISABLE_CUSTOM_OPS="1"

.\.venv\Scripts\python.exe third_party\stylegan2-ada-pytorch\calc_metrics.py `
  --metrics=fid50k_full,kid50k_full `
  --data=datasets/pneumonia_256_conditional.zip `
  --network=training-runs/00011-pneumonia_256_conditional-cond-auto1-gamma2-kimg100-batch4-ada-target0.6-resumecustom/network-snapshot-000100.pkl
```

Repeat for alternative snapshots (000080, 000060) using identical settings for fair comparison.

### Checkpoint Selection Criteria

1. **Quantitative**: Lowest FID/KID among candidates
2. **Qualitative**: Visual inspection for realism, diversity, no obvious artifacts
3. **Leakage check**: Verify generated images are not near-duplicates of training data
4. **Decision rule**: If kimg 100 shows only marginal metric improvement but worse visual quality, prefer earlier checkpoint (kimg 80 or 60)

### Reviewer Evaluation Suite (Snapshot 000200)

To satisfy reviewer requests, run the following suite for the selected StyleGAN snapshot:

`training-runs/00013-pneumonia_256_conditional-cond-auto1-gamma2-kimg200-batch4-ada-target0.6-resumecustom/network-snapshot-000200.pkl`

#### 1) FID and KID (StyleGAN, official)

```powershell
$env:STYLEGAN_DISABLE_CUSTOM_OPS="1"

.\.venv\Scripts\python.exe third_party\stylegan2-ada-pytorch\calc_metrics.py `
  --metrics=fid50k_full,kid50k_full `
  --data=datasets/pneumonia_256_conditional.zip `
  --network=training-runs/00013-pneumonia_256_conditional-cond-auto1-gamma2-kimg200-batch4-ada-target0.6-resumecustom/network-snapshot-000200.pkl
```

#### 2) Precision and Recall for GANs (StyleGAN, official)

```powershell
$env:STYLEGAN_DISABLE_CUSTOM_OPS="1"

.\.venv\Scripts\python.exe third_party\stylegan2-ada-pytorch\calc_metrics.py `
  --metrics=pr50k3_full `
  --data=datasets/pneumonia_256_conditional.zip `
  --network=training-runs/00013-pneumonia_256_conditional-cond-auto1-gamma2-kimg200-batch4-ada-target0.6-resumecustom/network-snapshot-000200.pkl
```

#### 3) SSIM (class-matched real vs synthetic)

First generate synthetic images from snapshot 200:

```powershell
$env:STYLEGAN_DISABLE_CUSTOM_OPS="1"

# Normal class
.\.venv\Scripts\python.exe third_party\stylegan2-ada-pytorch\generate.py `
  --network=training-runs/00013-pneumonia_256_conditional-cond-auto1-gamma2-kimg200-batch4-ada-target0.6-resumecustom/network-snapshot-000200.pkl `
  --seeds=0-511 --class=0 `
  --outdir=outputs/snapshot200/class0

# Pneumonia class
.\.venv\Scripts\python.exe third_party\stylegan2-ada-pytorch\generate.py `
  --network=training-runs/00013-pneumonia_256_conditional-cond-auto1-gamma2-kimg200-batch4-ada-target0.6-resumecustom/network-snapshot-000200.pkl `
  --seeds=512-1023 --class=1 `
  --outdir=outputs/snapshot200/class1
```

Then compute SSIM:

```powershell
.\.venv\Scripts\python.exe scripts/eval_ssim_pairs.py `
  --real-zip datasets/pneumonia_256_conditional.zip `
  --synthetic-normal-dir outputs/snapshot200/class0 `
  --synthetic-pneumonia-dir outputs/snapshot200/class1 `
  --max-pairs 256
```

#### 4) VGG16 downstream accuracy at 256x256 with 50/50 real-synthetic train mix

Prepare synthetic classifier folders (same layout as real):

```text
outputs/snapshot200/for_classifier/
  0_normal/
  1_pneumonia/
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts/train_vgg16_real_synth_split.py `
  --real-root data/processed/mendeley_256 `
  --synth-root outputs/snapshot200/for_classifier `
  --epochs 10 --batch-size 32
```

Report the held-out real test accuracy against prior 92.43% baseline.

#### 5) Visual Turing Test (radiologist review packet)

Build a blinded CSV packet for independent reviewers:

```powershell
.\.venv\Scripts\python.exe scripts/build_visual_turing_packet.py `
  --real-normal-dir data/processed/mendeley_256/0_normal `
  --real-pneumonia-dir data/processed/mendeley_256/1_pneumonia `
  --synth-normal-dir outputs/snapshot200/class0 `
  --synth-pneumonia-dir outputs/snapshot200/class1 `
  --n-per-group 100 `
  --outdir outputs/visual_turing_snapshot200
```

Outputs:
- `outputs/visual_turing_snapshot200/blinded_review.csv` (send to radiologists)
- `outputs/visual_turing_snapshot200/answer_key.csv` (keep hidden)

#### 6) cGAN comparison requirement

Run the same evaluation protocol for cGAN outputs/checkpoints and report both models side-by-side:

- FID (same reference real dataset and sample count)
- SSIM (same pairing strategy and max pairs)
- Precision/Recall (same feature extractor and settings)
- VGG16 downstream accuracy (same split policy)
- Visual Turing pass rate (same reviewer packet size and review instructions)

## 8. Final Testing Outcomes and Checkpoint Selection

Date recorded: April 4, 2026

**Quantitative Results (FID/KID):**

| Snapshot | kimg | FID | KID | Note |
|----------|------|-----|-----|------|
| 000080 | 80 | 159.0 | - | Early baseline |
| 000100 | 100 | 133.3156 | 0.1831 | Improved vs 80 |
| 000200 | 200 | 109.9 | 0.14 | Older baseline |
| 000220 | 220 | 110.3611 | - | Essentially tied with 200, slightly worse |
| 000240 | 240 | 152.0 | - | Clear degradation onset |
| 000300 | 300 | 337.0 | - | Overtraining collapse |
| 000400 | 400 | 407.0 | - | Severe collapse |

| 000200 (gamma 8) | 200 | 54.0 | 0.06 | Previous best |
| 000240 (gamma 6, ADA 0.65) | 240 | 25.3038 | 0.02246 | **Final selected checkpoint** |

**Precision/Recall for GANs (snapshot 000240):**

| Metric | Value |
|--------|-------|
| pr50k3_full_precision | 0.48944 |
| pr50k3_full_recall | 0.00765 |

Interpretation for reviewer context:
- Precision improved substantially over earlier checkpoints, while recall remains low under `pr50k3_full`.
- This supports selecting snapshot 000240 for strongest fidelity while explicitly reporting limited manifold coverage.
- Additional evidence (SSIM, downstream VGG16, and radiologist Turing test) remains necessary for comprehensive validation.

**SSIM evaluation (gamma 6, ADA 0.65, snapshot 000240):**

| Class | Mean SSIM | Std | n |
|-------|-----------|-----|---|
| Normal | 0.3121 | 0.0483 | 256 |
| Pneumonia | 0.3713 | 0.0721 | 256 |
| Overall | 0.3417 | 0.0681 | 512 |

Interpretation:
- Pneumonia samples show slightly stronger structural similarity than Normal samples.
- SSIM remains moderate overall, supporting visual plausibility while still leaving room for diversity-focused improvements.

**VGG16 downstream evaluation (50/50 real-synthetic train mix, held-out real test):**

- Test accuracy: **97.99%**

Confusion matrix:

| True\\Pred | Normal | Pneumonia |
|-----------|--------|-----------|
| Normal | 257 | 13 |
| Pneumonia | 8 | 769 |

Classification summary:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.9698 | 0.9519 | 0.9607 | 270 |
| Pneumonia | 0.9834 | 0.9897 | 0.9865 | 777 |
| Accuracy | - | - | 0.9799 | 1047 |
| Macro avg | 0.9766 | 0.9708 | 0.9736 | 1047 |
| Weighted avg | 0.9799 | 0.9799 | 0.9799 | 1047 |

Interpretation:
- Downstream utility is strong and clearly exceeds the prior 92.43% baseline.
- This supports the practical usefulness of the improved StyleGAN snapshot for augmentation.

**Final Selection for Research:**
- **Chosen checkpoint: snapshot 000240 from run 00018 (gamma 6.0, ADA target 0.65)**
- Rationale: best measured FID/KID in this repository and improved precision versus prior baselines.
- Practical conclusion: this is the strongest StyleGAN baseline available for the cGAN comparison, with low recall documented as a limitation.

### Optimization Attempt 2: Improved Precision/Recall (April 2, 2026 – Completed)

Initial results showed low precision/recall (0.096/0.0) despite good FID. The retrained gamma-8 run now has stronger quantitative performance:

**New hyperparameters:**
- `--gamma=8.0` (increased from 2.0 for stronger R1 regularization → better manifold coverage)
- `--target=0.7` (increased from 0.6 for more augmentation diversity)
- `--kimg=300` (extended to capture convergence and find optimal plateau)
- All other settings preserved: batch=4, snap=5, seed=42, mirror=0

**Expected improvements:**
- Strong gamma encourages generator to explore diverse modes (targeting higher recall).
- Slightly higher ADA target provides gentler guidance for manifold learning.
- Dense snapshots (every 20 kimg from 0–300) enable precise checkpoint selection.

**Updated results:**
- FID 54.0 at snapshot 200
- KID 0.06 at snapshot 200
- pr50k3_full precision 0.2284
- pr50k3_full recall 0.00038
- SSIM overall 0.3456 (Normal 0.3145, Pneumonia 0.3768)
- VGG16 held-out real test accuracy 98.09%

**Evaluation plan:**
- Completed: SSIM and VGG16 on gamma-8 snapshot 200
- Remaining: Visual Turing test packet review and side-by-side cGAN table finalization
- Keep the gamma 2 baseline in the table for comparison only

**Usage for comparison study:**
- Use `run 00018 / snapshot 000240 / gamma 6.0 / ADA target 0.65` as the primary StyleGAN2-ADA result when comparing against cGAN baselines under the same protocol.

## 9. Future Work

- **Addressing Class Imbalance**: Implement weighted loss functions to improve recall for the minority "Normal" class
- **Multi-Class Expansion**: Extend conditional generation to include COVID-19 and Tuberculosis
- **Resolution Scaling**: Pursue 512×512 or higher resolution for improved automated diagnosis in resource-constrained settings
- **Downstream Utility**: Train pneumonia classifiers with synthetic + real augmentation and benchmark against real-only baselines
- **Hyperparameter Sweep**: Systematic exploration of gamma (1, 2, 4) and ADA target (0.5, 0.6, 0.7)
- **StyleGAN2-ADA vs cGAN**: Publish comparative benchmarking against previous 64×64 cGAN implementation on identical metrics

## 10. Important Notes

- **Clinical Use**: This work focuses on GAN training and evaluation. Downstream clinical model development and regulatory validation are out of scope.
- **Data Ethics**: Keep `mirror=0` for chest X-rays unless clinical guidance explicitly allows horizontal flips. Synthetic data should not replace real clinical samples but augment limited datasets responsibly.
- **Reproducibility**: All commands require the Windows fallback ops environment variable. Ensure identical seeds, class balance, and metric settings for fair cross-checkpoint comparisons.
- **Performance**: Single GPU (RTX 86xx) used; multi-GPU scaling possible with standard PyTorch DDP.

## 11. Citation

If you use this work, please cite the original research:

```bibtex
@article{sawhney2024synthetic,
  title={Synthetic Chest X-ray Image Generation for Normal and Pneumonia Classes Using Conditional GANs},
  author={Sawhney, Garv and Bairwa, Amit Kumar and Narooka, Preeti and Tiwari, Varun},
  year={2024}
}
```

And cite StyleGAN2-ADA:

```bibtex
@inproceedings{karras2020training,
  title={Training Generative Adversarial Networks with Limited Data},
  author={Karras, Tero and Aittala, Miika and Hellsten, Jarmo and Laine, Samuli and Lehtinen, Jaakko},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

---

**Last Updated**: April 4, 2026  
**Status**: Training/evaluation complete; checkpoint selection finalized at run 00018 snapshot 000240 (kimg 240).  
**Contact**: For questions or reproducibility issues, please file an issue or contact the maintainer.
