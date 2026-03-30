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

## Current Project Status (March 30, 2026)

- Training reached **kimg 100** successfully.
- Final completed run directory:
  - `training-runs/00011-pneumonia_256_conditional-cond-auto1-gamma2-kimg100-batch4-ada-target0.6-resumecustom`
- Final checkpoint:
  - `training-runs/00011-pneumonia_256_conditional-cond-auto1-gamma2-kimg100-batch4-ada-target0.6-resumecustom/network-snapshot-000100.pkl`
- Training completed cleanly after tick 25 with `Exiting...` signal.

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
| 00011 | 100 | **Completed** | Final production run; all checkpoints saved |

### Dataset and Labels

- **Dataset source**: Mendeley chest X-ray repository
- **Dataset zip**: `datasets/pneumonia_256_conditional.zip`
- **Resolution**: 256×256 pixels
- **Total samples**: 5,232 images
- **Class distribution**: 0 = Normal, 1 = Pneumonia
- **Preprocessing**: Lanczos resampling, no horizontal flip

## 4. Evaluation Outcomes

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

## 8. First Testing Outcomes (Snapshot 000100)

Date recorded: March 30, 2026

**Quantitative Results:**
- `fid50k_full = 133.3156`
- `kid50k_full = 0.1831`

**Interpretation:**
- Valid baseline indicating a quality gap to real X-ray distribution
- Does not confirm overfitting without additional checks (leakage test, downstream task evaluation)
- Comparison against earlier snapshots (000080, 000060) needed to finalize checkpoint selection

**Next Steps:**
1. Evaluate snapshots 000080 and 000060 with identical protocol
2. Visual inspection and leakage checks for all candidates
3. Selected checkpoint will serve as baseline for StyleGAN2-ADA vs cGAN comparison study

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

**Last Updated**: March 30, 2026  
**Status**: Training complete (kimg 100); initial metrics collected; checkpoint selection pending.  
**Contact**: For questions or reproducibility issues, please file an issue or contact the maintainer.
