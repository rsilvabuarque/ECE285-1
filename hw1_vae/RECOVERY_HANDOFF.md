# Homework 1 recovery and supercomputer handoff

## What was recovered

On October 30, 2026 (America/Los_Angeles), the local Codex session files were
read to recover the work from the inaccessible **Implement Homework 1 VAE**
chat. The matching session is `01a18eeb-fe56-7443-b0a6-1ffa099acdf1`.
Its two completed turns ran approximately 3:26–3:34 pm and 3:36–3:46 pm PDT.
The requests were to implement and explain the assigned chest X-ray VAE,
then prepare GPU diagnostics and a portable teaching notebook.

The recovered transcript and repository agree: **implementation and synthetic
CPU validation are complete; real chest X-ray training has not started.**
There are no real-data checkpoints, loss curves, or final FID/IS scores to
transfer. The Codespace exposed no NVIDIA hardware or driver and used
CPU-only PyTorch, so the laptop's RTX 4080 was never used.

The screenshot's message-loading and other-app errors do not establish their
underlying cause. This recovery preserves the work without changing the
application's session database or ownership state.

## Files preserved in Git

| Files | Purpose |
| --- | --- |
| `model.py`, `data.py`, `utils.py` | Convolutional VAE, split-preserving image loader, device/checkpoint helpers. |
| `train.py` | Training, validation, atomic best/last checkpoints, resume, history and plots. |
| `generate.py`, `evaluate.py` | Prior sampling and the assigned Ignite FID/IS evaluation protocol. |
| `check_compute.py`, `smoke_test.py` | Compute diagnostics and offline synthetic checks. |
| `HW1_VAE.ipynb`, `build_notebook.py` | Self-contained, unexecuted teaching notebook and its reproducible builder. |
| `README.md`, `GPU_SETUP.md`, `REPORT_OUTLINE.md` | Study guide, hardware setup, and two-page report outline. |
| `requirements.txt`, `requirements-notebook.txt` | Script and optional notebook dependencies. |
| `../for_chat_gpt/*.pdf` | Homework handout, study guide, and project proposal supplied before recovery. |

The default model uses 64×64 grayscale images, four encoder/decoder stages,
128 latent coordinates, and pixel-mean MSE plus `0.001 × KL`. It has
2,955,201 trainable parameters. Evaluation explicitly uses the tutorial's
1000-dimensional Inception probability representation for FID; describe that
protocol in the report.

## What actually ran

The original session logs and retained temporary artifacts establish:

- PDF text extraction and inspection of lecture frames to scope the implementation.
- Creation of a Python 3.12 virtual environment with CPU PyTorch, project
  dependencies, JupyterLab, and an `ece285-vae` notebook kernel.
- `python -m hw1_vae.smoke_test`: synthetic data loading, model shapes,
  gradients, analytic KL, checkpoint reload, and metric/preprocessing checks.
- Synthetic CLI training, resume, generation, and pretrained-Inception
  evaluation. The small fixture contained 10 train, 4 validation, and 4 test
  artificial images. Resumed and uninterrupted training matched exactly.
- A standalone execution of the notebook outside the repository, plus a
  notebook pilot/resume/generation/export test and notebook schema/source checks.
- GPU diagnostics showing no CUDA device; requiring CUDA correctly failed.

The synthetic Inception evaluation completed, but its scores measure pipeline
execution on artificial images and must not be used as homework results.
Recovery also reran the offline smoke suite and a fresh synthetic CLI
train/resume/generate comparison; resumed model tensors again matched the
uninterrupted run exactly. Notebook schema, Python cell syntax, all nine embedded
modules, and installed dependency consistency passed verification.

The recovered CPU environment used Python 3.12.3, Torch 2.15.0+cpu,
Torchvision 0.29.2+cpu, Ignite 0.5.5, NumPy 2.5.4, and SciPy 1.18.1.
These describe the recovery environment; install a CUDA-compatible pair on the
cluster instead of reproducing the CPU wheel selection.

A local, ignored `.recovery/` directory contains the original session logs,
a readable user/assistant transcript, extracted shell commands, CPU dependency
snapshot, and archived surviving synthetic test artifacts. These are recovery
records, not training results. Download them separately if you want to keep
them after deleting the Codespace. The 85 MiB lecture MP4 also remains local
and is ignored. The code and notebook run without that video.

## Continue on the university supercomputer

During the recovery commit, the Codespaces Git hook detected that the original
`chenyzzn/ECE285` repository was read-only for this account. It configured the
writable fork `rsilvabuarque/ECE285-1` as `origin` and the original repository
as `upstream`.

First, from this Codespace, upload the new commit to your fork:

```bash
git push origin main
```

Then on the cluster, clone the repository (or pull in an existing clean clone):

```bash
git clone https://github.com/rsilvabuarque/ECE285-1.git ECE285
cd ECE285
git log -1 --oneline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use the university's documented Python/CUDA modules or environment setup and
install a matching CUDA-enabled Torch/Torchvision pair using the
[official PyTorch selector](https://pytorch.org/get-started/locally/). Then:

```bash
python -m pip install -r hw1_vae/requirements.txt
python -m hw1_vae.smoke_test
```

For notebook use, also install `hw1_vae/requirements-notebook.txt` and select
the environment's kernel. Scripts work directly from the clone. The standalone
notebook writes its own embedded package to its configured workspace; edits
there need to be copied back to the clone if you want to version them.

Obtain a GPU compute allocation using the university's scheduler, and run the
following commands inside that allocation from the repository root. The
cluster name, scheduler, account, partition, and module names were not provided,
so no site-specific job directives are assumed here.

Choose permitted storage locations, replacing the example paths below. Keep
checkpoints on storage that survives the job and back them up if scratch has
an expiry policy:

```bash
export VAE_DATA_DIR=/path/to/chest_xray
export VAE_RUN_DIR=/path/to/persistent/hw1_vae_run

python -m hw1_vae.check_compute --test-vae --require-cuda \
  --output "$VAE_RUN_DIR/compute.json"
python -m hw1_vae.data --data-dir "$VAE_DATA_DIR"

python -m hw1_vae.train \
  --data-dir "$VAE_DATA_DIR" --output-dir "$VAE_RUN_DIR" \
  --device cuda --epochs 2 --batch-size 64
```

The dataset must contain the official `train`, `val`, and `test` directories,
each containing `NORMAL` and `PNEUMONIA`. Download and extract the assigned
Kaggle dataset first; `python -m hw1_vae.data --download` is an optional helper
that prints its resolved location. If compute nodes have no Internet, prepare
the dataset and pretrained Inception weights on a permitted networked node
using storage visible to the compute allocation. The evaluator's
`InceptionProbabilities()` constructor downloads/caches its weights on first
use. Set the same `TORCH_HOME` on both nodes if using a shared custom cache.

Inspect the pilot's loss curves, reconstruction/prior samples, and epoch time.
Then resume to **50 total epochs**, keeping the complete run directory:

```bash
python -m hw1_vae.train \
  --data-dir "$VAE_DATA_DIR" --output-dir "$VAE_RUN_DIR" \
  --device cuda --epochs 50 --batch-size 64 \
  --resume "$VAE_RUN_DIR/last.pt"

python -m hw1_vae.generate \
  --checkpoint "$VAE_RUN_DIR/best.pt" \
  --output-dir "$VAE_RUN_DIR/generated" --device cuda --num-samples 64

python -m hw1_vae.evaluate \
  --checkpoint "$VAE_RUN_DIR/best.pt" --data-dir "$VAE_DATA_DIR" \
  --output-dir "$VAE_RUN_DIR/evaluation" --device cuda --num-samples 0

python -m pip freeze > "$VAE_RUN_DIR/requirements-resolved.txt"
```

Training, generation, and evaluation must finish within a GPU allocation or
be submitted as appropriately sized jobs. Evaluation is separate work beyond
the training budget. If interrupted, resume from `last.pt`; preserve its
matching `best.pt` when moving the run. The training code also checks the
dataset manifest before resuming.

Git preserves code and documentation. Dataset files, virtual environments,
notebook workspaces, and experiment outputs are ignored, so back up the whole
run directory separately. Use [REPORT_OUTLINE.md](REPORT_OUTLINE.md) after the
real run to assemble settings, loss curves, generated examples, and final
scores into the required two-page NeurIPS report.
