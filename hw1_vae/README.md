# Homework 1: learn and implement a chest X-ray VAE

This directory contains the Python VAE for the [Homework 1 handout](../for_chat_gpt/mv47c1gud5n1ka.pdf). The existing repository's C++ renderer, scenes, and rendered images are a separate graphics project. The other supplied materials are the [VAE lecture](../for_chat_gpt/Implementation%20of%20VAE.mp4), the [earlier study guide](../for_chat_gpt/Deep%20Generative%20Models%20Study%20Guide.pdf), and a [research project proposal](../for_chat_gpt/ECE285_project_proposal.pdf).

The homework asks you to implement a VAE, train it on the linked [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), generate new images, and submit code plus a **two-page report in NeurIPS format** to Gradescope. Its suggested report contents are hyperparameters, loss curves, Inception Score, FID, and generated examples. Image size, network width, latent dimension, and training duration are design choices. See [REPORT_OUTLINE.md](REPORT_OUTLINE.md) for a compact report plan.

## Run the teaching notebook

Open [HW1_VAE.ipynb](HW1_VAE.ipynb) and select **Python (ECE285 VAE)** in this workspace, or the appropriate GPU kernel on your own computer. JupyterLab and that kernel are installed in this workspace. To launch JupyterLab from the repository root:

```bash
.venv/bin/python -m jupyter lab hw1_vae/HW1_VAE.ipynb
```

The notebook contains the full implementation in visible, editable source cells and can be uploaded by itself to Colab, Kaggle, or another Jupyter environment. It writes a separate working copy of the Python package, leaving the repository source in place. Its default Run All executes small synthetic checks; real downloads, training, final metrics, and export each have a switch in the configuration cell. Set the dataset path and enable the two-epoch pilot when ready.

The current Codespace is a cloud VM with no NVIDIA GPU exposed. Your laptop's RTX 4080 requires a kernel running on the laptop or another GPU host. See [GPU_SETUP.md](GPU_SETUP.md) for Windows, WSL/Linux, hosted notebooks, and remote GPU instructions. Verify any new environment with:

```bash
python -m hw1_vae.check_compute --test-vae --require-cuda
```

For a fresh notebook environment, install the correct PyTorch build first, then `python -m pip install -r hw1_vae/requirements-notebook.txt`. The notebook also provides a controlled missing-dependency installation cell for hosted runtimes. Register a local kernel with `python -m ipykernel install --user --name ece285-vae --display-name "Python (ECE285 VAE)"`.

After repository module changes, `python -m hw1_vae.build_notebook` refreshes the notebook's embedded source. This replaces the clean notebook, so keep personal notebook edits or saved outputs in a separate copy.

## 1. Follow an image through the model

Start by reading [data.py](data.py), then [model.py](model.py), then [train.py](train.py). The important path is:

```text
real image x ── encoder ──> mu, log_var ── sample z ── decoder ──> reconstruction
                                               ↑
                                  standard-normal noise epsilon

standard-normal sample z ── decoder ──> fresh generated image
```

The first path learns from existing images. The second path is the generation required by the homework.

### Step 1: turn a file into a tensor

The dataset loader reads the provided split directories, converts each image to grayscale, resizes it to `64×64`, and converts pixel values to floating point in `[0, 1]`. A batch of `B` images therefore has shape `[B, 1, 64, 64]`: batch, channels, height, width.

Both `NORMAL` and `PNEUMONIA` images contribute to the unconditional image distribution. The network learns from the image itself; class labels are unnecessary for this objective. Preserve the dataset's `train`, `val`, and `test` folders: training updates use `train`, checkpoint selection uses `val`, and final FID uses held-out `test` images. The official validation folder is very small, so its loss can be noisy and provides limited evidence for model selection. Record the actual file counts printed by the loader.

Resizing to `64×64` makes a first experiment practical, while removing fine anatomical detail. The chosen preprocessing directly determines what the model can reproduce.

### Step 2: compress spatial features with the encoder

`ConvVAE` applies four convolution blocks. Each convolution uses kernel size 4, stride 2, and padding 1, followed by LeakyReLU. Stride 2 halves width and height; the channel count grows as spatial detail is summarized.

| Operation | Output shape for the default settings |
| --- | --- |
| Input | `[B, 1, 64, 64]` |
| Encoder block 1 | `[B, 32, 32, 32]` |
| Encoder block 2 | `[B, 64, 16, 16]` |
| Encoder block 3 | `[B, 128, 8, 8]` |
| Encoder block 4 | `[B, 256, 4, 4]` |
| Flatten | `[B, 4096]` |
| Mean head `mu` | `[B, 128]` |
| Log-variance head `log_var` | `[B, 128]` |

The two linear heads describe a diagonal Gaussian distribution, `q(z | x)`, for each image. A latent dimension of 128 means the encoder predicts 128 means and 128 log variances, then samples a vector containing 128 coordinates. These coordinates are learned numerical features; an individual coordinate need not correspond to a recognizable anatomical property.

### Step 3: sample while keeping gradients

The implementation uses the reparameterization trick:

```python
std = torch.exp(0.5 * log_var)
epsilon = torch.randn_like(std)
z = mu + std * epsilon
```

`exp(log_var)` is the variance, so taking half the log before exponentiation produces the standard deviation. Randomness enters through `epsilon`. Gradients can flow through the multiplication and addition into both encoder heads, allowing reconstruction error to teach the encoder.

The prior used for generation is `p(z) = N(0, I)`: every latent coordinate has mean zero and variance one, independently. This assumption concerns latent vectors; it does not say that X-ray pixels follow a standard-normal distribution.

### Step 4: expand a latent vector with the decoder

A linear layer maps `[B, 128]` back to `[B, 256, 4, 4]`. Four transposed convolutions then double spatial size:

```text
[B, 256, 4, 4] → [B, 128, 8, 8] → [B, 64, 16, 16]
              → [B, 32, 32, 32] → [B, 1, 64, 64]
```

The final sigmoid restricts output intensities to `[0, 1]`, matching the input range. The output represents the decoder's predicted image mean. During generation, the code displays that mean rather than adding pixel-level observation noise.

This is a smaller grayscale adaptation of the lecture's convolutional example. The practical lecture slides around 65–66 minutes show RGB inputs, five encoder stages, a 1024-dimensional latent, and Tanh output. Their displayed decoder appears to omit one upsampling stage. Our four-stage mirror has matching input/output dimensions; the smoke test checks this explicitly.

### Step 5: combine reconstruction and KL penalties

Read the loss calculation alongside this equivalent PyTorch expression:

```python
reconstruction_mse = F.mse_loss(recon, images, reduction="mean")
kl = -0.5 * (
    1 + log_var - mu.square() - log_var.exp()
).sum(dim=1).mean()
loss = reconstruction_mse + kl_weight * kl
```

The reconstruction term asks whether each output resembles its input. It averages squared error over **all pixels and images**. MSE can be motivated by a Gaussian observation model with fixed variance.

The KL term measures how far each encoder Gaussian departs from the standard-normal prior. It **sums over latent coordinates, then averages over images**. KL is zero when all means are zero and all variances are one. Minimizing only KL would allow the encoder to discard information about the input, so the balance with reconstruction matters.

The default `kl_weight=0.001` follows the practical lecture's reduction convention. It is a starting hyperparameter, not a universal constant. At `64×64` grayscale resolution, replacing pixel-mean MSE with a pixel sum multiplies the reconstruction term by 4096; leaving the KL weight unchanged would substantially alter training.

The evidence lower bound (ELBO) combines expected log likelihood and KL. Maximizing the standard ELBO corresponds to minimizing a correctly scaled negative log likelihood plus KL. Report this implementation's exact **weighted MSE-plus-KL objective** rather than calling its raw numerical value a unit-weight negative ELBO.

### Step 6: update the weights

For each training batch, [train.py](train.py) performs these operations:

```python
optimizer.zero_grad(set_to_none=True)
reconstruction, mu, logvar = model(images)
losses = vae_loss(reconstruction, images, mu, logvar, kl_weight)
losses["total"].backward()
optimizer.step()
```

Clearing gradients prevents accumulation from the previous batch. Backpropagation differentiates the total loss through the decoder and the reparameterized encoder. Adam updates both networks' parameters. An epoch means one pass through the training split.

Validation runs without gradient updates. Checkpoint selection compares validation loss using the final configured KL weight. The code saves a best checkpoint for later generation and a last checkpoint for resuming. Optional KL warmup gradually increases the training weight over the requested number of epochs; the default uses the full weight from the start.

## 2. Understand the starting settings

| Setting | Default | What changing it does |
| --- | --- | --- |
| Image size | `64×64` | Higher resolution preserves detail and increases memory and computation. |
| Encoder channels | `32, 64, 128, 256` | Wider layers add capacity and cost. |
| Latent dimension | `128` | More coordinates increase the possible information carried by the code. |
| Batch size | `64` images | Larger batches use more memory and average gradients over more images. |
| Optimizer | Adam | Uses running estimates of gradient moments to adapt updates. |
| Learning rate | `0.001` | Controls the update scale. |
| KL weight | `0.001` | Larger values penalize departures from the prior more strongly. |
| KL warmup | `0` epochs | A positive value starts with weaker regularization. |
| Training length | `50` epochs | A starting budget to adjust after measuring actual GPU runtime. |
| Seed | `42` | Makes repeated random choices reproducible within the documented environment. |

Inspect reconstruction, total loss, and KL together. Declining training loss with rising validation loss suggests overfitting. KL near zero together with nearly identical decoded images can indicate posterior collapse: the decoder uses little input-specific latent information. High-quality reconstructions alone do not guarantee good prior samples. Start with one documented run before changing settings, and compare altered settings using validation data.

## 3. Prepare the environment and dataset

Run commands from the root of your repository clone. For moving from the
Codespace to a university cluster, start with the
[recovery and supercomputer handoff](RECOVERY_HANDOFF.md).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a matching PyTorch/Torchvision pair for the training computer using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r hw1_vae/requirements.txt
python -m hw1_vae.smoke_test
```

The smoke test uses synthetic data on CPU and needs neither a dataset nor pretrained Inception weights. It verifies data loading, dimensions, finite losses, gradients, checkpoint reload, and metric plumbing with known synthetic probabilities. It establishes that the code can execute; trained-image quality still requires a real run.

The implementation environment is a Codespace with a CPU-only verification installation. Its GPU is not exposed. On the computer that will train, install the CUDA build appropriate for that computer through the official selector before using `--device cuda`.

If you reuse an environment containing CPU-only PyTorch, remove that pair first with `python -m pip uninstall -y torch torchvision`, then run the selector's CUDA installation command. A plain install command can otherwise leave an already installed CPU build in place. Re-run the GPU check below afterward. On a machine without this `.venv`, create a fresh environment there; virtual environments should not be copied between computers.

Download and extract the [assigned Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Point `--data-dir` at the directory containing these folders:

```text
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

Some archives contain an extra outer `chest_xray` folder. The loader recognizes common wrappers and selects one complete root. Inspect its resolved location and split/class counts before training:

```bash
python -m hw1_vae.data --data-dir data/chest_xray
```

Alternatively, explicitly request the built-in Kaggle download:

```bash
python -m hw1_vae.data --download
```

This uses `kagglehub` and prints the downloaded dataset's resolved root. Use that printed path as `--data-dir` in subsequent commands. If Kaggle requests authentication, follow its sign-in instructions or download the ZIP manually. Dataset files, generated figures, checkpoints, and virtual environments stay outside version control.

## 4. Train when the GPU is available

Check that the Python environment sees the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Begin with two timed epochs:

```bash
python -m hw1_vae.train \
  --data-dir data/chest_xray \
  --device cuda \
  --epochs 2 \
  --batch-size 64 \
  --output-dir runs/hw1_vae
```

Use the measured seconds per epoch to estimate the remaining training budget. The printed timing covers training and validation; allow additional time for plotting, checkpoint writes, and final metric evaluation. The CPU validation performed during implementation does not establish a GPU runtime estimate. If the pilot is healthy, continue to **50 total epochs**:

```bash
python -m hw1_vae.train \
  --data-dir data/chest_xray \
  --device cuda \
  --epochs 50 \
  --batch-size 64 \
  --output-dir runs/hw1_vae \
  --resume runs/hw1_vae/last.pt
```

For faster file loading, try `--num-workers 4` if the machine has sufficient CPU resources. If GPU memory runs out, reduce `--batch-size` to 32 or 16. Keep the resolution and objective conventions recorded in the run configuration when comparing experiments.

The run creates these artifacts:

| File | How to use it |
| --- | --- |
| `config.json`, `environment.json` | Exact settings and the Python/PyTorch/device environment. |
| `data_manifest.json` | Training/validation filenames and sizes used by the run. |
| `history.csv` | Per-epoch train/validation MSE, KL, total objective, KL weights, and measured time. |
| `loss_curves.png` | Three-panel loss figure for inspection and the report. |
| `best.pt` | Model selected by the lowest validation objective. |
| `last.pt` | Most recent model, optimizer, history, and random state for continuing training. |
| `previews/samples_###.png` | Prior-sample grids at each epoch, using fixed latent codes. |
| `reconstructions.png` | Alternating original and reconstruction, for up to eight validation images. |

The reconstruction preview decodes the posterior mean `mu` to make the picture stable. Validation loss instead samples a latent from the full posterior, using fixed validation randomness across epochs. That distinction helps explain why the preview MSE need not equal the recorded validation MSE.

## 5. Generate images and calculate the requested scores

Fresh generation starts with `z ~ N(0, I)` and calls the decoder directly:

```bash
python -m hw1_vae.generate \
  --checkpoint runs/hw1_vae/best.pt \
  --output-dir runs/hw1_vae/generated \
  --num-samples 64 \
  --device cuda
```

These prior samples are the generated examples for the report. The command saves individual PNGs, `samples_grid.png`, the sampled `latents.pt`, and `generation.json` with the seed and checkpoint. A reconstruction uses information from a real input and should be identified separately.

Calculate final metrics on the held-out test split:

```bash
python -m hw1_vae.evaluate \
  --checkpoint runs/hw1_vae/best.pt \
  --data-dir data/chest_xray \
  --output-dir runs/hw1_vae/evaluation \
  --num-samples 0 \
  --device cuda
```

Here, `--num-samples 0` uses the complete test split and an equal number of fresh prior samples. The first evaluation downloads pretrained Inception weights. Evaluation may take appreciable time beyond training. Scores and the complete evaluation protocol are saved to `evaluation/metrics.json` under the chosen run directory.

**FID** compares the means and covariances of real and generated Inception features; lower values are conventionally better. **Inception Score** combines confident per-image class predictions with variety across generated images; higher values are conventionally better. IS does not directly compare generated images with real examples.

The implementation follows the homework's [Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/) metric choice, explicitly using 1000-dimensional Inception class probabilities for the tutorial-compatible FID variant. State that representation in your report; these values are not directly comparable with the common 2048-dimensional pooled-feature FID. Both real and generated images are repeated into three channels, resized to `299×299` using bilinear interpolation with antialiasing, and ImageNet-normalized before the frozen classifier. The evaluator uses one aggregate IS rather than a mean and standard deviation across splits.

Inception was trained for natural-image categories. These scores provide limited evidence about chest X-ray anatomy or pathology, and finite test-set size makes them sensitive to sampling. With fewer than 1001 images, the 1000-feature empirical covariance is rank deficient; the calculation can still run, but the small-sample limitation should be reported. Record the sample count, seed, feature extractor, and preprocessing alongside the values. Combine the numbers with generated images and the loss curves.

## 6. Study checkpoints

Before the full run, explain these without looking at the implementation:

1. Why does the encoder produce two vectors for each input?
2. Why does `exp(0.5 * log_var)` appear in sampling?
3. Which tensor is compared with the image in the reconstruction term?
4. What happens if KL is minimized while reconstruction is ignored?
5. Why can a reconstruction look good when a fresh prior sample looks poor?
6. Why must train, validation, and test images have different roles?
7. Why would changing the loss reduction require reconsidering the KL weight?

For lecture review, prioritize approximately 26–34 minutes for the objective, 54–59 minutes for reparameterization, and 63–69 minutes for the practical code. Those timestamps come from the supplied study guide and inspected lecture frames; they are approximate.
