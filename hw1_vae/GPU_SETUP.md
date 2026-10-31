# Run the VAE on your RTX 4080 or another GPU

## Why the current workspace cannot see the laptop

The workspace inspected on October 30, 2026 is an Azure-hosted GitHub Codespace. It exposes a Microsoft virtual display device (`1414:5353`, `hyperv_drm`), no NVIDIA device files or driver, and a CPU-only PyTorch build. The laptop's RTX 4080 is on a separate computer. A browser or VS Code connection to this Codespace carries editing traffic; it does not pass through the laptop's PCI devices.

GitHub documents that [Codespaces run in cloud VMs and containers](https://docs.github.com/en/codespaces/about-codespaces/deep-dive), and [retired the GPU Codespaces machine type in August 2025](https://github.blog/changelog/2025-08-01-upcoming-deprecation-of-gpu-machine-type-in-codespaces/). Adding CUDA packages or a `--gpus all` container flag to this VM cannot create access to the RTX 4080.

Run this diagnostic from the repository root whenever you change machines or notebook kernels:

```bash
python -m hw1_vae.check_compute --test-vae --require-cuda --output runs/compute.json
```

It records hardware visibility, the active interpreter, PyTorch's build, driver-reported GPU names, and CUDA execution. If CUDA is available it tests a matrix operation and a small VAE forward/backward. Exit code `0` means the requested checks passed; `2` means CUDA was required but unavailable; `1` means an execution test failed. Without `--require-cuda`, CPU-only inspection exits successfully while explicitly reporting that CUDA is unavailable. No driver or package changes occur.

## Option A: use the laptop directly

Copy or clone the updated repository onto the laptop, including `hw1_vae`.
Push local commits before cloning elsewhere; the
[recovery and supercomputer handoff](RECOVERY_HANDOFF.md) gives the transfer
commands. Create a new environment on the laptop rather than copying this
Codespace's `.venv`.

### Windows with native Python

1. Open a **local** PowerShell terminal and run `nvidia-smi`. Confirm the RTX 4080 appears. If it does not, install or update the laptop's NVIDIA Windows driver before proceeding.
2. From the local repository root, create the environment. Using the executable directly avoids PowerShell activation-policy issues:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

3. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) with **Windows / Pip / Python / CUDA**. Run its command using `.\.venv\Scripts\python.exe -m pip` instead of bare `pip`. The CUDA build must suit the installed NVIDIA driver. Start with a fresh environment so a preinstalled CPU build cannot silently satisfy the requirement. No CUDA source compilation is needed for this VAE.
4. Install the project and notebook dependencies and register the kernel:

```powershell
.\.venv\Scripts\python.exe -m pip install -r hw1_vae/requirements.txt -r hw1_vae/requirements-notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name ece285-vae --display-name "ECE285 VAE"
.\.venv\Scripts\python.exe -m hw1_vae.check_compute --test-vae --require-cuda
.\.venv\Scripts\python.exe -m jupyterlab
```

5. Open the homework notebook and select **ECE285 VAE**. Check that its diagnostic cell names the RTX 4080. The code defaults to `num_workers=0`, which works with Windows; increase worker count only after the initial run works.

### Linux or Ubuntu under WSL 2

On Linux, install a compatible NVIDIA driver on the actual GPU host and verify `nvidia-smi`. On Windows with WSL, use WSL 2 and update WSL if necessary. NVIDIA's [WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) specifies that the NVIDIA Windows driver supplies GPU support to WSL; do not install a Linux display driver inside WSL. In WSL, `nvidia-smi` may also be available at `/usr/lib/wsl/lib/nvidia-smi`.

From the local Linux/WSL repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CUDA-enabled pair from the [PyTorch selector](https://pytorch.org/get-started/locally/) for **Linux / Pip / Python / CUDA**, then run:

```bash
python -m pip install -r hw1_vae/requirements.txt -r hw1_vae/requirements-notebook.txt
python -m ipykernel install --user --name ece285-vae --display-name "ECE285 VAE"
python -m hw1_vae.check_compute --test-vae --require-cuda
python -m jupyterlab
```

If `venv` reports missing `ensurepip`, install your distribution's Python venv support and recreate the environment. If replacing an existing CPU-only installation, remove `torch` and `torchvision` from that environment, install the selector's CUDA pair, and restart the notebook kernel.

### Choose the execution kernel in VS Code

Open the **local** repository folder in VS Code, or open that folder through **WSL**. Open the notebook and choose **Select Kernel → Python Environments → the local `.venv`**, or the registered **ECE285 VAE** kernel. The [VS Code kernel documentation](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) explains these choices. A Python environment selected inside a Codespaces window still runs on the Codespace.

## Option B: use a hosted notebook or GPU server

- **Colab:** upload the homework notebook or open the updated files there. In its runtime settings request a GPU, then run the notebook's compute diagnostic. Use the accelerator already assigned to that runtime. Hardware availability varies; a request alone is not evidence that CUDA is active. See [Colab's accelerator and resource FAQ](https://research.google.com/colaboratory/faq.html).
- **Kaggle:** create a notebook with a GPU accelerator, attach the homework's `paultimothymooney/chest-xray-pneumonia` dataset, and set the notebook's data path to the attached input directory. Run the same CUDA diagnostic before training. Use the runtime's preinstalled matching PyTorch/Torchvision pair if it already works. See [Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks).
- **A university or other GPU server:** transfer the updated code/notebook, create a Python environment on that machine, and run the diagnostic on the allocated GPU node. On a cluster, opening a terminal on the login node does not allocate a GPU. Follow the cluster's scheduler instructions. VS Code can use a Jupyter server on that machine, as described in its [remote notebook documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_connect-to-a-remote-jupyter-server).

The notebook must execute in the compute environment that owns the GPU. Merely opening the file on a laptop or browser does not determine the execution host.

## Data and checkpoint portability

Set dataset and output paths in the notebook for each machine. Preserve the official `train`, `val`, and `test` directories. Downloading the dataset and pretrained Inception weights needs network access on first use.

To continue an interrupted run elsewhere, transfer the **whole output directory**, including `best.pt`, `last.pt`, history, and configuration. Point the resumed command at `last.pt`; use `--epochs` as the total target, not the number of extra epochs. A moved dataset is accepted if its relative filenames and sizes match the saved manifest. Exact reproducibility can vary across hardware and PyTorch releases, even with the same seed.

For a container on a real GPU host, the host must expose an NVIDIA device and have a working driver before container configuration helps. NVIDIA documents the additional [Container Toolkit prerequisites and Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The direct local Python setup above avoids that extra layer.
