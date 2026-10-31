"""Build the portable teaching notebook from the actual implementation files.

Run ``python -m hw1_vae.build_notebook`` after changing module source. This
replaces HW1_VAE.ipynb with a clean notebook; keep personal notebook edits in
a separate copy. No optional notebook libraries are needed to build it.
"""

import hashlib
import json
import textwrap
from pathlib import Path


def build_notebook():
    here = Path(__file__).resolve().parent
    cells = []

    def add(kind, source, tags=()):
        source = textwrap.dedent(source).strip() + "\n"
        digest = hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:12]
        cell = {"cell_type": kind, "id": digest,
                "metadata": {"tags": list(tags)} if tags else {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(source):
        add("markdown", source)

    def code(source, tags=()):
        add("code", source, tags)

    def module(name):
        # Keep the exact module text visible and editable in the notebook.
        add("code", f"%%writefile hw1_vae/{name}\n" + (here / name).read_text(),
            ("embedded-source", name))

    md(r"""
    # ECE 285 Homework 1 — build, understand, and train a chest X-ray VAE

    **This notebook is self-contained.** Download this `.ipynb` alone and open it
    in local Jupyter/VS Code, Colab, Kaggle, or a remote GPU notebook. The code
    cells contain the full Python implementation and write it into a separate
    working directory. They do not need a GitHub login, a repository clone, or
    the lecture video. Python 3.12 is a practical starting environment.

    The homework asks for a VAE trained on
    [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia),
    fresh generated images, and a **two-page NeurIPS-format report** with code.
    We prepare hyperparameters, loss curves, Inception Score, FID, and images.

    **How to use this notebook:**

    1. Choose the machine/kernel below, then run the setup and teaching cells.
    2. Read the tensor shapes, sampling, and loss explanations as you run them.
    3. Set a dataset path or explicitly enable the Kaggle download.
    4. Enable a two-epoch pilot, inspect it, then enable resume for the full run.
    5. Enable generation/evaluation and export the results for your report.

    Default **Run All** performs offline CPU/GPU checks and a tiny synthetic
    learning demonstration. It does not download data, train on real images,
    download Inception weights, or report fabricated experimental results.
    Those stages have explicit switches in the configuration cell.
    """)
    md(r"""
    ## 1. Choose where the Python kernel runs

    A browser on your laptop can display a notebook whose Python process runs
    on another machine. The kernel's machine determines which GPU you can use.
    The workspace inspected for this implementation is **GitHub Codespaces on
    Azure**, with a Microsoft virtual display adapter and no NVIDIA device.
    Installing CUDA in that Codespace cannot attach your laptop's RTX 4080.

    | Environment | Preparation | Kernel choice |
    | --- | --- | --- |
    | Laptop with RTX 4080 | Run Jupyter locally in Windows or WSL2/Linux; install NVIDIA driver and CUDA-enabled PyTorch | Your local `Python (ECE285 VAE)` kernel |
    | Colab | Upload this notebook; choose a GPU runtime in Runtime → Change runtime type | Colab's GPU Python kernel |
    | Kaggle | Upload/import this notebook; enable a GPU accelerator and Internet for downloads; optionally attach the assigned dataset | Kaggle's GPU Python kernel |
    | Remote GPU/JupyterHub | Open/upload the notebook on the allocated GPU node | Kernel on that GPU node |
    | Current Codespace / CPU | Use it to study and verify the implementation | `Python (ECE285 VAE)` here; CPU checks only by default |

    On your laptop, create a fresh environment rather than copying `.venv`
    from the Codespace. In a terminal in your local copy of the repository:

    **Windows PowerShell:**

    ```powershell
    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    # Run the CUDA-enabled torch/torchvision command from the official selector
    # below, using .\.venv\Scripts\python.exe -m pip instead of pip.
    .\.venv\Scripts\python.exe -m pip install -r hw1_vae/requirements-notebook.txt
    .\.venv\Scripts\python.exe -m ipykernel install --user --name ece285-vae --display-name "Python (ECE285 VAE)"
    .\.venv\Scripts\python.exe -m jupyter lab hw1_vae/HW1_VAE.ipynb
    ```

    **Linux / WSL2:**

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    # First run the CUDA-enabled torch/torchvision installation command below.
    python -m pip install -r hw1_vae/requirements-notebook.txt
    python -m ipykernel install --user --name ece285-vae --display-name "Python (ECE285 VAE)"
    python -m jupyter lab hw1_vae/HW1_VAE.ipynb
    ```

    Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
    for a matching CUDA-enabled `torch` and `torchvision` pair. If your current
    environment has CPU-only PyTorch, replace that pair first; an ordinary
    install can leave an already satisfied CPU package in place. Restart the
    notebook kernel after replacing PyTorch. A separate CUDA toolkit is not
    required for ordinary prebuilt-wheel training; the GPU driver must work.
    In WSL, follow [NVIDIA's WSL driver instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

    If you have only this notebook, install Jupyter with `python -m pip install
    jupyterlab ipykernel`, open it, then use the dependency cell below. Colab
    and Kaggle already provide the notebook interface and usually PyTorch.
    Do not replace their working PyTorch installation unnecessarily.

    References: [Jupyter installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html),
    [kernel registration](https://ipython.readthedocs.io/en/stable/install/kernel_install.html),
    [Codespaces GPU machine deprecation](https://github.blog/changelog/2025-08-01-upcoming-deprecation-of-gpu-machine-type-in-codespaces/).
    """)
    md("""
    ## 2. Configure this experiment

    Edit this cell before running later stages. Paths can point to a local disk,
    an attached Kaggle dataset, or a mounted persistent directory. An empty
    `DATA_DIR` means the teaching cells still run and data-dependent cells skip.

    The isolated `WORKSPACE` holds generated Python source and results. When a
    repository is detected it defaults to `runs/notebook_workspace`; elsewhere
    it is `ece285_vae_notebook` under the starting directory. Source cells below
    overwrite only their generated copies. To edit the model, change its source
    cell, then restart and rerun from the top so imports do not use stale code.

    Use one training switch at a time: `RUN_TRAINING` starts a **new** pilot;
    `RUN_RESUME` extends an existing run to `TOTAL_EPOCHS` total epochs. After a
    successful pilot, set `RUN_TRAINING=False` before enabling `RUN_RESUME`.
    """)
    code("""
    from pathlib import Path
    import os
    import sys
    import subprocess
    import json

    INSTALL_MISSING_DEPENDENCIES = False  # Set True on a new runtime if needed.
    WORKSPACE_OVERRIDE = None  # e.g. r"D:\\ECE285_notebook" or "/content/ece285_vae_notebook"
    DATA_DIR = None  # e.g. r"D:\\datasets\\chest_xray" or "/kaggle/input/chest-xray-pneumonia/chest_xray"
    DOWNLOAD_DATA = False
    RUN_TRAINING = False
    RUN_RESUME = False
    RUN_GENERATION = False
    RUN_EVALUATION = False  # First use downloads ~104 MB of Inception weights.
    EXPORT_RESULTS = False

    DEVICE_REQUEST = "auto"  # "auto", "cuda", or "cpu"; auto selects CUDA if visible.
    ALLOW_CPU_TRAINING = False  # Change deliberately for a CPU experiment.
    IMAGE_SIZE = 64
    LATENT_DIM = 128
    BASE_CHANNELS = 32
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    KL_WEIGHT = 0.001
    KL_WARMUP_EPOCHS = 0
    SEED = 42
    NUM_WORKERS = 0  # Portable baseline; try 2–4 on Linux after the pilot works.
    CPU_THREADS = 2
    PILOT_EPOCHS = 2
    TOTAL_EPOCHS = 50
    RUN_NAME = "baseline64"
    RUN_DIR_OVERRIDE = None  # Prefer persistent storage for Colab/remote jobs.
    NUM_GENERATED = 64
    EVAL_NUM_SAMPLES = 0  # 0 = all official test images, matched generated count.

    # Preserve the starting directory when this cell is rerun in one kernel.
    if "_ECE285_INITIAL_CWD" not in globals():
        _ECE285_INITIAL_CWD = Path.cwd().resolve()
    repository = next((p for p in (_ECE285_INITIAL_CWD, *_ECE285_INITIAL_CWD.parents)
                       if (p / "hw1_vae" / "model.py").is_file()
                       and (p / ".git").is_dir()), None)
    default_workspace = (repository / "runs" / "notebook_workspace" if repository
                         else _ECE285_INITIAL_CWD / "ece285_vae_notebook")
    WORKSPACE = Path(WORKSPACE_OVERRIDE).expanduser().resolve() if WORKSPACE_OVERRIDE else default_workspace
    if repository and WORKSPACE.resolve() == repository.resolve():
        raise ValueError("Choose a separate notebook workspace, not the repository root.")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "hw1_vae").mkdir(exist_ok=True)
    os.chdir(WORKSPACE)
    RUN_DIR = (Path(RUN_DIR_OVERRIDE).expanduser().resolve() if RUN_DIR_OVERRIDE
               else WORKSPACE / "runs" / RUN_NAME)
    if DATA_DIR is not None:
        DATA_DIR = Path(DATA_DIR).expanduser()
        if not DATA_DIR.is_absolute():
            DATA_DIR = (_ECE285_INITIAL_CWD / DATA_DIR).resolve()
    if RUN_TRAINING and RUN_RESUME:
        raise ValueError("Enable either a fresh pilot or resume, one at a time.")
    print("Kernel Python:", sys.executable)
    print("Generated source/workspace:", WORKSPACE)
    print("Experiment outputs:", RUN_DIR)
    """, ("parameters",))
    md("""
    The next cell installs only missing/too-old dependencies when the switch is
    enabled, using **this kernel's** Python. It does not select a CUDA wheel or
    install drivers. If packages were already imported when you upgraded them,
    restart the kernel before continuing.
    """)
    dependency_specs = [line.strip() for line in (here / "requirements.txt").read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
    code(f"DEPENDENCY_SPECS = {dependency_specs!r}\n" + textwrap.dedent("""
    import importlib.metadata
    from packaging.requirements import Requirement

    missing = []
    for spec in DEPENDENCY_SPECS:
        requirement = Requirement(spec)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or not requirement.specifier.contains(installed, prereleases=True):
            missing.append(spec)
    if missing and INSTALL_MISSING_DEPENDENCIES:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    elif missing:
        raise RuntimeError("Missing/old packages: " + ", ".join(missing)
                           + ". Install the correct PyTorch build first if needed, then set "
                           "INSTALL_MISSING_DEPENDENCIES=True and rerun this cell.")
    else:
        print("All VAE dependencies meet the required versions.")

    def run_module(module, *arguments):
        # Run the displayed package with this kernel's Python.
        command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
        print("Running:", subprocess.list2cmdline(command), flush=True)
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = str(CPU_THREADS)
        environment["MKL_NUM_THREADS"] = str(CPU_THREADS)
        environment["PYTHONPATH"] = str(WORKSPACE)
        # Stream output in Jupyter rather than buffering a long training run.
        process = subprocess.Popen(command, cwd=WORKSPACE, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    """))
    md("""
    ## 3. Check the actual compute device

    These source cells create the small package used by the rest of the
    notebook. `%%writefile` writes the visible code to the isolated workspace;
    it does not train or change your repository implementation.

    The diagnostic separates four questions: is NVIDIA hardware exposed, does
    a driver respond, was PyTorch built with CUDA, and can a real CUDA operation
    plus VAE backpropagation execute? Seeing an RTX 4080 in your laptop's Task
    Manager does not imply a remote kernel can use it.
    """)
    module("__init__.py")
    module("model.py")
    module("check_compute.py")
    code("""
    # Prefer the generated source even if another hw1_vae was imported earlier.
    for name in tuple(sys.modules):
        if name == "hw1_vae" or name.startswith("hw1_vae."):
            del sys.modules[name]
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    import importlib
    importlib.invalidate_caches()
    from hw1_vae.check_compute import collect_report
    from IPython.display import display, JSON, Image as DisplayImage, Markdown

    compute_report = collect_report(test_vae=True)
    display(JSON(compute_report))
    (WORKSPACE / "compute_report.json").write_text(json.dumps(compute_report, indent=2) + "\\n")
    print(compute_report["conclusion"])
    """, ("compute-check",))
    md(r"""
    ## 4. Understand the encoder, Gaussian sample, and decoder

    The complete `ConvVAE` is visible in the `model.py` cell above. Its input
    is a batch $x$ of grayscale images with shape $[B,1,64,64]$.

    | Encoder operation | Default output shape |
    | --- | --- |
    | Conv2d(1,32,4,2,1), LeakyReLU | $[B,32,32,32]$ |
    | Conv2d(32,64,4,2,1), LeakyReLU | $[B,64,16,16]$ |
    | Conv2d(64,128,4,2,1), LeakyReLU | $[B,128,8,8]$ |
    | Conv2d(128,256,4,2,1), LeakyReLU | $[B,256,4,4]$ |
    | Flatten, then two independent linear heads | $\mu,\log\sigma^2 \in \mathbb R^{B\times128}$ |

    Stride two halves the spatial dimensions. More channels summarize more
    features. The two heads describe a **distribution for each image**:
    $q_\phi(z|x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x)))$.
    They are not image-batch statistics or diagnostic labels.

    To sample and retain gradients, use reparameterization:
    $$\epsilon\sim\mathcal N(0,I),\qquad
      z=\mu+\exp(\tfrac12\log\sigma^2)\odot\epsilon.$$
    The factor one-half converts log-variance into log-standard-deviation.
    Backpropagation can then reach both Gaussian heads through $z$.

    The decoder linearly expands $z$ into $256\times4\times4$, then applies
    four transposed convolutions to double spatial size at each stage. Its
    sigmoid produces $[B,1,64,64]$ values in $[0,1]$. We display its predicted
    image mean rather than adding observation noise.

    This is a smaller grayscale adaptation of the lecture example. The lecture
    around 65–66 minutes uses RGB, five encoder stages, and latent size 1024.
    Our four-stage mirror also avoids the apparent missing upsampling stage
    on that displayed decoder slide.
    """)
    module("data.py")
    module("utils.py")
    code("""
    import torch
    import numpy as np
    from PIL import Image
    from hw1_vae.model import ConvVAE, vae_loss
    from hw1_vae.data import ChestXrayDataset, resolve_data_root
    from hw1_vae.utils import resolve_device, seed_everything, save_grid, load_model

    torch.set_num_threads(CPU_THREADS)
    seed_everything(SEED)
    DEVICE = resolve_device(DEVICE_REQUEST)
    print("Selected device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CPU teaching checks are enabled. Real CPU training requires ALLOW_CPU_TRAINING=True.")

    demo_model = ConvVAE(IMAGE_SIZE, LATENT_DIM, BASE_CHANNELS).to(DEVICE)
    demo_x = torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    demo_reconstruction, demo_mu, demo_logvar = demo_model(demo_x)
    print("Input:", tuple(demo_x.shape))
    print("Encoded Gaussian heads:", tuple(demo_mu.shape), tuple(demo_logvar.shape))
    print("Reconstruction:", tuple(demo_reconstruction.shape))
    assert demo_reconstruction.shape == demo_x.shape
    print("Parameters:", f"{sum(p.numel() for p in demo_model.parameters()):,}")
    """)
    md(r"""
    ## 5. Derive and execute the training objective

    Training balances image reconstruction against latent regularization:

    $$R=\frac{1}{BD}\sum_{b=1}^B\|\hat x_b-x_b\|_2^2,\qquad
    K=-\frac{1}{2B}\sum_{b,j}
    (1+\log\sigma_{bj}^2-\mu_{bj}^2-\exp(\log\sigma_{bj}^2)),\qquad
    L=R+\lambda K.$$

    Here $D=64\times64$ pixels by default. The reconstruction term averages
    over batch and pixels; KL sums over latent coordinates and averages over
    images. $K=0$ for $\mu=0,\log\sigma^2=0$, the standard-normal prior.

    MSE is motivated by a fixed-variance Gaussian observation model. The
    practical lecture uses mean MSE plus `0.001 * KL`, which we retain as the
    starting convention. State that exact weighted objective in your report;
    its numeric value is not a unit-weight negative ELBO. Replacing mean MSE
    by pixel-summed MSE multiplies that term by 4096 at this resolution and
    changes the balance unless the KL weight is adjusted.

    The following cell performs **one update on random synthetic pixels** to
    show the exact mechanics. It discards this model afterward. The real
    training command creates its own freshly initialized model or resumes a
    checkpoint; it never uses this demonstration as a trained result.
    """)
    code("""
    demo_optimizer = torch.optim.Adam(demo_model.parameters(), lr=LEARNING_RATE)
    demo_optimizer.zero_grad(set_to_none=True)
    losses = vae_loss(demo_reconstruction, demo_x, demo_mu, demo_logvar, KL_WEIGHT)
    losses["total"].backward()
    for name, head in (("mu", demo_model.fc_mu), ("logvar", demo_model.fc_logvar)):
        print(name, "head gradient norm:", head.weight.grad.norm().item())
        assert torch.isfinite(head.weight.grad).all()
    demo_optimizer.step()
    print("SYNTHETIC demonstration losses:", {k: v.item() for k, v in losses.items()})

    zeros = torch.zeros(2, LATENT_DIM, device=DEVICE)
    prior_check = vae_loss(demo_x, demo_x, zeros, zeros)
    assert abs(prior_check["kl"].item()) < 1e-7
    print("Analytic KL at the prior:", prior_check["kl"].item())
    del demo_optimizer, demo_model, demo_x, demo_reconstruction, demo_mu, demo_logvar, losses
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    """)
    md("""
    ## 6. Prepare and inspect the assigned data

    The loader includes **both** NORMAL and PNEUMONIA images. This is an
    unconditional generator; the diagnostic labels are not training targets.
    Each image becomes a grayscale tensor, bilinear-resized to the configured
    square size, with uint8 intensity scaled to `[0,1]`.

    Keep the official folders separate: `train` updates weights, `val` chooses
    checkpoints, and `test` is reserved for final evaluation. The official
    validation folder is tiny, so its loss is noisy. Preserving these folders
    does not by itself establish patient-level independence or clinical validity.

    Set `DATA_DIR` to the directory containing `train/NORMAL`,
    `train/PNEUMONIA`, and matching `val`/`test` folders. Common extra
    `chest_xray` wrappers and hidden Apple metadata files are handled. Or set
    `DOWNLOAD_DATA=True` to call Kagglehub for the exact assigned dataset.
    That switch uses network and disk and may require Kaggle authentication.
    Do not paste credentials into a notebook that you will share.

    On Kaggle, attach the dataset and point `DATA_DIR` at its mounted folder.
    On Colab, download into the runtime or mount your own storage first. Runtime
    disks can disappear, so keep run/checkpoint copies in persistent storage.
    """)
    code("""
    if DOWNLOAD_DATA:
        import kagglehub
        DATA_DIR = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))

    datasets = {}
    if DATA_DIR is None:
        print("Dataset stage skipped. Set DATA_DIR or DOWNLOAD_DATA=True when ready.")
    else:
        DATA_DIR = resolve_data_root(DATA_DIR)
        datasets = {split: ChestXrayDataset(DATA_DIR, split, IMAGE_SIZE)
                    for split in ("train", "val", "test")}
        print("Resolved data root:", DATA_DIR)
        for split, dataset in datasets.items():
            print(split, "total:", len(dataset), "classes:", dataset.class_counts)
        first_images = torch.stack([datasets["train"][i]
                                    for i in range(min(8, len(datasets["train"])))])
        input_grid = WORKSPACE / "input_preview.png"
        save_grid(first_images, input_grid, nrow=8)
        display(DisplayImage(filename=str(input_grid)))
        print("Real training inputs after preprocessing; not generated images.")
    """, ("dataset",))
    md("""
    ## 7. Inspect the complete training loop and run offline checks

    The visible `train.py` cell below contains the full loop. Each training
    batch zeroes gradients, samples a latent, reconstructs its inputs, computes
    the loss, backpropagates, and steps Adam. Validation uses no gradient updates
    and repeats the same latent noise each epoch to reduce sampling variation.

    Losses are averaged by the number of images, including a short last batch.
    `best.pt` is selected with validation objective at the target KL weight;
    `last.pt` also carries optimizer state and RNG state for resuming. Optional
    warmup changes the training KL weight while selection uses the final weight.
    Source cells do not execute their CLI `main()`; only later command cells do.
    """)
    module("train.py")
    module("generate.py")
    module("evaluate.py")
    module("smoke_test.py")
    code('run_module("hw1_vae.smoke_test")', ("offline-checks",))
    md("""
    ## 8. Start a timed two-epoch pilot

    The defaults are 64×64 grayscale, channels `[32,64,128,256]`, latent size
    128, batch size 64, Adam learning rate 0.001, and KL weight 0.001. These are
    starting choices. Higher resolution/width costs memory; increasing KL weight
    pushes posteriors toward the prior more strongly. No augmentation or class
    rebalancing is applied, so the empirical training mixture is retained.

    Set `RUN_TRAINING=True` in the configuration cell, then rerun that cell and
    this cell after preparing the data. Two epochs measure actual throughput;
    do not infer GPU runtime from this Codespace's CPU checks. The printed epoch
    time covers training/validation; plots and checkpoint writes add overhead.

    Watch MSE, KL, and the fixed-latent image grids together. KL near zero with
    nearly identical samples may indicate posterior collapse. Improve settings
    using validation information, and reserve the test split for final scores.
    For out-of-memory failures, start a fresh run with a smaller batch size.
    """)
    code("""
    def require_training_setup():
        if not datasets:
            raise RuntimeError("Set DATA_DIR or download and rerun the dataset cell first.")
        if DEVICE.type != "cuda" and not ALLOW_CPU_TRAINING:
            raise RuntimeError("No CUDA kernel is selected. Move to your GPU runtime or explicitly "
                               "set ALLOW_CPU_TRAINING=True for CPU training.")
        if DEVICE.type == "cuda" and not compute_report["cuda_ready"]:
            raise RuntimeError("CUDA compute test failed; inspect compute_report before training.")

    if RUN_TRAINING:
        require_training_setup()
        run_module("hw1_vae.train", "--data-dir", DATA_DIR, "--output-dir", RUN_DIR,
                   "--device", DEVICE, "--epochs", PILOT_EPOCHS, "--batch-size", BATCH_SIZE,
                   "--image-size", IMAGE_SIZE, "--latent-dim", LATENT_DIM,
                   "--base-channels", BASE_CHANNELS, "--learning-rate", LEARNING_RATE,
                   "--kl-weight", KL_WEIGHT, "--kl-warmup-epochs", KL_WARMUP_EPOCHS,
                   "--seed", SEED, "--num-workers", NUM_WORKERS)
    else:
        print("Fresh training skipped. Enable RUN_TRAINING after choosing data and compute.")
    """, ("training",))
    md("""
    ## 9. Resume on the same or another machine

    Set `RUN_TRAINING=False`, `RUN_RESUME=True`, then rerun configuration and this
    cell. `TOTAL_EPOCHS=50` means 50 epochs **in total**, including the pilot.
    Existing checkpoints are not overwritten by a fresh run; use a new
    `RUN_NAME` for a different experiment.

    To move compute resources, copy the **whole run directory**, including
    `last.pt` and `best.pt`, then change `RUN_DIR_OVERRIDE` and `DATA_DIR` to their
    new absolute locations. The data manifest uses portable relative paths.
    The same filenames/sizes/splits must remain available. Dataset manifests
    are not content hashes and do not detect every image-content change.

    Resume restores the saved architecture, optimizer, batch size, learning rate,
    seed, and KL schedule. Changing those notebook variables does not override
    an existing checkpoint. Start a new run for hyperparameter changes.
    A CPU↔GPU or hardware/library change can change the stochastic trajectory;
    portability does not mean bitwise equality between machines. Keep recorded
    environment information when comparing runs.
    """)
    code("""
    if RUN_RESUME:
        require_training_setup()
        run_module("hw1_vae.train", "--data-dir", DATA_DIR, "--output-dir", RUN_DIR,
                   "--device", DEVICE, "--epochs", TOTAL_EPOCHS,
                   "--resume", RUN_DIR / "last.pt", "--num-workers", NUM_WORKERS)
    else:
        print("Resume skipped. Enable RUN_RESUME after the pilot, or after copying a run here.")
    """, ("resume",))
    md("""
    ## 10. Read curves and inspect reconstructions

    `history.csv` records train/validation total, MSE, KL, KL weights, and time.
    `loss_curves.png` shows three panels. The total objective depends on the KL
    weight, so a warmup run's training total is not directly comparable across
    weights without considering its components.

    `reconstructions.png` pairs each original with a deterministic reconstruction
    made from its posterior mean, `decode(mu)`. This differs from the sampled
    reconstruction used in the training/validation objective. The per-epoch
    `previews/samples_###.png` grids use the same prior latents each epoch.
    """)
    code("""
    if (RUN_DIR / "history.csv").is_file():
        import csv
        with (RUN_DIR / "history.csv").open(newline="") as handle:
            history_rows = list(csv.DictReader(handle))
        print("Latest epoch:", history_rows[-1])
        print("Settings:", json.loads((RUN_DIR / "config.json").read_text()))
        for name in ("loss_curves.png", "reconstructions.png"):
            display(Markdown(f"**{name}**"))
            display(DisplayImage(filename=str(RUN_DIR / name)))
        prior_previews = sorted((RUN_DIR / "previews").glob("samples_*.png"))
        if prior_previews:
            display(Markdown("**Latest fixed-latent prior samples**"))
            display(DisplayImage(filename=str(prior_previews[-1])))
    else:
        print("No real training results yet. Curves and images appear after your pilot.")
    """)
    md(r"""
    ## 11. Generate genuinely new images

    To generate, draw $z\sim\mathcal N(0,I)$ and call `decode(z)` directly.
    The encoder and a real input image are not involved. These fresh prior
    samples are the generated examples required for the homework; reconstructions
    must be labeled separately. Set `RUN_GENERATION=True` when `best.pt` exists.

    The command saves individual PNGs, a grid, latent vectors, seed, and checkpoint
    metadata. It uses a unique output folder for each invocation to keep runs
    with different sample counts separate. Intensities use a fixed linear
    `[0,1]`→`[0,255]` mapping with no per-image contrast normalization.
    """)
    code("""
    if RUN_GENERATION:
        from datetime import datetime, timezone
        generation_dir = RUN_DIR / "generated" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_module("hw1_vae.generate", "--checkpoint", RUN_DIR / "best.pt",
                   "--output-dir", generation_dir, "--num-samples", NUM_GENERATED,
                   "--batch-size", BATCH_SIZE, "--device", DEVICE, "--seed", SEED + 1)
        display(DisplayImage(filename=str(generation_dir / "samples_grid.png")))
    else:
        print("Generation skipped. Enable RUN_GENERATION after a checkpoint exists.")
    """, ("generation",))
    md(r"""
    ## 12. Calculate the requested FID and Inception Score

    **FID** compares feature means/covariances between real and generated images:
    $\|m_r-m_g\|^2+\operatorname{Tr}(C_r+C_g-2(C_rC_g)^{1/2})$.
    Lower is conventionally better. **IS** is
    $\exp(\mathbb E_x\operatorname{KL}(p(y|x)\|p(y)))$:
    it rewards confident predictions for each image and variety across images.
    Higher is conventionally better; IS does not directly compare against real data.

    Following the homework's
    [Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/),
    this implementation explicitly uses **1000 ImageNet class probabilities**
    for FID, not the common 2048-dimensional pooled-feature variant. It corrects
    the tutorial's closing direction typo and documents its representation.
    Do not compare our values with a different feature definition.

    Both real and generated images start at model resolution in `[0,1]`, repeat
    grayscale to RGB, resize to 299×299 with antialiased bilinear interpolation,
    and receive identical ImageNet normalization. The frozen torchvision
    Inception model returns class probabilities; Ignite computes the metrics.
    Each test reference is matched by a fresh prior sample. We do not score
    reconstructions or select a checkpoint using test FID.

    Set `RUN_EVALUATION=True` after choosing the final checkpoint. First use
    downloads about 104 MB of pretrained weights. `EVAL_NUM_SAMPLES=0` uses
    the whole test split; a positive value selects that many without replacement.
    The code saves score values **and** the complete evaluation protocol.

    The ImageNet classifier was trained on natural-image categories. Its scores
    do not demonstrate medical validity, and a small test set produces a
    sample-sensitive, rank-deficient covariance estimate. Report sample count,
    seed, feature definition, preprocessing, and limitations beside the scores.
    """)
    code("""
    if RUN_EVALUATION:
        if not datasets:
            raise RuntimeError("Prepare the real dataset before final evaluation.")
        evaluation_dir = RUN_DIR / "evaluation"
        run_module("hw1_vae.evaluate", "--checkpoint", RUN_DIR / "best.pt",
                   "--data-dir", DATA_DIR, "--output-dir", evaluation_dir,
                   "--num-samples", EVAL_NUM_SAMPLES, "--batch-size", min(BATCH_SIZE, 32),
                   "--device", DEVICE, "--seed", SEED, "--num-workers", NUM_WORKERS)
        display(JSON(json.loads((evaluation_dir / "metrics.json").read_text())))
    else:
        print("Final metrics skipped. No score is claimed before training/evaluation.")
    """, ("evaluation",))
    md("""
    ## 13. Save your work and prepare the report

    Keep the notebook, generated source, checkpoint pair, configuration,
    environment information, loss curves, fresh samples, and metrics protocol.
    The export cell packages only the selected run and implementation; it omits
    the dataset and virtual environment. Download/save the ZIP before a cloud
    runtime disconnects, or use persistent storage for `RUN_DIR` from the start.

    A notebook edit changes its embedded generated source. The repository CLI
    is a separate copy; export the source actually used for the submitted run.
    The report outline below is a plan with placeholders, not measured results.
    """)
    portable_outline = (here / "REPORT_OUTLINE.md").read_text().replace(
        "[handout](../for_chat_gpt/mv47c1gud5n1ka.pdf)", "Homework 1 handout")
    add("code", "%%writefile REPORT_OUTLINE.md\n" + portable_outline,
        ("embedded-doc",))
    md(portable_outline)
    code("""
    if EXPORT_RESULTS:
        if not (RUN_DIR / "last.pt").is_file():
            raise RuntimeError("No training checkpoint exists to export yet.")
        import zipfile
        archive = WORKSPACE / f"{RUN_DIR.name}_results.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for source in sorted((WORKSPACE / "hw1_vae").glob("*.py")):
                bundle.write(source, f"source/hw1_vae/{source.name}")
            bundle.writestr("source/requirements.txt", "\\n".join(DEPENDENCY_SPECS) + "\\n")
            bundle.write(WORKSPACE / "REPORT_OUTLINE.md", "REPORT_OUTLINE.md")
            bundle.write(WORKSPACE / "compute_report.json", "compute_report.json")
            for artifact in sorted(RUN_DIR.rglob("*")):
                if (artifact.is_file() and artifact.suffix != ".tmp"
                        and artifact.resolve() != archive.resolve()):
                    bundle.write(artifact, "run/" + artifact.relative_to(RUN_DIR).as_posix())
        print("Saved:", archive, "bytes:", archive.stat().st_size)
        print("Also save/download this notebook using your notebook application's File menu.")
        try:
            import google.colab
        except ImportError:
            print("Use your Jupyter file browser or operating system's file manager to "
                  "save/download the ZIP at the printed absolute path.")
        else:
            from google.colab import files
            files.download(str(archive))
    else:
        print("Export skipped. Enable EXPORT_RESULTS after training, before leaving the runtime.")
    """, ("export",))
    md("""
    ## Check your understanding

    1. Why does the encoder output a mean and a log-variance for every image?
    2. Where does randomness enter, and how does the gradient reach both heads?
    3. What does zero KL mean, and why can minimizing KL alone discard information?
    4. Why can reconstructions look better than fresh prior samples?
    5. How would changing pixel-mean MSE to a sum affect the KL balance?
    6. Which data split selects the checkpoint, and which one supplies final FID references?
    7. What can ImageNet-based FID and IS tell you about chest X-ray anatomy?

    For lecture review, revisit roughly 54–59 minutes for reparameterization
    and 63–69 minutes for implementation. Record observations from your real
    run before drawing conclusions about image quality.
    """)

    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12"},
                    "colab": {"name": "HW1_VAE.ipynb", "provenance": []},
                    "ece285": {"self_contained": True, "builder": "hw1_vae/build_notebook.py"}}}
    target = here / "HW1_VAE.ipynb"
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {target} ({len(cells)} cells, {target.stat().st_size:,} bytes)")
    return target


if __name__ == "__main__":
    build_notebook()
