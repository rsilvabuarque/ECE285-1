# Homework 1 report plan

Use this outline to prepare the **two-page NeurIPS-format report** requested in the [handout](../for_chat_gpt/mv47c1gud5n1ka.pdf). Fill bracketed fields from the completed run. No training results have been supplied in this outline.

Use the course's template if one is provided. Otherwise, the [official NeurIPS 2026 page](https://neurips.cc/Conferences/2026/CallForPapers) links the [LaTeX author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip). Apply the homework's two-page limit; conference submission limits do not replace the assignment's instructions.

## Suggested page 1: task, method, and experiment

**Title:** A Convolutional Variational Autoencoder for Chest X-Ray Generation

**Abstract, about 50 words.** State the task, the unconditional convolutional VAE, the dataset, and the main measured outcome. Complete the outcome after training.

**Data and method, about 200 words.** Identify the [assigned dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), the actual train/validation/test counts, and the use of both diagnostic categories without label conditioning. Describe grayscale conversion, `64×64` resizing, and `[0,1]` scaling. Explain the Gaussian encoder, standard-normal prior, reparameterization, and decoder. Give the channel progression and latent dimension.

Write the implemented objective explicitly:

```text
R = squared reconstruction error averaged over images and pixels
K = Gaussian KL summed over latent coordinates, averaged over images
L = R + lambda * K
```

State the actual `lambda` and any warmup schedule. Identify this as the weighted MSE-plus-KL objective used for training. Mention the fixed-variance Gaussian motivation for MSE without claiming that the recorded scalar is a fully specified, unit-weight negative ELBO.

**Settings table.** Copy the actual run settings rather than assuming the defaults were used.

| Setting | Actual value |
| --- | --- |
| Resolution / channels | `[ ]` |
| Encoder channels / latent dimension | `[ ]` |
| Optimizer / learning rate | `[ ]` |
| Batch size / epochs completed | `[ ]` |
| KL weight / warmup | `[ ]` |
| Seed / training device | `[ ]` |
| Training duration / selected epoch | `[ ]` |

**Training figure.** Include training and validation curves. Show reconstruction MSE and KL as well as total loss, either as compact subplots or clearly labeled traces. If warmup changes the weight, identify the schedule so changes in total training loss are interpretable. Checkpoint selection uses validation data; report how the checkpoint was chosen.

## Suggested page 2: results and interpretation

**Generated-image figure.** Include a readable grid of fresh samples from `z ~ N(0,I)`. Label the model checkpoint and seed. A small reconstruction grid can be useful if space permits; label originals and reconstructions clearly.

**Quantitative results.** Fill in actual values from the evaluation JSON:

| Metric | Value | Evaluation set |
| --- | --- | --- |
| FID, tutorial-compatible 1000-probability representation | `[ ]` | `[N]` held-out real and `[N]` generated images |
| Inception Score | `[ ]` | Same `[N]` generated images |

Describe the Inception weights, resizing, grayscale-to-RGB conversion, input normalization, sample count, and seed as recorded by the evaluator. Cite the [assigned Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/). State that this probability-feature FID variant differs from standard 2048-dimensional pooled-feature FID; only compare values computed using the same protocol.

**Discussion, about 150 words.** Explain what the curves and images actually show. Useful questions include: Is validation improving with training? Is KL close to zero? Are samples diverse? Are lung fields visible? Are reconstructions sharper than prior samples? Describe only observations supported by the saved outputs.

Discuss limitations relevant to the experiment: low image resolution, the small official validation split, sample-sensitive metric estimates, and the mismatch between an ImageNet classifier and chest X-ray semantics. A higher IS or lower FID alone does not establish anatomical or clinical validity.

**Closing sentence.** State the result achieved and one specific improvement suggested by the evidence.

Before submission, replace every placeholder, make figure labels readable at the final size, confirm the PDF has two pages, and include the runnable code. Follow any course-specific citation or AI-assistance disclosure instructions supplied separately.
