# GenerateVision: Vision Modeling & Image Generation from First Principles

## Abstract

This repository documents my independent, research-oriented study of **visual representation learning and image generation**, reconstructed **from first principles**.

The work traces the architectural evolution from:

> **pixel grids → linear models → convolutions → deep CNNs → residual learning → vision transformers → latent representations → diffusion models**

with every transition motivated by **explicit inductive biases, failure modes, and mathematical necessity**, not by framework abstractions or empirical shortcuts.

All core models are implemented **from scratch**, using PyTorch only for tensor operations and automatic differentiation.
No high-level vision APIs, pretrained backbones, or diffusion libraries are used.

---

## Motivation

Unlike language, images do not arrive as sequences of symbols — they are **dense spatial signals** with strong locality, symmetry, and scale assumptions.

Modern vision frameworks hide these assumptions behind layers like `Conv2d`, `BatchNorm`, and `U-Net`, which obscures:

* why convolution is necessary
* what spatial inductive bias actually means
* how representation hierarchies emerge
* why generative models fail silently when latent structure is wrong

While reconstructing vision models manually, a recurring pattern emerged:

> A vision model can converge numerically, generate plausible outputs, and still be **structurally incorrect** with respect to spatial reasoning or generative assumptions.

This repository exists to document those failures **rigorously**, using math, code, and controlled experiments — in the same spirit as *GenerateMore* for language.

---

## Scope of Work

This project covers **three tightly coupled dimensions**:

---

### A. From-Scratch Vision Architectures

I implemented the following model families directly from their original formulations:

* Image tensor representations and pixel-level operators
* Linear and MLP-based image models (baseline failures)
* 2D convolution layers with explicit backpropagation
* Shallow and deep CNNs (LeNet, VGG-style)
* Residual Networks (ResNet-style skip connections)
* Vision Transformers (ViT) with patch tokenization
* Autoencoders and Variational Autoencoders (VAEs)
* Denoising Diffusion Probabilistic Models (DDPMs)
* Latent Diffusion architectures (Stable Diffusion–style, small-scale)

Each implementation exposes:

* spatial inductive biases
* parameter sharing assumptions
* receptive field growth
* gradient flow across depth
* representation collapse and failure modes

Final implementations live in `src/`.

---

### B. Architectural Reasoning & Mathematical Validation

For every architectural transition, I explicitly documented:

* why the previous model class fails
* what assumption the next model introduces
* how that assumption appears mathematically

This reasoning is captured in:

* `notes/` — architectural analysis and evolution
* `math/` — handwritten derivations, spatial diagrams, and gradient flows

The notes are intended to be read **sequentially**, forming a coherent research narrative from pixels to diffusion.

---

### C. Controlled Experimental Studies

Using from-scratch implementations, I conducted focused experiments on:

* locality vs global reasoning in CNNs
* depth vs receptive field tradeoffs
* residual connections and gradient preservation
* patch size sensitivity in Vision Transformers
* latent dimensionality collapse in VAEs
* noise schedule sensitivity in diffusion models

Experiments are isolated in `experiments/` and designed to test **one modeling assumption at a time**.

---

## Repository Structure (How to Navigate)

```
notes/        → architectural assumptions, failures, motivations
math/         → derivations, spatial reasoning, diagrams
notebooks/    → exploratory implementations with full reasoning
src/          → clean, consolidated model implementations
experiments/  → controlled empirical studies
artifacts/    → trained checkpoints (excluded from version control)
```

**Recommended reading order:**

1. `notes/`
2. `math/`
3. `src/`
4. `experiments/`
5. `notebooks/`

---

## Key Experimental Findings

1. **MLPs fail on images due to missing spatial inductive bias**
2. **Convolutions succeed by enforcing locality, not depth**
3. **Residual connections preserve gradient geometry, not information**
4. **Patch tokenization weakens low-level structure**
5. **VAEs define the generative geometry of diffusion**
6. **Diffusion models fail silently when latent structure is incorrect**
7. **Sampling correctness matters more than visual quality**

---

## Implementation Principles

* No use of pretrained models
* No use of `torchvision.models`
* Explicit convolution and attention mechanics
* No hidden spatial assumptions
* Separation of:

  * model definitions
  * training pipelines
  * experimental probes
  * visualization logic

All implementations in `src/` represent **final, validated architectures**, not exploratory drafts.

---

## References

* LeCun et al., 1998 — *Gradient-Based Learning Applied to Document Recognition*
* Simonyan & Zisserman, 2014 — *Very Deep Convolutional Networks*
* He et al., 2016 — *Deep Residual Learning for Image Recognition*
* Dosovitskiy et al., 2020 — *An Image is Worth 16×16 Words*
* Kingma & Welling, 2013 — *Auto-Encoding Variational Bayes*
* Ho et al., 2020 — *Denoising Diffusion Probabilistic Models*
* Rombach et al., 2022 — *High-Resolution Image Synthesis with Latent Diffusion Models*

---

## Conclusion

> In vision modeling, **spatial assumptions matter more than parameter count**.

This repository demonstrates — through derivation, implementation, and controlled experimentation — that image models can appear successful while violating fundamental structural assumptions.

Understanding **why vision architectures evolved** is essential before attempting image generation.