## ST-STORM: What Makes It Specific (The information needed to guarantee reproducibility will be available soon).
### What the Other Two Do Not Do

ST-STORM is built on the idea that **appearance** (weather, textures, spectral signatures) is a **semantic modality in its own right**. Rather than seeking a single all-purpose representation, it aims for an explicit **factorization**:

### 1) Two Explicit Latent Spaces: Content vs Style
ST-STORM relies on two distinct streams:  
- a **content stream** (invariant), and  
- a **style stream** (appearance-sensitive).

The goal is to avoid the impossible compromise between being both **invariant** and **sensitive to appearance** within a single representation.

### 2) Architectural Disentanglement (Not Only Loss-Based)

ST-STORM enforces this separation **by design**:

- **U-Net / skip connections** capture structural information: **where / what**  
- **SPADE** injects style through affine modulation (scales/biases), i.e. **how it appears**, without having to alter the geometry

$$
h' = \alpha(m,t) \odot IN(h) + \delta(m,t)
$$

This inductive bias is crucial: **appearance is not treated as a residual that must survive invariance, but as a dedicated channel**.

### 3) “Stylistic Chaos”: Directed Appearance Perturbation

Instead of relying on generic augmentations, ST-STORM creates views whose main difference lies in **style** (through inter pseudo-domain transfer, style bank, replay, and spectral noise).

This produces an **appearance-oriented self-supervised signal**.

### 4) Semantic Learning of Style: Adversarial + Spectrum + Style-JEPA

Style is not learned through raw pixel reconstruction:

- **Adversarial PatchGAN** constrains the distribution of micro-textures (high-frequency details)  
- **FFT / SWD** anchor style to frequency-based and distributional invariants  
- **Style-JEPA** enforces style tokens to be predictable, and therefore stable and reusable, filtering out contingency
