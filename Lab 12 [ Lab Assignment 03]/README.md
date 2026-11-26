# CNN Feature Visualization, Parameter Audit & Critical Analysis

Deep Learning & Computer Vision, Practical Lab Assignment

A single Google Colab notebook covers every part of the assignment:

| Assignment part | Where in the notebook | Output file(s) |
|---|---|---|
| **Part I**: 5 custom 3×3 kernels, Stages A–D, **5 × 4 grid** | Sections 1.1 – 1.4 | `01_feature_grid_5x4.png`, `part1_stage_statistics.csv` |
| **Part II**: Model A vs Model B, manual parameter audit vs `model.summary()` | Sections 2.1 – 2.2 | screenshot the two `summary()` outputs, `part2_parameter_audit.csv` |
| **Part III, Q1**: ReLU blackout, Leaky ReLU, Sigmoid | Question 1 | `q1_relu_blackout_zoom.png`, `q1_activation_comparison.png` |
| **Part III, Q2**: Max vs Avg pooling, dark text on white | Question 2 | `q2_max_vs_avg_maps.png`, `q2_profile.png`, `q2_text_on_white.png`, `q2_pooling_metrics.csv` |
| **Part III, Q3**: % parameter increase, receptive field, savings | Question 3 | `q3_receptive_field_savings.csv` |
| **Part III, Q4**: Conv→ReLU→Pool vs Conv→Pool→ReLU | Question 4 | printed proof |
| **Part III, Q5**: filter collapse (symmetric init, dead ReLU) | Question 5 | `q5_filter_collapse.png`, `q5_filter_collapse.csv` |


---

## Input image

By default the notebook draws a **synthetic 256×256 building facade**: sky gradient, bright cornice, textured wall, a 3×4 grid of dark windows, a door and the ground. It matches the "high-contrast architectural subject" the assignment describes. Because its geometry is known exactly, the answers can name precise regions (for example "the top edge of every window = −1.78 → black after ReLU").

**To use your own photo:** set `USE_OWN_IMAGE = True` in section 1.1 and upload the file when Colab asks. The answers in Part III describe the default image, so if you switch, re-read your own grid and update the region names in the answers.

---

## Key design decisions (be ready to defend these in the viva)

- **Cross-correlation, not flipped convolution.** `tf.nn.conv2d` (like every DL framework) does not flip the kernel.
- **Reflect padding + `VALID`.** Keeps the 256×256 size without the fake bright edges that zero padding creates at the image border.
- **Kernel naming.** Each Sobel filter is named by the **edge orientation it detects**. The x-gradient kernel `[[-1,0,1],[-2,0,2],[-1,0,1]]` detects **vertical** edges; the y-gradient kernel `[[-1,-2,-1],[0,0,0],[1,2,1]]` detects **horizontal** edges. Textbooks swap the "X/Y" labels, so state your convention.
- **Consistent intensity scaling in the grid.** Within each row, Stage A uses `[-m, +m]` (mid-grey = 0), and Stages B, C and D share `[0, m]` (black = exactly 0). That way the ReLU blackout and the max-vs-avg brightness difference can be compared visually.
- **Model architecture (Part II).** `Input 32×32×3 → Conv(5) → Pool → Conv(10) → Pool → GAP → Dense(10)`, using `padding="same"`. Padding, pooling and GAP do not change the parameter count.

---

## Results at a glance

**Part II audit** (manual = Keras, both)

| Model | Conv1 | Conv2 | Dense | **Total** |
|---|---|---|---|---|
| A: 3×3, Max pool | (3·3·3+1)·5 = 140 | (3·3·5+1)·10 = 460 | (10+1)·10 = 110 | **710** |
| B: 5×5, Avg pool | (5·5·3+1)·5 = 380 | (5·5·5+1)·10 = 1260 | (10+1)·10 = 110 | **1750** |

**Short versions of the viva answers**
- **Q1:** ReLU blacks out the **top and left edges of every window** (and the left wall, the underside of the cornice and the sky gradient), because those are bright→dark transitions where the kernel gives about −1.8. Leaky ReLU keeps them at −0.18 (faint lines). Sigmoid maps everything into (0, 1), with flat areas at 0.5.
- **Q2:** Max pooling keeps 1–2 px edge lines at full strength; average pooling halves them (about 51 % of the peak). For dark text on white, max pooling erases the strokes (0 % after two pools). **Average** pooling is better of the two, and min pooling or inverting the image is best.
- **Q3:** **+146.48 %** total (+173.33 % conv-only). **2** stacked 3×3 layers equal one 5×5 receptive field, saving **7C² − C (≈ 28 %)**. Applied to Model B this gives 1750 → 1024 (**−726, 41.5 %**).
- **Q4:** **Identical**, because ReLU is monotonic, so max(ReLU(x)) = ReLU(max(x)). Pooling first means 4× fewer ReLU evaluations. This does **not** hold for average pooling.
- **Q5:** (1) **Symmetric/constant init** gives identical gradients and filters that stay identical forever. Fix: He-normal init. (2) **Dead ReLUs** from a high learning rate or negative bias give all-zero maps. Fix: Leaky ReLU, a lower learning rate, BatchNorm, zero bias init.

---
the values to quote.

