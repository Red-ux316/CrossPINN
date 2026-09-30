# CrossPINN (C-PINN): Attention-Based PINN Solvers for Volterra Integro-Differential Equations

Code for **CrossPINN (C-PINN)**, a query-based cross-attention neural architecture designed specifically to solve complex Volterra Integro-Differential Equations (VIDEs) and non-local memory systems.

---

## 1. Overview & Architecture

Physics-Informed Neural Networks (PINNs) relying on pointwise Multi-Layer Perceptrons (MLPs) struggle with Volterra equations due to non-local spatial and temporal memory dependencies. **CrossPINN (C-PINN)** addresses this limitation by replacing quadratic self-attention encoders with a **learnable global latent basis** interrogated by spatial queries via Sequential Cross-Attention.

Key Features:
- **Cross-Attention Latent Basis**: Efficient query-based spatial encoding without $\mathcal{O}(N^2)$ memory overhead.
- **Global Residual Skip Connection**: Combines a baseline MLP prior with cross-attention residual corrections ($\hat{u} = \text{MLP} + \Delta u_{\text{Attn}}$) to stabilize optimization over stiff integral loss landscapes.
- **Exact Double Backpropagation**: PyTorch-native `Math SDPA` routing allows exact 2nd-order PDE loss gradients ($\frac{\partial^2 u}{\partial x^2}$) through attention blocks.
- **Native Gaussian Quadrature**: Differentiable Gauss-Legendre integration natively integrated into the autograd graph for Volterra/Fredholm integrals.
- **Optimization**: AdamW with a warmup-cosine schedule, dynamic collocation resampling and early stopping (the default, used for all reported results). An optional hybrid AdamW → L-BFGS mode (`optimizer: Hybrid`) is implemented in the trainer but was not used for the reported results.
- **Multi-Objective Loss Balancing (ReLoBRaLo)**: Dynamic loss weighting across PDE, initial condition (IC), and boundary condition (BC) residuals.

---

## 2. Supported Models

- **`tlp.py` (CrossPINN / C-PINN)**: Flagship architecture featuring a learnable latent memory matrix, spatial query cross-attention decoder, global skip scaffolding, and optional warm-start prior initialization.
- **`risn.py` (Residual Integration Solver Network)**: Baseline skip-connection architecture from arXiv:2501.16370.
- **`transpinn.py` (TransPINN)**: Full support-point self-attention operator learning model.
- **`mlp.py` (Baseline MLP)**: Standard feed-forward baseline for benchmark comparisons.

---

## 3. Directory Structure

```
CrossPINN/
├── config/             # Path management, Study API, and YAML configs (config/yamls/)
├── core/               # Training loop engine (trainer.py) and LR schedulers (scheduler.py)
├── equations/          # VIDE, Abel, Fredholm, 1D/2D, and Benchmark equation definitions
├── losses/             # Volterra/Fredholm PDE loss handlers and Gaussian Quadrature rules
├── models/             # PyTorch model definitions (tlp.py, risn.py, transpinn.py, mlp.py)
├── tools/              # Command-line tools: train.py, tune.py, evaluate.py, plot_comparison.py
├── utils/              # Logging and loss landscape utilities
├── runner.ipynb        # Benchmark suite runner notebook
├── runner.py           # Jupytext twin script for runner.ipynb
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 4. Quick Start

### Installation

```bash
git clone https://github.com/Red-ux316/CrossPINN.git
cd CrossPINN
pip install -r requirements.txt
```

### Training a Model

Logging to Weights & Biases is on by default (`wandb login` required); the commands below disable it with `--no_wandb`.

A 5-epoch smoke test on CPU:
```bash
python tools/train.py --prob_id 1 --model tlp --debug --no_wandb
```

To train **CrossPINN (`tlp`)** on Problem 1 (1D Volterra equation):
```bash
python tools/train.py --prob_id 1 --model tlp --no_wandb
```
By default the `tlp` model warm-starts from a trained MLP checkpoint of the same problem; if none is found it prints a warning and falls back to a cold start. The benchmark suite in `runner.py` trains the MLP baseline first for this reason.

To train baseline models:
```bash
python tools/train.py --prob_id 1 --model risn --no_wandb
python tools/train.py --prob_id 1 --model mlp --no_wandb
```

### Hyperparameter Tuning (Optuna)

```bash
python tools/tune.py --prob_id 1 --model tlp --n_trials 20 --no_wandb
```

### Model Evaluation & Visualization

```bash
python tools/evaluate.py --prob_id 1 --model tlp
python tools/visualize/plot_comparison.py --prob_id 1
```

---

## 5. Results

Six benchmark problems with known exact solutions (method of manufactured solutions), test grid of 1000 points, metric: mean absolute error (MAE). Numbers are those of the accompanying project report.

| Report ID | `--prob_id` | Problem | MLP | RISN | C-PINN (no skip) | C-PINN (global skip) |
|---|---|---|---|---|---|---|
| A | 1 | 1D linear, 2nd kind | 2.23e-02 | 1.04e-03 | 4.21e-03 | **5.42e-04** |
| B | 22 | 1D IDE, 1st order | 5.42e-04 | 1.48e-04 | 3.74e-05 | **8.86e-06** |
| C | 24 | 1D non-linear IDE | 5.10e-05 | 4.56e-05 | 2.81e-05 | **3.37e-06** |
| D | 25 | 1D IDE, 2nd order | 1.56e-05 | 7.05e-06 | 4.41e-05 | **1.31e-06** |
| E | 34 | 2D viscoelastic PDE-IDE | 6.76e-04 | **1.37e-04** | 3.32e-03 | 5.37e-04 |
| F | 19 | 1D coupled IDE system | 2.97e-04 | 2.31e-04 | **7.35e-05** | 2.65e-04 |

C-PINN with the global skip connection is the best model on A-D; the version without skip is best on the coupled system F; RISN is best on the 2D problem E, where C-PINN does not improve on the baseline. The results come from the benchmark suite in `runner.py` (tuning followed by training); the report gives no seed statistics. Trained weights and visualization reports are shared in a [Google Drive folder](https://drive.google.com/drive/folders/1ooljfmzFwYySVHlPgAsX7_Weob1u-H_C?usp=sharing). The full report is not bundled in this repository.

To re-run the whole suite (tuning, training and evaluation for each problem and model), use `runner.py` / `runner.ipynb`.

## 6. Tests

```bash
pip install -r requirements.txt
pytest
```
The smoke test runs a 5-epoch debug training on CPU.

## 7. References & Citation

- **CrossPINN report**: *Attention-Based PINN Solvers for Volterra Integro-Differential Equations* (course project report, University of Padova, 2026; not bundled here).
- **RISN Baseline**: [Physics-Informed Neural Networks for Integral and Integro-Differential Equations](https://arxiv.org/abs/2501.16370v3) (arXiv:2501.16370v3).
