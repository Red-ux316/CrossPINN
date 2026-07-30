# CrossPINN (C-PINN): Attention-Based PINN Solvers for Volterra Integro-Differential Equations

Official repository for **CrossPINN (C-PINN)**, a query-based cross-attention neural architecture designed specifically to solve complex Volterra Integro-Differential Equations (VIDEs) and non-local memory systems.

---

## 1. Overview & Architecture

Physics-Informed Neural Networks (PINNs) relying on pointwise Multi-Layer Perceptrons (MLPs) struggle with Volterra equations due to non-local spatial and temporal memory dependencies. **CrossPINN (C-PINN)** addresses this limitation by replacing quadratic self-attention encoders with a **learnable global latent basis** interrogated by spatial queries via Sequential Cross-Attention.

Key Features:
- **Cross-Attention Latent Basis**: Efficient query-based spatial encoding without $\mathcal{O}(N^2)$ memory overhead.
- **Global Residual Skip Connection**: Combines a baseline MLP prior with cross-attention residual corrections ($\hat{u} = \text{MLP} + \Delta u_{\text{Attn}}$) to stabilize optimization over stiff integral loss landscapes.
- **Exact Double Backpropagation**: PyTorch-native `Math SDPA` routing allows pristine 2nd-order PDE loss gradients ($\frac{\partial^2 u}{\partial x^2}$) through attention blocks.
- **Native Gaussian Quadrature**: Differentiable Gauss-Legendre integration natively integrated into the autograd graph for Volterra/Fredholm integrals.
- **Hybrid Adam + L-BFGS Optimization**: Dual-phase optimization using Adam with dynamic collocation points followed by L-BFGS fine-tuning for extreme precision.
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

To train **CrossPINN (TLP)** on Problem 1 (1D Volterra equation):
```bash
python tools/train.py --prob_id 1 --model tlp
```

To train baseline models:
```bash
python tools/train.py --prob_id 1 --model risn
python tools/train.py --prob_id 1 --model mlp
```

### Hyperparameter Tuning (Optuna)

```bash
python tools/tune.py --prob_id 1 --model tlp --n_trials 20
```

### Model Evaluation & Visualization

```bash
python tools/evaluate.py --prob_id 1 --model tlp
python tools/visualize/plot_comparison.py --prob_id 1
```

---

## 5. References & Citation

- **CrossPINN Paper**: *Attention-Based PINN Solvers for Volterra Integro-Differential Equations* (Report included in `documents/final_report/CrossPINN.tex`).
- **RISN Baseline**: [Physics-Informed Neural Networks for Integral and Integro-Differential Equations](https://arxiv.org/abs/2501.16370v3) (arXiv:2501.16370v3).
