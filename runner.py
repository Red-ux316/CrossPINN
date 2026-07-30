# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: CrossPINN
#     language: python
#     name: python3
# ---

# %% [markdown] id="header"
# # CrossPINN: Benchmark Execution Runner
# Notebook for tuning, training, and evaluating CrossPINN (C-PINN) and baseline models across Volterra integro-differential benchmark equations.

# %% colab={"base_uri": "https://localhost:8080/"} id="setup"
import os
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown] id="master_benchmark_runner_header"
# # ==============================================================================
# # MASTER BENCHMARK SUITE: RUN ALL PROBLEMS (TUNE + TRAIN + EVALUATE)
# # Runs MLP, RISN, C-PINN (Global Skip True), and C-PINN (Global Skip False)
# # across active benchmark problems: 1, 19, 22, 24, 25, 34
# # ==============================================================================

# %% colab={"base_uri": "https://localhost:8080/"} id="master_benchmark_runner_code"
from config.study import Study
from tools.visualize.plot_comparison import process_problem
from config.path_manager import PathManager

benchmark_suite = [
    # (prob_id, num_quadrature_points)
    (1, 10),
    (19, 10),
    (22, 10),
    (24, 10),
    (25, 10),
    (34, 5),
]

print("=== [MASTER SUITE] STARTING FULL BENCHMARK RE-RUN FOR ALL PROBLEMS ===")

for prob_id, num_quad in benchmark_suite:
    print(f"\n======================================================================")
    print(f"   STARTING FULL BENCHMARK SUITE FOR PROBLEM {prob_id}")
    print(f"======================================================================")

    # 1. MLP Baseline (20 trials)
    print(f"\n---> [Prob {prob_id}] 1/4: MLP Baseline...")
    try:
        s_mlp = Study(prob_id=prob_id, model="mlp", study_name=f"MLP_prob{prob_id}", num_quadrature_points=num_quad)
        s_mlp.tune(n_trials=20)
        s_mlp.train(reset=True)
        s_mlp.evaluate(device=device)
    except Exception as e:
        print(f"⚠️ Error running MLP for Problem {prob_id}: {e}")

    # 2. RISN Baseline (20 trials)
    print(f"\n---> [Prob {prob_id}] 2/4: RISN Baseline...")
    try:
        s_risn = Study(prob_id=prob_id, model="risn", study_name=f"RISN_prob{prob_id}", num_quadrature_points=num_quad)
        s_risn.tune(n_trials=20)
        s_risn.train(reset=True)
        s_risn.evaluate(device=device)
    except Exception as e:
        print(f"⚠️ Error running RISN for Problem {prob_id}: {e}")

    # 3. C-PINN Full (Global Skip True) (50 trials)
    print(f"\n---> [Prob {prob_id}] 3/4: C-PINN (Global Skip = True)...")
    try:
        s_tlp_skip = Study(prob_id=prob_id, model="tlp", study_name=f"TLP_posEnc_GlobalSkip_prob{prob_id}", global_skip=True, num_quadrature_points=num_quad)
        s_tlp_skip.tune(n_trials=50)
        s_tlp_skip.train(reset=True)
        s_tlp_skip.evaluate(device=device)
    except Exception as e:
        print(f"⚠️ Error running C-PINN (Skip True) for Problem {prob_id}: {e}")

    # 4. C-PINN (Global Skip False) (50 trials)
    print(f"\n---> [Prob {prob_id}] 4/4: C-PINN (Global Skip = False)...")
    try:
        s_tlp_noskip = Study(prob_id=prob_id, model="tlp", study_name=f"TLP_posEnc_noWarmStart_prob{prob_id}", global_skip=False, num_quadrature_points=num_quad)
        s_tlp_noskip.tune(n_trials=50)
        s_tlp_noskip.train(reset=True)
        s_tlp_noskip.evaluate(device=device)
    except Exception as e:
        print(f"⚠️ Error running C-PINN (Skip False) for Problem {prob_id}: {e}")

    # Generate comparison plots and PDF report for this problem
    try:
        pm = PathManager(prob_id=prob_id)
        process_problem(f"prob{prob_id}", device=device, data_dir=pm.data_root, generate_plots=True)
    except Exception as e:
        print(f"⚠️ Error generating comparison report for Problem {prob_id}: {e}")

print("\n🎉 === [MASTER SUITE] ALL BENCHMARKS COMPLETED SUCCESSFULLY! ===")


# %% [markdown] id="prob1_header"
# # ==============================================================================
# # 1. PROBLEM 1: Volterra 1D Linear
# # ==============================================================================

# %% [markdown] id="p1_mlp"
# ### 1.1 Problem 1 - MLP Baseline (20 Trials)
# %%
from config.study import Study
s = Study(prob_id=1, model="mlp", study_name="MLP_prob1", num_quadrature_points=10)
s.tune(n_trials=20)
s.train(reset=True)
s.evaluate(device=device)

# %% [markdown] id="p1_tlp_skip_true"
# ### 1.2 Problem 1 - C-PINN Global Skip True (50 Trials)
# %%
from config.study import Study
s = Study(prob_id=1, model="tlp", study_name="TLP_posEnc_GlobalSkip_prob1", global_skip=True, num_quadrature_points=10)
s.tune(n_trials=50)
s.train(reset=True)
s.evaluate(device=device)

# %% [markdown] id="p1_risn"
# ### 1.3 Problem 1 - RISN Baseline (20 Trials)
# %%
from config.study import Study
s = Study(prob_id=1, model="risn", study_name="RISN_prob1", num_quadrature_points=10)
s.tune(n_trials=20)
s.train(reset=True)
s.evaluate(device=device)

# %% [markdown] id="p1_tlp_skip_false"
# ### 1.4 Problem 1 - C-PINN Global Skip False (50 Trials)
# %%
from config.study import Study
s = Study(prob_id=1, model="tlp", study_name="TLP_posEnc_noWarmStart_prob1", global_skip=False, num_quadrature_points=10)
s.tune(n_trials=50)
s.train(reset=True)
s.evaluate(device=device)
