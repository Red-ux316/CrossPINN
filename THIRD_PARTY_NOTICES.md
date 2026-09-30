# Third-party notices

## `losses/relobralo.py` (ReLoBRaLo loss balancing)

This file is adapted from the PyTorch implementation
[Khadrawi/ReLoBRaLo_PyTorch](https://github.com/Khadrawi/ReLoBRaLo_PyTorch), which implements the method of

> R. Bischof and M. Kraus, *Multi-Objective Loss Balancing for Physics-Informed Deep Learning*, arXiv:2110.09813 (2021). The paper is licensed CC BY 4.0. The authors' original (TensorFlow) implementation is at [rbischof/relative_balancing](https://github.com/rbischof/relative_balancing).

As of 2026-09-30 neither repository declares a software license. The MIT license of this project therefore does **not** cover `losses/relobralo.py`: all rights in the adapted code remain with its original author(s). The rest of the code base only calls this module through `core/trainer.py` and can run with `use_relobralo: false` in the training configuration. If you are the copyright holder and want the attribution changed or the file removed, please open an issue.
