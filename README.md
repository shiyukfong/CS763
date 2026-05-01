# CS 763: Investigating Privacy Leakage in Neural Network Training: Neural Collapse and Neural Tangent Regime
 
Group Member: Shi-Yuk Fong, Peiyuan Zhang
This repository contains code and configs used for CS 763 project.


---

`jax_privacy/`: a copy / modified version of DeepMind's `jax_privacy` tailored for this course project, adapted from https://github.com/google-deepmind/jax_privacy. All experiments live under `jax_privacy/experiments/image_classification`.

Quick start
1. Open a terminal and change to the image classification experiment folder:

```
cd jax_privacy/experiments/image_classification
```

2. Run an experiment using the provided `run_experiment.py` entry point. Example (full fine-tune on CIFAR):

```
python run_experiment.py --config=configs/cifar10_wrn_40_4_eps1_finetune_full.py --jaxline_mode=train_eval_multithreaded
```

There is a sample bash wrapper for a layerwise WRN-40 finetune at:

```
jax_privacy/experiments/image_classification/run_wrn40_layerwise_finetuning.sh
```

Notes on configuration
The experiment behavior is controlled by a Python `config` file passed to `run_experiment.py`. The most relevant hyper-parameters you will commonly change are listed below:

- Augmult: `config.experiment_kwargs.config.data.augmult`
- Batch-size: `config.experiment_kwargs.config.training.batch_size.init_value`
- Learning-rate value: `config.experiment_kwargs.config.optimizer.lr.kwargs.value`
- Model definition: `config.experiment_kwargs.config.model`
- Noise multiplier sigma: `config.experiment_kwargs.config.training.dp.noise_multiplier`
- Number of updates: `config.experiment_kwargs.config.num_updates`
- Privacy budget (delta): `config.experiment_kwargs.config.dp.target_delta`
- Privacy budget (epsilon): `config.experiment_kwargs.config.dp.stop_training_at_epsilon`

Practical tips
- If you run into OOM errors, reduce `config.experiment_kwargs.config.training.batch_size.per_device_per_step`. The effective batch size per update is determined by `init_value` and the number of devices — keep those consistent.
- If `stop_training_at_epsilon` is set, training stops automatically when the target privacy budget is reached, and `num_updates` may be ignored.
- Use `auto_tune` in cfgs to calibrate noise multiplier for a target (epsilon, delta) and chosen batch settings.

---

`ntk/`: experiments and code related to Neural Tangent Kernel work for this project.

---

References
- Paper: "Unlocking High-Accuracy Differentially Private Image Classification through Scale" — https://arxiv.org/abs/2204.13650
- Original repository: https://github.com/google-deepmind/jax_privacy



