# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tuning a WRN-40-4 on CIFAR-10 with the last 12 residual units trainable (24 conv layers)."""

from jax_privacy.experiments import image_data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.experiments.image_classification.configs import wrn40_layerwise_filters
from jax_privacy.src import accounting
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import optimizer_config
import ml_collections
    

def get_config() -> ml_collections.ConfigDict:
    """Experiment config."""

    target_epsilon = 1.0
    target_delta = 1e-5
    fixed_updates = 50
    total_batch = 4096
    num_examples = 50000  # CIFAR-10 train

    nm = accounting.calibrate_noise_multiplier(
        target_epsilon=target_epsilon,
        batch_sizes=total_batch,
        num_steps=fixed_updates,
        num_examples=num_examples,
        target_delta=target_delta,
        dp_accountant_config=accounting.PldAccountantConfig(),
    )

    config = config_base.ExperimentConfig(
        num_updates=fixed_updates,
        optimizer=optimizer_config.sgd_config(
            lr=optimizer_config.constant_lr_config(0.75),
        ),
        model=config_base.ModelConfig(
            name='wideresnet',
            kwargs={
                'depth': 40,
                'width': 4,
            },
            restore=config_base.ModelRestoreConfig(
                path=config_base.MODEL_CKPT.WRN_40_4_IMAGENET32,
                params_key='params',
                network_state_key='network_state',
                layer_to_reset='wide_res_net/Softmax',
            ),
        ),
        training=experiment_config.TrainingConfig(
            batch_size=experiment_config.BatchSizeTrainConfig(
                total=total_batch,
                per_device_per_step=8,
            ),
            weight_decay=0.0,
            train_only_layer=wrn40_layerwise_filters.make_train_last_wrn40_layers_filter(26),
            dp=experiment_config.DPConfig(
                delta=target_delta,
                clipping_norm=1.0,
                stop_training_at_epsilon=None,  # target_epsilon,
                rescale_to_unit_norm=True,
                noise_multiplier=nm,
                auto_tune=None,
            ),
            logging=experiment_config.LoggingConfig(
                grad_clipping=True,
                grad_alignment=False,
                snr_global=True,  # signal-to-noise ratio across layers
                snr_per_layer=False,  # signal-to-noise ratio per layer
            ),
        ),
        averaging=experiment_config.AveragingConfig(ema_coefficient=0.95,),
        data_train=image_data.Cifar10Loader(
            config=image_data.Cifar10TrainValidConfig(
                preprocess_name='standardise',
            ),
            augmult_config=image_data.AugmultConfig(
                augmult=16,
                random_flip=True,
                random_crop=True,
                random_color=False,
            ),
        ),
        data_eval=image_data.Cifar10Loader(
            config=image_data.Cifar10TestConfig(
                preprocess_name='standardise',
            ),
        ),
        data_nc=image_data.Cifar10Loader(
            config=image_data.Cifar10TrainValidConfig(
                preprocess_name='standardise',
                class_balanced_num_per_class=250,
            ),
        ),
        nc_evaluation=experiment_config.NCEvaluationConfig(
            batch_size=100,
            max_num_batches=25,
            params_to_eval=('last', 'ema'),
            use_single_device=True,
            host0_only=True,
        ),
        evaluation=experiment_config.EvaluationConfig(
            batch_size=100,
        ),
    )

    return config_base.build_jaxline_config(
        experiment_config=config,
    )
