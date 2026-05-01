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

"""Jaxline experiment to define training and eval loops."""

import collections
from typing import Iterable, Iterator

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.experiments import image_data as data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.experiments.image_classification import forward
from jax_privacy.experiments.image_classification import models
from jax_privacy.src.training import experiment
from jax_privacy.src.training import metrics as metrics_module
import numpy as np


class Experiment(experiment.AbstractExperiment):
  """Jaxline experiment.

  This class controls the training and evaluation loop at a high-level.
  """

  def __init__(
      self,
      mode: str,
      init_rng: chex.PRNGKey,
      config: config_base.ExperimentConfig,
  ):
    """Initializes experiment.

    Args:
      mode: 'train' or 'eval'.
      init_rng: random number generation key for initialization.
      config: ConfigDict holding all hyper-parameters of the experiment.
    """
    # Unused since we rather rely on `config.random_seed`. The argument
    # `init_rng` is kept to conform to jaxline's expectation.
    del init_rng

    self.config = config

    self._forward_fn = forward.MultiClassForwardFn(
        net=hk.transform_with_state(self._model_fn))
    self._nc_eval_fn = None

    super().__init__(
        mode=mode,
        random_seed=self.config.random_seed,
        training_config=self.config.training,
        optimizer_config=self.config.optimizer,
        averaging_config=self.config.averaging,
        num_training_samples=self.config.data_train.config.num_samples,
        num_updates=self.config.num_updates,
    )

  @property
  def forward_fn(self) -> forward.MultiClassForwardFn:
    return self._forward_fn

  def _model_fn(self, inputs, is_training=False, return_features: bool = False):
    model_kwargs = {
        'num_classes': self.config.data_train.config.num_classes,
        **self.config.model.kwargs,
    }
    model_instance = models.get_model_instance(self.config.model.name,
                                               model_kwargs)
    return model_instance(
        inputs,
        is_training=is_training,
        return_features=return_features,
    )

  def _should_restore_model(self) -> bool:
    return bool(self.config.model.restore.path)

  def _restore_model(self):
    self._params, self._network_state = models.restore_from_path(
        restore_path=self.config.model.restore.path,
        params_key=self.config.model.restore.params_key,
        network_state_key=self.config.model.restore.network_state_key,
        layer_to_reset=self.config.model.restore.layer_to_reset,
        params_init=self._params,
        network_state_init=self._network_state,
    )

  def _build_train_input(self) -> Iterator[data.DataInputs]:
    """Builds the training input pipeline."""
    return self.config.data_train.load_dataset(
        batch_dims=(
            jax.local_device_count(),
            self.batching.batch_size_per_device_per_step,
        ),
        is_training=True,
        shard_data=True,
    )

  def _build_eval_input(self) -> Iterator[data.DataInputs]:
    """Builds the evaluation input pipeline."""
    return self.config.data_eval.load_dataset(
        batch_dims=(
            jax.process_count(),
            jax.local_device_count(),
            self.config.evaluation.batch_size,
        ),
        is_training=False,
        shard_data=False,
        max_num_batches=self.config.evaluation.max_num_batches,
    )

  def _build_nc_input(self) -> Iterator[data.DataInputs]:
    """Builds the NC evaluation input pipeline."""
    if self.config.data_nc is None or self.config.nc_evaluation is None:
      raise ValueError('NC evaluation is not configured.')
    return self.config.data_nc.load_dataset(
        batch_dims=(self.config.nc_evaluation.batch_size,),
        is_training=False,
        shard_data=False,
        max_num_batches=self.config.nc_evaluation.max_num_batches,
    )

  def _get_nc_eval_fn(self):
    if self._nc_eval_fn is None:
      self._nc_eval_fn = jax.jit(self._nc_eval_fn_impl)
    return self._nc_eval_fn

  def _nc_eval_fn_impl(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng: chex.PRNGKey,
      images: chex.Array,
      labels: chex.Array,
  ) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Computes per-class sums needed for NC metrics on one batch."""
    outputs, unused_network_state = self._forward_fn._net.apply(
        params, network_state, rng, images, is_training=False,
        return_features=True)
    logits, features = outputs
    del logits
    labels = labels.astype(features.dtype)
    counts = jnp.sum(labels, axis=0)
    sums = jnp.einsum('nc,nd->cd', labels, features)
    sq_norms = jnp.sum(features * features, axis=1)
    sum_sq_norms = jnp.einsum('nc,n->c', labels, sq_norms)
    return counts, sums, sum_sq_norms

  def _extract_classifier_weights(
      self,
      params: hk.Params,
      num_classes: int,
  ) -> chex.Array:
    """Finds the classifier weight matrix of shape [D, C]."""
    candidates = []
    for module_name, param_name, value in hk.data_structures.traverse(params):
      if (param_name == 'w' and value.ndim == 2 and
          value.shape[1] == num_classes):
        candidates.append((module_name, value))

    if not candidates:
      raise ValueError('No classifier weights found for NC metrics.')

    restore_name = None
    if self.config.model.restore is not None:
      restore_name = self.config.model.restore.layer_to_reset
    if restore_name:
      for module_name, value in candidates:
        if module_name == restore_name:
          return value

    if len(candidates) == 1:
      return candidates[0][1]

    for keyword in ('Softmax', 'linear_1', 'fc'):
      for module_name, value in candidates:
        if keyword in module_name:
          return value

    return candidates[0][1]

  def _compute_nc_metrics_for_params(
      self,
      rng: chex.PRNGKey,
      params: hk.Params,
      network_state: hk.State,
  ) -> dict[str, float]:
    """Computes NC metrics for a single set of parameters."""
    nc_config = self.config.nc_evaluation
    if nc_config is None:
      return {}

    if nc_config.use_single_device:
      params = jax.tree_map(lambda x: x[0], params)
      network_state = jax.tree_map(lambda x: x[0], network_state)

    counts_total = None
    sums_total = None
    sum_sq_total = None

    for inputs in self._build_nc_input():
      rng, rng_eval = jax.random.split(rng)
      counts, sums, sum_sq = self._get_nc_eval_fn()(  # pylint: disable=not-callable
          params, network_state, rng_eval, inputs.image, inputs.label)
      counts, sums, sum_sq = jax.device_get((counts, sums, sum_sq))
      if counts_total is None:
        counts_total = np.zeros_like(counts)
        sums_total = np.zeros_like(sums)
        sum_sq_total = np.zeros_like(sum_sq)
      counts_total += counts
      sums_total += sums
      sum_sq_total += sum_sq

    if counts_total is None:
      return {}

    weights = self._extract_classifier_weights(
        params, self.config.data_train.config.num_classes)
    weights = jax.device_get(weights)
    return self._compute_nc_from_stats(
        counts_total, sums_total, sum_sq_total, weights)

  def _compute_nc_from_stats(
      self,
      counts: np.ndarray,
      sums: np.ndarray,
      sum_sq: np.ndarray,
      weights: np.ndarray,
  ) -> dict[str, float]:
    """Computes NC metrics from aggregated per-class stats."""
    eps = 1e-8
    counts = counts.astype(np.float64)
    sums = sums.astype(np.float64)
    sum_sq = sum_sq.astype(np.float64)
    weights = weights.astype(np.float64)

    num_classes = counts.shape[0]
    total_count = np.sum(counts)
    if total_count <= 0:
      return {}

    means = sums / (counts[:, None] + eps)
    global_mean = np.sum(sums, axis=0) / (total_count + eps)
    centered_means = means - global_mean[None, :]

    tr_sw = np.sum(sum_sq) - np.sum(counts * np.sum(means**2, axis=1))
    tr_sb = np.sum(counts * np.sum(centered_means**2, axis=1))
    nc1 = tr_sw / (tr_sb + eps)

    if num_classes > 1:
      centered_norms = np.linalg.norm(centered_means, axis=1, keepdims=True)
      centered_hat = centered_means / (centered_norms + eps)
      gram = centered_hat @ centered_hat.T
      target = ((1.0 + 1.0 / (num_classes - 1)) * np.eye(num_classes) -
                (1.0 / (num_classes - 1)) * np.ones((num_classes, num_classes)))
      nc2 = (np.linalg.norm(gram - target, ord='fro') /
             (np.linalg.norm(target, ord='fro') + eps))
    else:
      nc2 = 0.0

    weight_norms = np.linalg.norm(weights, axis=0, keepdims=True)
    weights_hat = weights / (weight_norms + eps)
    mean_norms = np.linalg.norm(means, axis=1, keepdims=True)
    means_hat = means / (mean_norms + eps)

    nc3 = float(np.mean(np.sum(weights_hat * means_hat.T, axis=0)))
    nc4 = (np.linalg.norm(weights_hat - means_hat.T, ord='fro') /
           (np.linalg.norm(means_hat.T, ord='fro') + eps))

    return {
        'nc1_within_class_collapse_ratio': float(nc1),
        'nc2_mean_simplex_etf_error': float(nc2),
        'nc3_weight_mean_alignment': float(nc3),
        'nc4_self_duality_gap': float(nc4),
    }

  def _compute_nc_metrics(
      self,
      rng: chex.PRNGKey,
      params_dict: dict[str, hk.Params],
      network_state: hk.State,
  ) -> dict[str, float]:
    """Computes NC metrics for configured parameter sets."""
    if self.config.data_nc is None or self.config.nc_evaluation is None:
      return {}
    if self.config.nc_evaluation.host0_only and jax.process_index() != 0:
      return {}

    metrics = {}
    for params_name in self.config.nc_evaluation.params_to_eval:
      params = params_dict.get(params_name)
      if params is None:
        continue
      rng, rng_eval = jax.random.split(rng)
      nc_metrics = self._compute_nc_metrics_for_params(
          rng=rng_eval,
          params=params,
          network_state=network_state,
      )
      metrics.update({f'{k}_{params_name}': v for k, v in nc_metrics.items()})
    return metrics

  def _eval_epoch(self, rng, unused_global_step):
    """Evaluates an epoch."""
    avg_metrics = collections.defaultdict(metrics_module.Avg)

    # Checkpoints broadcast for each local device, which we undo here since the
    # evaluation is performed on a single device (it is not pmapped).
    if isinstance(self._averaging_config.ema_coefficient, Iterable):
      ema_params = {
          f'ema_{ema_decay}': params_ema for ema_decay, params_ema in zip(
              self._averaging_config.ema_coefficient,
              self._params_ema,
              strict=True)
      }
    else:
      ema_params = {'ema': self._params_ema}
    params_dict = {
        'last': self._params,
        **ema_params,
        'polyak': self._params_polyak,
    }
    # Some averaging variants can be disabled and therefore unset.
    params_dict = {
        params_name: params
        for params_name, params in params_dict.items()
        if params is not None
    }

    state = self._network_state
    num_samples = 0
    host_id = jax.process_index()

    # Iterate over the evaluation dataset and accumulate the metrics.
    for inputs in self._build_eval_input():
      rng, rng_eval = jax.random.split(rng)
      num_hosts, num_devices_per_host, batch_size_per_device, *_ = (
          inputs.image.shape)
      batch_size = num_hosts * num_devices_per_host * batch_size_per_device
      num_samples += batch_size
      local_inputs = jax.tree_map(lambda x: x[host_id], inputs)

      # Evaluate batch for each set of parameters.
      for params_name, params in params_dict.items():
        metrics = self.updater.evaluate(params, state, rng_eval, local_inputs)

        # Update accumulated average for each metric.
        for metric_name, val in metrics.scalars.items():
          avg_metrics[f'{metric_name}_{params_name}'].update(val, n=batch_size)

    metrics = {k: v.avg for k, v in avg_metrics.items()}
    metrics['num_samples'] = num_samples
    metrics.update(self._compute_nc_metrics(rng, params_dict, state))

    return metrics
