from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .dpo_trainer import DPOTrainer

class CPOPlusTrainer(DPOTrainer):
    """DPOTrainer specialized for dual-stream CPO++ preference pairs.

    The trainer is intentionally a subclass of ms-swift's ``DPOTrainer`` so it
    retains its model/template setup, PEFT reference-policy handling,
    distributed metric collection, RPO regularization, padding-free execution,
    and MoE auxiliary loss.

    Trainer-only keyword arguments may be supplied directly.  If omitted, the
    corresponding attributes are read from ``args`` and finally fall back to
    the defaults shown below.

    Args:
        cpo_source_key: Batch key containing the negative-source label.
        cpo_sample_weight_key: Batch key containing optional pair weights.
        cpo_thinking_weight: Weight for counterfactual-thinking pairs.
        cpo_perception_weight: Weight for counterfactual-perception pairs.
        cpo_other_weight: Weight for unlabelled or other pairs.
        cpo_normalize_weights: Normalize pair weights to mean one.  This keeps
            the effective learning-rate scale unchanged when source weights
            are used.
    """

    THINKING = 0
    PERCEPTION = 1
    OTHER = 2

    _SOURCE_ALIASES = {
        '0': THINKING,
        'thinking': THINKING,
        'thought': THINKING,
        'reasoning': THINKING,
        'text': THINKING,
        'textual': THINKING,
        'counterfactual_thinking': THINKING,
        'thinking_counterfactual': THINKING,
        '1': PERCEPTION,
        'perception': PERCEPTION,
        'perceptual': PERCEPTION,
        'vision': PERCEPTION,
        'visual': PERCEPTION,
        'image': PERCEPTION,
        'counterfactual_perception': PERCEPTION,
        'perception_counterfactual': PERCEPTION,
        '2': OTHER,
        'other': OTHER,
        'unknown': OTHER,
    }

    def __init__(
        self,
        model=None,
        ref_model=None,
        *_args,
        cpo_source_key: Optional[str] = None,
        cpo_sample_weight_key: Optional[str] = None,
        cpo_thinking_weight: Optional[float] = None,
        cpo_perception_weight: Optional[float] = None,
        cpo_other_weight: Optional[float] = None,
        cpo_normalize_weights: Optional[bool] = None,
        **kwargs,
    ):
        args = kwargs['args']

        self.cpo_source_key = cpo_source_key or getattr(args, 'cpo_source_key', 'cpo_source')
        self.cpo_sample_weight_key = (
            cpo_sample_weight_key or getattr(args, 'cpo_sample_weight_key', 'cpo_sample_weight'))
        self.cpo_thinking_weight = self._resolve_option(
            cpo_thinking_weight, args, 'cpo_thinking_weight', 1.0)
        self.cpo_perception_weight = self._resolve_option(
            cpo_perception_weight, args, 'cpo_perception_weight', 1.0)
        self.cpo_other_weight = self._resolve_option(cpo_other_weight, args, 'cpo_other_weight', 1.0)
        self.cpo_normalize_weights = self._resolve_option(
            cpo_normalize_weights, args, 'cpo_normalize_weights', True)

        for name in ('cpo_thinking_weight', 'cpo_perception_weight', 'cpo_other_weight'):
            if float(getattr(self, name)) < 0:
                raise ValueError(f'`{name}` must be non-negative, but received {getattr(self, name)}.')

        super().__init__(model, ref_model, *_args, **kwargs)

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        if loss_types != ['sigmoid']:
            raise ValueError(
                'CPOPlusTrainer implements the CPO++ Bradley-Terry objective and therefore requires '
                '`loss_type="sigmoid"`. Use DPOTrainer directly for other DPO-family losses.')
        if self.reference_free:
            raise ValueError('CPO++ requires the SFT reference policy; `reference_free` must be false.')
        if self.f_divergence_type != 'reverse_kl':
            raise ValueError('CPO++ Eq. (9) requires `f_divergence_type="reverse_kl"`.')

    @staticmethod
    def _resolve_option(value, args, name, default):
        if value is not None:
            return value
        return getattr(args, name, default)

    @classmethod
    def _source_id(cls, value) -> int:
        if isinstance(value, str):
            return cls._SOURCE_ALIASES.get(value.strip().lower(), cls.OTHER)
        try:
            value = int(value)
        except (TypeError, ValueError):
            return cls.OTHER
        return value if value in (cls.THINKING, cls.PERCEPTION, cls.OTHER) else cls.OTHER

    def _pop_cpo_metadata(self, batch: Dict) -> Tuple[Dict, object, object]:
        """Return a model-safe shallow copy and the CPO++ pair metadata."""
        model_batch = batch.copy()

        source = model_batch.pop(self.cpo_source_key, None)
        if source is None and self.cpo_source_key != 'negative_type':
            source = model_batch.pop('negative_type', None)

        sample_weight = model_batch.pop(self.cpo_sample_weight_key, None)

        # Precomputed reference log-probabilities are trainer metadata and must
        # never be passed as keyword arguments to the policy model.
        model_batch.pop('ref_chosen_logps', None)
        model_batch.pop('ref_rejected_logps', None)
        return model_batch, source, sample_weight

    def _source_tensor(self, source, batch_size: int, device: torch.device) -> torch.LongTensor:
        if source is None:
            values = [self.OTHER] * batch_size
        elif isinstance(source, torch.Tensor):
            if source.numel() == 1:
                values = [self._source_id(source.item())] * batch_size
            else:
                values = [self._source_id(value) for value in source.detach().cpu().reshape(-1).tolist()]
        elif isinstance(source, (list, tuple)):
            values = [self._source_id(value) for value in source]
        else:
            values = [self._source_id(source)] * batch_size

        if len(values) != batch_size:
            raise ValueError(
                f'CPO++ source metadata has {len(values)} entries, but the preference batch has '
                f'{batch_size} pairs.')
        return torch.tensor(values, dtype=torch.long, device=device)

    @staticmethod
    def _sample_weight_tensor(sample_weight, batch_size: int, device: torch.device, dtype: torch.dtype):
        if sample_weight is None:
            weights = torch.ones(batch_size, dtype=dtype, device=device)
        elif isinstance(sample_weight, torch.Tensor):
            weights = sample_weight.to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(sample_weight, (list, tuple)):
            weights = torch.tensor(sample_weight, dtype=dtype, device=device).reshape(-1)
        else:
            weights = torch.full((batch_size, ), float(sample_weight), dtype=dtype, device=device)

        if weights.numel() == 1 and batch_size != 1:
            weights = weights.expand(batch_size)
        if weights.numel() != batch_size:
            raise ValueError(
                f'CPO++ sample weights have {weights.numel()} entries, but the preference batch has '
                f'{batch_size} pairs.')
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('CPO++ sample weights must be finite and non-negative.')
        return weights

    def _pair_weights(self, source_ids, sample_weight, dtype):
        weights = torch.full(source_ids.shape, float(self.cpo_other_weight), dtype=dtype, device=source_ids.device)
        weights = torch.where(
            source_ids == self.THINKING,
            torch.as_tensor(self.cpo_thinking_weight, dtype=dtype, device=source_ids.device),
            weights,
        )
        weights = torch.where(
            source_ids == self.PERCEPTION,
            torch.as_tensor(self.cpo_perception_weight, dtype=dtype, device=source_ids.device),
            weights,
        )
        weights = weights * self._sample_weight_tensor(
            sample_weight, source_ids.numel(), source_ids.device, dtype)

        if not (weights > 0).any():
            raise ValueError('At least one CPO++ pair in a batch must have a positive weight.')
        if self.cpo_normalize_weights:
            weights = weights / weights.mean().clamp_min(torch.finfo(dtype).eps)
        return weights

    def cpo_plus_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CPO++ objective from Eq. (9).

        Both counterfactual-thinking and counterfactual-perception pairs use
        this objective.  Their source affects only optional pair weighting and
        source-wise monitoring in :meth:`get_batch_loss_metrics`.
        """
        device = self.accelerator.device
        chosen_logps = chosen_logps.to(device)
        rejected_logps = rejected_logps.to(device)
        ref_chosen_logps = ref_chosen_logps.to(device)
        ref_rejected_logps = ref_rejected_logps.to(device)

        ref_scale = 0.0 if self.reference_free else 1.0
        chosen_logratios = chosen_logps - ref_scale * ref_chosen_logps
        rejected_logratios = rejected_logps - ref_scale * ref_rejected_logps
        preference_logits = chosen_logratios - rejected_logratios

        losses = (
            -F.logsigmoid(self.beta * preference_logits) * (1 - self.label_smoothing)
            -F.logsigmoid(-self.beta * preference_logits) * self.label_smoothing)

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        return losses, chosen_rewards, rejected_rewards

    def _add_source_metrics(self, metrics, prefix, source_ids, losses, accuracies, pair_weights):
        gathered_source_ids = self.accelerator.gather_for_metrics(source_ids).detach()
        gathered_losses = self.accelerator.gather_for_metrics(losses).detach()
        gathered_accuracies = self.accelerator.gather_for_metrics(accuracies).detach()
        gathered_weights = self.accelerator.gather_for_metrics(pair_weights).detach()

        source_names = {
            self.THINKING: 'thinking',
            self.PERCEPTION: 'perception',
            self.OTHER: 'other',
        }
        for source_id, source_name in source_names.items():
            mask = gathered_source_ids == source_id
            if mask.any():
                metrics[f'{prefix}cpo_plus/{source_name}_loss'] = gathered_losses[mask].mean().item()
                metrics[f'{prefix}cpo_plus/{source_name}_accuracy'] = gathered_accuracies[mask].mean().item()
                metrics[f'{prefix}cpo_plus/{source_name}_weight'] = gathered_weights[mask].mean().item()
                metrics[f'{prefix}cpo_plus/{source_name}_count'] = mask.sum().item()

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute dual-stream CPO++ loss and source-aware metrics."""
        metrics = {}
        model_batch, source, sample_weight = self._pop_cpo_metadata(batch)
        model_output = self.concatenated_forward(model, model_batch)

        if 'ref_chosen_logps' in batch and 'ref_rejected_logps' in batch:
            ref_chosen_logps = batch['ref_chosen_logps']
            ref_rejected_logps = batch['ref_rejected_logps']
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(model_batch)

        unweighted_losses, chosen_rewards, rejected_rewards = self.cpo_plus_loss(
            model_output['chosen_logps'],
            model_output['rejected_logps'],
            ref_chosen_logps,
            ref_rejected_logps,
        )

        batch_size = unweighted_losses.shape[0]
        source_ids = self._source_tensor(source, batch_size, unweighted_losses.device)
        pair_weights = self._pair_weights(source_ids, sample_weight, unweighted_losses.dtype)
        losses = unweighted_losses * pair_weights
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output['nll_loss']
        if self.use_weighting:
            losses = losses * model_output['policy_weights']
        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output['aux_loss']

        prefix = 'eval_' if train_eval == 'eval' else ''
        gather = self.accelerator.gather_for_metrics
        metrics[f'{prefix}rewards/chosen'] = gather(chosen_rewards).mean().item()
        metrics[f'{prefix}rewards/rejected'] = gather(rejected_rewards).mean().item()
        metrics[f'{prefix}rewards/accuracies'] = gather(reward_accuracies).mean().item()
        metrics[f'{prefix}rewards/margins'] = gather(chosen_rewards - rejected_rewards).mean().item()
        metrics[f'{prefix}logps/chosen'] = gather(model_output['chosen_logps']).detach().mean().item()
        metrics[f'{prefix}logps/rejected'] = gather(model_output['rejected_logps']).detach().mean().item()
        metrics[f'{prefix}logits/chosen'] = gather(model_output['mean_chosen_logits']).detach().mean().item()
        metrics[f'{prefix}logits/rejected'] = gather(model_output['mean_rejected_logits']).detach().mean().item()
        metrics[f'{prefix}cpo_plus/loss'] = gather(unweighted_losses).detach().mean().item()
        metrics[f'{prefix}cpo_plus/weighted_loss'] = gather(
            unweighted_losses * pair_weights).detach().mean().item()

        self._add_source_metrics(
            metrics, prefix, source_ids, unweighted_losses, reward_accuracies, pair_weights)

        if self.args.rpo_alpha is not None:
            metrics[f'{prefix}nll_loss'] = gather(model_output['nll_loss']).detach().mean().item()
        if self.aux_loss_enabled:
            metrics[f'{prefix}aux_loss'] = gather(model_output['aux_loss']).detach().mean().item()

        return losses.mean(), metrics


# Convenient alias for codebases that abbreviate CPO++ as CPOPP.
CPOPPTrainer = CPOPlusTrainer


__all__ = ['CPOPlusTrainer', 'CPOPPTrainer']
