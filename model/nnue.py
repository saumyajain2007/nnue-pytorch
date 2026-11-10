import dataclasses

import torch
from torch import Tensor, nn
from torchmetrics import MeanMetric, MetricCollection

from .config import NNUELightningConfig
from .lambda_utils import LambdaController
from .model import NNUEModel
from .phase_lambda import load_lambda_table, lookup_lambda


def _get_parameters(layers: list[nn.Module], get_biases: bool = False):
    return [
        p
        for layer in layers
        for name, p in layer.named_parameters()
        if ("bias" in name) == get_biases and p.requires_grad
    ]


def remap_tablebase_score(
    score: Tensor,
    base: float,
    scale: float,
    decay: float
) -> Tensor:
    mate_score = 32000
    max_mate_ply = 245
    tb_mate_threshold = mate_score - max_mate_ply

    abs_score = score.abs()
    is_mate = abs_score >= tb_mate_threshold

    plies = (mate_score - abs_score).clamp(min=0)
    remapped_abs = base + (decay ** plies) * scale
    remapped_score = torch.where(score < 0, -remapped_abs, remapped_abs)

    return torch.where(is_mate, remapped_score, score)


def expected_score(x: Tensor, offset: float, scaling: float) -> Tensor:
    """Map a centipawn-like score to an expected game result in [0, 1].

    This is the symmetric two-sigmoid win-rate model the trainer has always
    used; it is factored out so the loss and the validation metrics share it.
    """
    p = (x - offset) / scaling
    pm = (-x - offset) / scaling
    return 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())


def calculate_sf_loss(scorenet, score, outcome, loss_params, actual_lambda):
    score = remap_tablebase_score(
        score,
        base=loss_params.tb_remap_base,
        scale=loss_params.tb_remap_scale,
        decay=loss_params.tb_remap_decay,
    )

    # Convert network and search scores into expected match results.
    qf = expected_score(scorenet, loss_params.in_offset, loss_params.in_scaling)
    pf = expected_score(score, loss_params.out_offset, loss_params.out_scaling)

    # Actual game outcome: win = 1, draw = 0.5, loss = 0.
    t = outcome

    # Blend search-derived targets with game outcomes (actual_lambda may be a
    # scalar or per sample, e.g. with jitter or --lambda-phase-delta).
    pt = pf * actual_lambda + t * (1.0 - actual_lambda)

    # Stockfish-style power-law loss, with optional asymmetric penalty.
    loss = torch.pow(torch.abs(pt - qf), loss_params.pow_exp)
    if loss_params.qp_asymmetry != 0.0:
        loss = loss * ((qf > pt) * loss_params.qp_asymmetry + 1)

    # Auxiliary outcome-calibration term (first introduced in b882e6c with a
    # hard-coded weight of 0.1). The Python-level branch keeps the original
    # loss bit-exact when it is disabled.
    if loss_params.outcome_aux_weight != 0.0:
        loss = loss + loss_params.outcome_aux_weight * (qf - t).square()

    # Confidence-dependent sample weighting.
    weights = 1 + (2.0**loss_params.w1 - 1) * torch.pow(
        (pf - 0.5) ** 2 * pf * (1 - pf), loss_params.w2
    )
    loss = (loss * weights).sum() / weights.sum()

    return loss


def apply_phase_lambda(actual_lambda, piece_count: Tensor, delta: float, like: Tensor) -> Tensor:
    """Shift lambda per sample by ``delta * phase`` where phase goes 0 -> 1
    from 32 pieces down to 2 pieces (bare kings).

    ``piece_count`` may arrive as (B,) or (B, 1) while scores are (B, 1); it
    is reshaped explicitly so the two never broadcast into a (B, B) matrix.
    """
    pc = piece_count.to(like.dtype).reshape(like.shape)
    phase = ((32.0 - pc) / 30.0).clamp(0.0, 1.0)
    return (actual_lambda + delta * phase).clamp(0.0, 1.0)


@torch.no_grad()
def calculate_reference_metrics(scorenet: Tensor, score: Tensor, outcome: Tensor, loss_params) -> dict[str, Tensor]:
    """Validation metrics that do not depend on the training objective.

    * ``ref_loss``: the original SF loss at lambda = 1 with no auxiliary term
      (fit to search scores).
    * ``outcome_mse``: unweighted mean of (expected_score(net) - result)^2,
      the Brier score of the net against the game result.

    ``ref_loss`` still uses pow_exp / qp_asymmetry / w1 / w2 and the scaling
    constants from ``loss_params``, so only compare runs that share them.
    """
    ref_params = dataclasses.replace(loss_params, outcome_aux_weight=0.0)
    ref_loss = calculate_sf_loss(scorenet, score, outcome, ref_params, 1.0)
    qf = expected_score(scorenet, loss_params.in_offset, loss_params.in_scaling)
    outcome_mse = (qf - outcome).square().mean()
    return {"ref_loss": ref_loss, "outcome_mse": outcome_mse}


class NNUE(nn.Module):

    def __init__(
        self,
        config: NNUELightningConfig,
        max_epoch=None,
        num_batches_per_epoch=None,
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
    ):
        super().__init__()

        self.model: NNUEModel = NNUEModel(
            config.features,
            config.model_config,
            num_psqt_buckets,
            num_ls_buckets,
        )
        self.config = config
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.param_index = param_index

        self.max_steps = 0
        if max_epoch is not None and num_batches_per_epoch is not None:
            self.max_steps = max_epoch * num_batches_per_epoch
        self.step_counter = 0

        # lazy init so `resume_from_model` with config changes works correctly
        self.optimizer_wrapper = None

        # Initialize the lambda controller
        self.lambda_scheduler = LambdaController()

        # Optional per-piece-count lambda table (--lambda-phase-table). Not
        # persistent: checkpoints stay loadable without the table file.
        table_path = config.loss_params.lambda_config.lambda_phase_table
        self.register_buffer(
            "lambda_phase_table",
            load_lambda_table(table_path) if table_path is not None else None,
            persistent=False,
        )

        self.loss_metrics = MetricCollection(
            {
                "train_loss_epoch": MeanMetric(),
                "val_loss_epoch": MeanMetric(),
                "test_loss_epoch": MeanMetric(),
            }
        )

    # --- setup optimizers and training hooks ---
    def configure_optimizers(self):
        optimizer_config = self.config.optimizer_config
        self.optimizer_wrapper = optimizer_config.get_optimizer_wrapper()

        LRs = [optimizer_config.lr] * 10

        ft_wd = optimizer_config.ft_weight_decay
        dense_wd = optimizer_config.dense_weight_decay
        factorized_wd = optimizer_config.factorized_weight_decay

        train_params = [
            # Feature Transformer
            {
                "params": _get_parameters([self.model.input], get_biases=False),
                "lr": LRs[0],
                "weight_decay": ft_wd,
            },
            {
                "params": _get_parameters([self.model.input], get_biases=True),
                "lr": LRs[1],
                "weight_decay": 0.0,
            },
            # Dense Layer Stacks
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.weight],
                "lr": LRs[2],
                "weight_decay": factorized_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.bias],
                "lr": LRs[3],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.weight],
                "lr": LRs[4],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.bias],
                "lr": LRs[5],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.weight],
                "lr": LRs[6],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.bias],
                "lr": LRs[7],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.output.linear.weight],
                "lr": LRs[8],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.output.linear.bias],
                "lr": LRs[9],
                "weight_decay": 0.0,
            },
        ]

        return self.optimizer_wrapper.configure_optimizers(train_params)

    # --- train / eval switch ---
    def train(self, mode: bool = True):
        retval = super().train(mode)

        if self.optimizer_wrapper is not None:
            if mode:
                self.optimizer_wrapper.switch_to_train(True)
            else:
                self.optimizer_wrapper.switch_to_eval()

        return retval

    def eval(self):
        return self.train(False)

    @staticmethod
    def load_from_checkpoint(path, config, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        nnue = NNUE(config=config)
        nnue.load_state_dict(checkpoint["state_dict"])
        return nnue

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # --- hooks ---
    def on_train_epoch_start(self):
        self.optimizer_wrapper.on_train_epoch_start(self)
        if self.max_epoch is not None and self.num_batches_per_epoch is not None:
            self.max_steps = self.max_epoch * self.num_batches_per_epoch

    def on_train_epoch_end(self):
        self.optimizer_wrapper.on_train_epoch_end(self)
        self.loss_metrics["train_loss_epoch"].reset()

    def on_validation_epoch_start(self):
        self.optimizer_wrapper.on_validation_epoch_start(self)

    def on_validation_epoch_end(self):
        self.loss_metrics["val_loss_epoch"].reset()

    def on_test_epoch_start(self):
        self.optimizer_wrapper.on_test_epoch_start(self)

    def on_test_epoch_end(self):
        self.loss_metrics["test_loss_epoch"].reset()

    def on_save_checkpoint(self, checkpoint):
        self.optimizer_wrapper.on_save_checkpoint(self, checkpoint)
        self.lambda_scheduler.on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint, resuming: bool):
        self.lambda_scheduler.on_load_checkpoint(self, checkpoint, resuming=resuming)

    def on_train_batch_start(self, batch, batch_idx):
        self.optimizer_wrapper.on_train_batch_start(self, batch, batch_idx)

    # --- checkpoint state helpers ---
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)

        # torch.compile() adds _orig_mod. prefixes to parameter/buffer keys.
        # Strip them so checkpoints remain loadable by an uncompiled NNUE.
        if any("_orig_mod." in k for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

        # Lambda scheduler state (jitter_buffer is persistent=False)
        state["lambda_scheduler_state_dict"] = self.lambda_scheduler.state_dict()
        state["jitter_buffer_value"] = self.lambda_scheduler.jitter_buffer

        # Optimizer state
        if self.optimizer_wrapper is not None and getattr(self.optimizer_wrapper, "optimizer", None) is not None:
            state["optimizer_state_dict"] = self.optimizer_wrapper.optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = dict(state_dict)
        lambda_scheduler_state = state_dict.pop("lambda_scheduler_state_dict", None)
        jitter_buffer_value = state_dict.pop("jitter_buffer_value", None)
        optimizer_state = state_dict.pop("optimizer_state_dict", None)

        # If this NNUE (or a submodule) was torch.compile'd, the expected keys
        # contain _orig_mod. prefixes that plain checkpoint keys don't have.
        # Build a mapping from the plain key to the actual key and remap.
        expected = super().state_dict()
        if any("_orig_mod." in k for k in expected):
            remap = {k.replace("_orig_mod.", ""): k for k in expected}
            state_dict = {remap.get(k, k): v for k, v in state_dict.items()}

        super().load_state_dict(state_dict, *args, **kwargs)

        if lambda_scheduler_state is not None:
            self.lambda_scheduler.load_state_dict(lambda_scheduler_state)

        if jitter_buffer_value is not None:
            self.lambda_scheduler.jitter_buffer.copy_(
                jitter_buffer_value.to(
                    device=self.lambda_scheduler.jitter_buffer.device,
                    dtype=self.lambda_scheduler.jitter_buffer.dtype,
                )
            )

        if (
            optimizer_state is not None
            and self.optimizer_wrapper is not None
            and getattr(self.optimizer_wrapper, "optimizer", None) is not None
        ):
            self.optimizer_wrapper.optimizer.load_state_dict(optimizer_state)

    # --- Training step implementation ---

    def train_step(self, batch, current_epoch, global_step):
        _ = current_epoch
        loss = self.compute_loss(batch, global_step)
        self.loss_metrics["train_loss_epoch"].update(loss)
        return {"loss": loss, "train_loss": loss.detach()}

    @torch.no_grad()
    def val_step(self, batch, current_epoch, global_step):
        _ = current_epoch
        scorenet = self._forward_batch(batch)
        loss = self.compute_loss_with_scorenet(scorenet, batch, global_step)
        self.loss_metrics["val_loss_epoch"].update(loss)

        outputs = {"val_loss": loss}
        _us, _them, _wi, _bi, outcome, score, *_rest = batch
        ref = calculate_reference_metrics(
            scorenet * self.model.quantization.nnue2score,
            score,
            outcome,
            self.config.loss_params,
        )
        outputs.update({f"val_{k}": v for k, v in ref.items()})
        return outputs

    @torch.no_grad()
    def test_step(self, batch, current_epoch, global_step):
        _ = current_epoch
        loss = self.compute_loss(batch, global_step)
        self.loss_metrics["test_loss_epoch"].update(loss)
        return {"test_loss": loss}

    def _forward_batch(self, batch: tuple[Tensor, ...]) -> Tensor:
        (
            us,
            them,
            white_indices,
            black_indices,
            _outcome,
            _score,
            piece_count,
        ) = batch
        return self.model(
            us,
            them,
            white_indices,
            black_indices,
            piece_count,
            self.config.use_fake_act_quantization,
            self.config.use_fake_weight_quantization,
        )

    def compute_loss(self, batch: tuple[Tensor, ...], current_step: int):
        scorenet = self._forward_batch(batch)
        return self.compute_loss_with_scorenet(scorenet, batch, current_step)

    def compute_loss_with_scorenet(
        self, scorenet: Tensor, batch: tuple[Tensor, ...], current_step: int
    ):
        (
            _us,
            _them,
            _white_indices,
            _black_indices,
            outcome,
            score,
            piece_count,
        ) = batch

        scorenet = scorenet * self.model.quantization.nnue2score

        actual_lambda = self.lambda_scheduler(
            loss_params=self.config.loss_params,
            current_step=current_step,
            max_steps=self.max_steps,
            is_training=self.training,
            scorenet=scorenet,
        )

        # Per-piece-count lambda table replaces the scheduled lambda.
        if self.lambda_phase_table is not None:
            actual_lambda = lookup_lambda(self.lambda_phase_table, piece_count, scorenet)

        # Phase-dependent lambda. Applied in training and validation alike:
        # it is part of the objective, not a regulariser like jitter.
        phase_delta = self.config.loss_params.lambda_config.lambda_phase_delta
        if phase_delta != 0.0:
            actual_lambda = apply_phase_lambda(actual_lambda, piece_count, phase_delta, scorenet)

        sf_loss = calculate_sf_loss(
            scorenet, score, outcome, self.config.loss_params, actual_lambda
        )

        return sf_loss
