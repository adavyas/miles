"""Two-rank CUDA optimizer + native DCP resume smoke test.

Run in the Miles training image on an isolated host:
MILES_DIFFUSION_TEST_CHECKPOINT_DIR=/tmp/diffusion-sft-smoke \
  torchrun --standalone --nproc-per-node=2 -m pytest --noconftest \
  tests/manual/backends/test_diffusiongemma_fsdp_sft.py -q
The checkpoint directory must be fresh and shared by both ranks.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a two-rank CUDA Miles training image")
def test_two_rank_update_and_native_checkpoint_resume():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch this smoke test through torchrun --nproc-per-node=2")
    # Distributed actor imports require the training image, not the CPU test environment.
    from tests.fast.backends.diffusion_gemma.test_engine import rows
    from tests.fast.backends.diffusion_gemma.test_model import tiny_config

    from miles.backends.fsdp_utils import checkpoint
    from miles.backends.fsdp_utils.actor import FSDPTrainRayActor, apply_fsdp2
    from miles.backends.fsdp_utils.diffusion_gemma.engine import _optimizer_step
    from miles.backends.fsdp_utils.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion

    directory = os.environ["MILES_DIFFUSION_TEST_CHECKPOINT_DIR"]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        actor = _actor(FSDPTrainRayActor, apply_fsdp2, DiffusionGemmaForBlockDiffusion, tiny_config, directory)
        cuda_rows = [
            {**row, "tokens": [t.cuda() for t in row["tokens"]], "loss_masks": [m.cuda() for m in row["loss_masks"]]}
            for row in rows()
        ]
        _, diffusion, encoder = _optimizer_step(actor, rows=cuda_rows, dp_size=2, rank=dist.get_rank(), group=None)
        assert diffusion > 0 and encoder > 0
        assert actor.global_step == 1 and actor.micro_step == 2
        checkpoint.save(actor, iteration=0)
        restored = _actor(FSDPTrainRayActor, apply_fsdp2, DiffusionGemmaForBlockDiffusion, tiny_config, directory)
        payload = checkpoint.load(restored)
        assert payload is not None
        checkpoint.finalize_load(restored, payload)
        assert restored.global_step == 1 and restored.micro_step == 2 and restored.args.start_rollout_id == 1
        _optimizer_step(actor, rows=cuda_rows, dp_size=2, rank=dist.get_rank(), group=None)
        _optimizer_step(restored, rows=cuda_rows, dp_size=2, rank=dist.get_rank(), group=None)
        for expected, actual in zip(actor.model.parameters(), restored.model.parameters(), strict=True):
            torch.testing.assert_close(expected.full_tensor(), actual.full_tensor(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _actor(actor_cls, shard, model_cls, config_factory, directory):
    from torch.distributed.device_mesh import init_device_mesh

    from miles.backends.fsdp_utils.adaptations.precision import PrecisionPolicy

    config = config_factory()
    config.canvas_length = 4
    config.text_config.sliding_window = 3
    config._attn_implementation = "sdpa"
    torch.manual_seed(123)
    model = model_cls(config).float().cuda()
    model.gradient_checkpointing_enable()
    for name, parameter in model.named_parameters():
        if ".router." in name:
            parameter.requires_grad_(False)
    actor = object.__new__(actor_cls)
    actor.args = SimpleNamespace(
        fp16=False,
        clip_grad=1.0,
        seed=42,
        diffusion_noise_epsilon=0.001,
        diffusion_self_conditioning_probability=0.5,
        diffusion_encoder_loss_weight=1.0,
        save=directory,
        load=directory,
        start_rollout_id=0,
    )
    actor.hf_config = config
    actor.precision_policy = PrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    actor.model = shard(model, mesh=init_device_mesh("cuda", (2,)), args=actor.args)
    actor.optimizer = torch.optim.AdamW(actor.model.parameters(), lr=1e-3)
    actor.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(actor.optimizer, lambda step: 1.0)
    actor.global_step = actor.micro_step = 0
    return actor
