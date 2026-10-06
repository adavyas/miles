import json
import shlex

import pytest
import torch
from safetensors.torch import load_file, save_file
from tests.e2e.fsdp.test_diffusiongemma_sft import create_fixture, train_arguments
from transformers import AutoConfig, AutoTokenizer

from miles.backends.fsdp_utils.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion
from miles.rollout.diffusion_gemma_sft import tokenize_final_response


def test_functional_rejects_missing_native_checkpoint_parameter(tmp_path):
    from tests.e2e.fsdp.test_diffusiongemma_sft import load_initial_model

    checkpoint, _ = create_fixture(tmp_path)
    filename = checkpoint / "model.safetensors"
    tensors = load_file(filename)
    tensors.pop("model.decoder.layers.0.self_attn.q_proj.weight")
    save_file(tensors, filename, metadata={"format": "pt"})
    with pytest.raises(AssertionError, match="missing_keys"):
        load_initial_model(checkpoint)


def test_functional_fixture_is_native_hf_and_uses_varied_text(tmp_path):
    checkpoint, dataset = create_fixture(tmp_path)
    config = AutoConfig.from_pretrained(checkpoint)
    assert config.text_config.layer_types == ["sliding_attention", "full_attention"]
    assert config.text_config.num_experts == 4
    assert config.text_config.head_dim == 16
    assert config.canvas_length == 8
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    assert tokenizer.is_fast
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(rows) >= 16
    responses = set()
    for row in rows:
        tokens, response_length = tokenize_final_response(tokenizer, messages=row["messages"], template_kwargs={})
        assert 0 < response_length < len(tokens)
        assert tokenizer.unk_token_id not in tokens
        responses.add(tuple(tokens))
    assert len(responses) == len(rows)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(checkpoint)
    assert model.model.decoder.layers[0].encoder_layer_scalar.item() == 0.75
    assert model.model.decoder.layers[0].layer_scalar.item() == 1.25
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_resume_uses_same_training_horizon_and_restores_all_state(tmp_path):
    uninterrupted = shlex.split(train_arguments(tmp_path, save=tmp_path / "full"))
    interrupted = shlex.split(train_arguments(tmp_path, save=tmp_path / "split", stop_after=2))
    resumed = shlex.split(train_arguments(tmp_path, save=tmp_path / "split", load=tmp_path / "split"))
    for argv in (uninterrupted, interrupted, resumed):
        for name, value in [
            ("--num-rollout", "4"),
            ("--lr-decay-iters", "4"),
            ("--lr-warmup-iters", "1"),
            ("--actor-num-gpus-per-node", "2"),
        ]:
            assert argv[argv.index(name) + 1] == value
        assert "--debug-train-only" in argv
        assert "--bf16" in argv
        assert "--no-load-optim" not in argv
        assert "--no-load-rng" not in argv
    assert interrupted[interrupted.index("--debug-exit-after-rollout") + 1] == "2"
    assert "--debug-exit-after-rollout" not in resumed
    assert resumed[resumed.index("--load") + 1] == str(tmp_path / "split")


def test_dcp_reader_preserves_nested_model_optimizer_and_scheduler(tmp_path):
    import torch.distributed.checkpoint as dcp
    from tests.e2e.fsdp.test_diffusiongemma_sft import assert_state_close, read_dcp

    state = {
        "model_state": {"model": {"layer.weight": torch.arange(8).reshape(2, 4).float()}},
        "optim_state": {
            "optim": {"state": {"layer.weight": {"step": torch.tensor(2.0), "exp_avg": torch.ones(2, 4)}}}
        },
        "lr_scheduler_state": {"lr_scheduler": {"last_epoch": 2, "_last_lr": [0.0007]}},
    }
    dcp.save(state, checkpoint_id=str(tmp_path / "checkpoint"))
    restored = read_dcp(tmp_path / "checkpoint", tmp_path / "state.pt")
    assert_state_close(state, restored)


def test_metric_reader_requires_expected_step_and_finite_training_values(tmp_path):
    import pytest
    from tests.e2e.fsdp.test_diffusiongemma_sft import read_metrics

    log = tmp_path / "train.log"
    metrics = {"train/loss": 2.0, "train/diffusion_loss": 1.0, "train/encoder_ar_loss": 1.0, "train/grad_norm": 0.2}
    log.write_text(f"(TrainActor pid=1) step 2: {metrics}\n")
    assert read_metrics(log, expected_steps=[2]) == [metrics]
    with pytest.raises(AssertionError):
        read_metrics(log, expected_steps=[0])
    metrics["train/grad_norm"] = 0.0
    log.write_text(f"step 2: {metrics}\n")
    with pytest.raises(AssertionError):
        read_metrics(log, expected_steps=[2])
