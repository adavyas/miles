"""Two-GPU offline DiffusionGemma SFT, including a real interrupted restart."""

import ast
import json
import math
import os
import re
import shlex
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from tests.ci.ci_register import register_cuda_ci
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import DiffusionGemmaConfig, DiffusionGemmaTextConfig, Gemma4VisionConfig, PreTrainedTokenizerFast
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

register_cuda_ci(est_time=600, suite="stage-b-2-gpu-h200", labels=["fsdp"], hardware=["hopper", "blackwell"])

NUM_GPUS = 2
NUM_ROLLOUTS = 4
BATCH_SIZE = 4


def create_fixture(root: Path) -> tuple[Path, Path]:
    """Write native HF weights, a fast tokenizer, and distinct labeled conversations."""
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "hf_checkpoint"
    words = ["<pad>", "<eos>", "<unk>", "<user>", "<assistant>", "question", "answer", "value", "is"]
    words += [str(index) for index in range(32)]
    words += [f"word{index}" for index in range(128 - len(words))]
    backend = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<user>", "<assistant>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + ' <eos> ' }}{% endfor %}"
    )
    tokenizer.save_pretrained(checkpoint)
    text_config = DiffusionGemmaTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=2,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
        sliding_window=16,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    # A tiny vision branch makes this a native HF checkpoint while the training
    # wrapper exercises its deliberate text-only projection and scalar mapping.
    config = DiffusionGemmaConfig(
        text_config=text_config,
        vision_config=Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            position_embedding_size=16,
        ),
        canvas_length=8,
        eos_token_id=1,
        pad_token_id=0,
    )
    torch.manual_seed(2026)
    model = DiffusionGemmaForBlockDiffusion(config)
    model.model.encoder.language_model.layers[0].layer_scalar.fill_(0.75)
    model.model.decoder.layers[0].layer_scalar.fill_(1.25)
    model.save_pretrained(checkpoint)
    dataset = root / "conversations.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"question {index} value is {index + 1}"},
                {
                    "role": "assistant",
                    "content": f"answer {index + 1} " + " ".join(f"word{j}" for j in range(1 + index % 5)),
                },
            ]
        }
        for index in range(16)
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return checkpoint, dataset


def train_arguments(root: Path, *, save: Path, load: Path | None = None, stop_after: int | None = None) -> str:
    argv = [
        "--train-backend",
        "fsdp",
        "--hf-checkpoint",
        str(root / "hf_checkpoint"),
        "--prompt-data",
        str(root / "conversations.jsonl"),
        "--input-key",
        "messages",
        "--rollout-function-path",
        "miles.rollout.diffusion_gemma_sft.generate_rollout",
        "--loss-type",
        "sft_loss",
        "--debug-train-only",
        "--disable-compute-advantages-and-returns",
        "--attn-implementation",
        "sdpa",
        "--kernel-backend",
        "native",
        "--qkv-format",
        "bshd",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(NUM_GPUS),
        "--num-gpus-per-node",
        str(NUM_GPUS),
        "--num-rollout",
        str(NUM_ROLLOUTS),
        "--rollout-batch-size",
        str(BATCH_SIZE),
        "--n-samples-per-prompt",
        "1",
        "--global-batch-size",
        str(BATCH_SIZE),
        "--micro-batch-size",
        "1",
        "--optimizer",
        "adam",
        "--lr",
        "0.001",
        "--min-lr",
        "0.0001",
        "--lr-decay-style",
        "linear",
        "--lr-decay-iters",
        str(NUM_ROLLOUTS),
        "--lr-warmup-iters",
        "1",
        "--weight-decay",
        "0.01",
        "--bf16",
        "--gradient-checkpointing",
        "--seed",
        "42",
        "--save",
        str(save),
        "--save-interval",
        "1",
    ]
    if load is not None:
        argv += ["--load", str(load)]
    if stop_after is not None:
        argv += ["--debug-exit-after-rollout", str(stop_after)]
    return shlex.join(argv)


def read_dcp(directory: Path, destination: Path) -> dict:
    # Import only during verification: fixture tests do not need the DCP runtime.
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    dcp_to_torch_save(str(directory), str(destination))
    return torch.load(destination, map_location="cpu", weights_only=False)


def load_initial_model(checkpoint: Path):
    # The native checkpoint deliberately contains a vision branch that this
    # text trainer omits. Every tensor belonging to the training stack must load.
    from miles.backends.fsdp_utils.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion as TrainingModel

    model, info = TrainingModel.from_pretrained(checkpoint, output_loading_info=True)
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not info[name], (name, info[name])
    native = load_file(checkpoint / "model.safetensors")
    for name, tensor in model.state_dict().items():
        native_name = name
        if name.endswith(".encoder_layer_scalar"):
            native_name = name.replace("model.decoder.layers.", "model.encoder.language_model.layers.")
            native_name = native_name.replace(".encoder_layer_scalar", ".layer_scalar")
        elif name == "lm_head.weight":
            native_name = "model.decoder.embed_tokens.weight"
        assert native_name in native, native_name
        torch.testing.assert_close(tensor, native[native_name], rtol=0, atol=0)
    return model


def assert_state_close(expected, actual, *, path: str = "state") -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.isfinite(actual).all(), path
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=path)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            assert_state_close(expected[key], actual[key], path=f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_state_close(left, right, path=f"{path}.{index}")
    else:
        assert expected == actual, (path, expected, actual)


def read_metrics(log_path: Path, *, expected_steps: list[int]) -> list[dict]:
    records = []
    steps = []
    for line in log_path.read_text().splitlines():
        match = re.search(r"\bstep (\d+): (\{.*\})", line)
        if not match:
            continue
        metrics = ast.literal_eval(match.group(2))
        for key in ("train/loss", "train/diffusion_loss", "train/encoder_ar_loss", "train/grad_norm"):
            assert math.isfinite(metrics[key]), (log_path, key, metrics)
            assert metrics[key] > 0, (log_path, key, metrics)
        steps.append(int(match.group(1)))
        records.append(metrics)
    assert steps == expected_steps, (log_path, steps, expected_steps)
    return records


def verify_checkpoint(root: Path, save: Path, *, steps: int) -> dict:
    checkpoint = save / f"iter_{steps:07d}"
    metadata = json.loads((checkpoint / "meta.json").read_text())
    for key, expected in {
        "global_step": steps,
        "micro_step": steps * 2,
        "next_rollout_id": steps,
        "world_size": NUM_GPUS,
    }.items():
        assert metadata[key] == expected, (key, metadata)
    dataset = torch.load(save / "rollout" / f"global_dataset_state_dict_{steps - 1}.pt", weights_only=False)
    assert dataset["sample_offset"] == steps * BATCH_SIZE, dataset
    assert dataset["sample_index"] == steps * BATCH_SIZE, dataset
    assert dataset["sample_group_index"] == steps * BATCH_SIZE, dataset
    state = {
        "metadata": metadata,
        "dataset": dataset,
        "model": read_dcp(checkpoint / "model", root / f"{save.name}-{steps}-model.pt")["model_state"]["model"],
        "optimizer": read_dcp(checkpoint / "optimizer", root / f"{save.name}-{steps}-optimizer.pt"),
        "scheduler": read_dcp(checkpoint / "lr_scheduler", root / f"{save.name}-{steps}-scheduler.pt"),
    }

    optimizer = state["optimizer"]["optim_state"]["optim"]["state"]
    assert optimizer, "Adam state was not saved"
    assert all(int(values["step"].item()) == steps for values in optimizer.values())
    assert any(values["exp_avg"].abs().sum().item() > 0 for values in optimizer.values())
    scheduler = state["scheduler"]["lr_scheduler_state"]["lr_scheduler"]
    assert scheduler["last_epoch"] == steps, scheduler
    assert scheduler["lr_decay_steps"] == NUM_ROLLOUTS, scheduler
    return state


def execute(root: Path) -> dict:
    # Ray's launcher dependencies are unnecessary for the standalone CPU fixtures.
    from miles.utils.external_utils import command_utils

    backend = command_utils.default_config().create_backend()
    checkpoint, _ = create_fixture(root)
    initial_model = load_initial_model(checkpoint)
    initial = initial_model.state_dict()
    parameter_names = set(dict(initial_model.named_parameters()))
    full, split = root / "uninterrupted", root / "resumed"
    jobs = [("uninterrupted", full, None, None), ("interrupted", split, None, 2), ("resumed", split, split, None)]
    metrics = {}
    interrupted_state = None
    for name, save, load, stop_after in jobs:
        log = root / f"{name}.log"
        backend.execute_train(
            train_args=train_arguments(root, save=save, load=load, stop_after=stop_after)
            + f" > {shlex.quote(str(log))} 2>&1",
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=None,
            job_lifetime="launcher",
            extra_env_vars={"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
        )
        metrics[name] = read_metrics(
            log,
            expected_steps=[0, 1, 2, 3] if name == "uninterrupted" else [0, 1] if name == "interrupted" else [2, 3],
        )
        if name == "interrupted":
            interrupted_state = verify_checkpoint(root, split, steps=2)
    resume_log = (root / "resumed.log").read_text()
    for message in ("Loaded model from", "Loaded optimizer from", "Loaded LR scheduler from", "load metadata from"):
        assert message in resume_log, message
    baseline = verify_checkpoint(root, full, steps=NUM_ROLLOUTS)
    resumed = verify_checkpoint(root, split, steps=NUM_ROLLOUTS)
    for section in ("model", "optimizer", "scheduler", "dataset"):
        assert_state_close(baseline[section], resumed[section], path=section)
    router_names = [name for name in initial if ".router." in name]
    assert router_names
    for name in router_names:
        torch.testing.assert_close(resumed["model"][name], initial[name], rtol=0, atol=0)
    changed = [
        name
        for name in parameter_names
        if name in resumed["model"] and not torch.equal(initial[name], resumed["model"][name])
    ]
    assert any("self_attn" in name for name in changed), changed
    assert any("experts" in name for name in changed), changed
    assert interrupted_state is not None
    assert interrupted_state["optimizer"] != {}, "optimizer checkpoint is empty"
    report = {
        "status": "passed",
        "num_gpus": NUM_GPUS,
        "precision": "bf16",
        "devices": [torch.cuda.get_device_name(index) for index in range(NUM_GPUS)],
        "torch_version": torch.__version__,
        "num_rollouts": NUM_ROLLOUTS,
        "changed_parameters": changed,
        "frozen_router_parameters": router_names,
        "metrics": metrics,
        "final_metadata": resumed["metadata"],
        "resume_comparison": {"rtol": 1e-5, "atol": 1e-6},
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    root = Path(os.environ.get("MILES_DIFFUSION_SFT_TEST_DIR") or tempfile.mkdtemp(prefix="miles-diffusion-sft-"))
    assert not (root / "resumed").exists(), f"use a fresh artifact directory: {root}"
    print(f"DiffusionGemma functional artifacts: {root}", flush=True)
    execute(root)
    print(f"DiffusionGemma functional SFT passed: {root / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
