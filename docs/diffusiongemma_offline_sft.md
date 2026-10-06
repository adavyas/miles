# DiffusionGemma offline SFT (FSDP2)

This recipe trains from fixed, labeled conversations using Miles' FSDP2 backend.
There is no SGLang engine or generation step. Earlier turns are clean context;
only the final assistant turn is the denoising target. Encoder AR loss additionally
covers all valid adjacent tokens in the clean sequence.

```bash
python scripts/run_diffusiongemma_26b_a4b_fsdp_sft.py \
  --hf-checkpoint /models/diffusiongemma-26B-A4B-it \
  --data-path /data/sft.jsonl --output-dir /runs/diffusion-sft \
  --num-gpus-per-node 8
```

Use the existing Miles training image and a local HF checkpoint on every worker.
The source is tested with the repository's Transformers 5.12.1 pin and with
5.15.0. No dependency upgrade is required. The launcher submits via the existing
`command_utils`/Ray path through serial `train.py`. Async prefetch is avoided so
dataset checkpoints do not include an untrained next batch. Multi-node runs need the usual preconfigured Miles
cluster and shared paths. Hardware capacity and throughput are not established
by this recipe; measure them with your actual batch and sequence lengths.

Example JSONL row:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4."}]}
```

The adapter uses the checkpoint's chat template and token offsets to preserve
rendered tokens and exclude the final assistant header. It includes the final
turn closure in supervision. Text-only `system`/`user`/`assistant` messages are
supported; tools, images, empty final answers, and content-dependent headers are
rejected. A custom stable template can be supplied with `--chat-template-path`.
Do not set `--apply-chat-template`: the adapter renders the conversation itself.

## Training contract

- One shared parameter stack runs the clean causal encoder and noisy decoder.
  Layer modules are called normally so FSDP hooks and checkpoint recomputation
  execute. Encoder KV remains in the gradient graph. HF decoder parameter keys
  and separate encoder/decoder layer scalar buffers are preserved; vision weights
  are not loaded into the text trainer.
- Responses are filled to a whole checkpoint-defined canvas with EOS. This fill
  is supervised; batch padding is excluded. One block is sampled uniformly per
  example per microbatch. Only that block is decoded, which is equivalent to the
  selected-block objective with block-diagonal noisy-canvas attention.
- Each example gets a uniform timestep in `[epsilon, 1]`; selected canvas tokens
  are independently replaced with uniformly sampled vocabulary IDs at that
  probability. The objective supervises all canvas positions, with same-position
  CE and no inverse-time weighting.
- Decoder attention sees clean keys strictly before its block and its own whole
  noisy canvas. Sliding visibility is anchored to the clean block start. The
  decoder cannot see its clean target or future clean response blocks.
- A first no-grad decoder pass always runs on all ranks. Its detached logits
  provide soft embeddings in the gradient-bearing second pass, gated by a
  per-example Bernoulli draw. Default self-conditioning probability is 0.5.
- Loss is `diffusion CE + encoder_loss_weight * encoder AR CE`, default weight
  1. Each term has its own token denominator, summed across the entire optimizer
  step and all DP ranks. Backward scaling compensates FSDP's gradient averaging.
  Routers are frozen by default; dense layers and experts train. Use
  `--no-diffusion-freeze-router` to train router parameters too.
- Random batches derive from seed, restored optimizer step, microbatch index,
  and DP rank. Existing native FSDP checkpoints save model, optimizer, scheduler,
  and counters. Exact continuation requires the same world size, batch schedule,
  dataset order, seed, and training flags. Resume uses `--load` (already included
  by the launcher).

Initial supported mode: pure DP/FSDP2, fixed divisible batches, SDPA, full-parameter
text SFT, non-reentrant activation checkpointing. Startup validation rejects RL,
reference/teacher losses, routing/sampling replay, LoRA, dynamic batch schedules,
and rollout evaluation. SGLang weight synchronization and HF serving export are
outside this offline recipe; saves use Miles' native distributed checkpoints.

## Validation

CPU checks (model/objective only; the macOS host lacks the full CUDA worker image):

```bash
python -m pytest --noconftest tests/fast/backends/diffusion_gemma -q
python -m pytest --noconftest tests/manual/launch_scripts/test_py_launch_scripts.py -k diffusiongemma -q
```

These cover HF forward parity, self-conditioning, encoder KV gradients, mixed
precision with actual FSDP2 single-rank fake collectives, checkpoint recomputation,
HF weight roundtrip, leakage boundaries, loss normalization, seeded optimizer
continuation, and the launcher snapshot. Fake collectives do not validate
multi-rank communication. The wider launcher suite requires Linux `/proc` and
the matching training environment.

CUDA/native distributed-checkpoint validation, provided but not run on this host:

```bash
MILES_DIFFUSION_TEST_CHECKPOINT_DIR=/tmp/diffusion-sft-smoke \
torchrun --standalone --nproc-per-node=2 -m pytest --noconftest \
  tests/manual/backends/test_diffusiongemma_fsdp_sft.py -q
```

Use a fresh shared checkpoint directory. This exercises BF16 FSDP2 with FP32
master weights, activation checkpointing, two optimizer steps, and native DCP
save/load continuation. Full 26B checkpoint ingestion, a real tokenizer/dataset
run, multi-rank CUDA execution, and throughput remain integration checks on GPU.

Source contracts: [Transformers DiffusionGemma](https://github.com/huggingface/transformers/tree/v5.12.1/src/transformers/models/diffusion_gemma)
and [NVIDIA AutoModel DiffusionGemma](https://github.com/NVIDIA-NeMo/Automodel/tree/03429168a27bfffd10d109d2adf436b49f1c82a7/nemo_automodel/components/models/diffusion_gemma),
with the [DLLM SFT recipe](https://github.com/NVIDIA-NeMo/Automodel/tree/03429168a27bfffd10d109d2adf436b49f1c82a7/nemo_automodel/recipes/dllm).
