# DiffusionGemma offline SFT in Miles FSDP2

Scope: fixed text conversations, final assistant turn supervised, random-token
corruption, teacher-forced clean encoder and one uniformly sampled response
canvas per example. No inference engines, RL objective, Megatron port, or LoRA.

1. Add batch preparation and leakage-proof masks. Test unequal prefixes, strict
   clean-block visibility, sliding window anchoring, EOS canvas fill, and seeded
   corruption. Reject discontiguous supervision and multimodal samples.
2. Add a single shared differentiable transformer stack using Transformers
   DiffusionGemma modules. Preserve native checkpoint parameters and distinct
   encoder/decoder scalar buffers. Test encoder/decoder gradients, detached
   self-conditioning, and checkpoint roundtrip with a tiny CPU model.
3. Add separate denoising and encoder AR losses, normalized over the whole
   optimizer step across accumulation and DP. Test target alignment and gradient
   equivalence under splitting. Seed stochastic batches from restored step
   counters and rank.
4. Integrate the specialized path into the existing FSDP actor and checkpoint
   lifecycle. Fail unsupported run modes before model construction. Keep existing
   causal SFT behavior unchanged.
5. Add a fixed-data rollout adapter and Python launcher following Miles script
   conventions, documentation, launcher snapshot, and a CUDA FSDP2 smoke test.
6. Run focused CPU tests and style checks, review the diff, and report CUDA/full
   checkpoint validation separately from tests run on this host.

References: Transformers v5.15.0 DiffusionGemma implementation and NVIDIA
NeMo AutoModel diffusion_gemma / DLLM SFT implementation at
03429168a27bfffd10d109d2adf436b49f1c82a7. Reuse layer math; do not use SGLang's
inference-only forward for autograd training.
