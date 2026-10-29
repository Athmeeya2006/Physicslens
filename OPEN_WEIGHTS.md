# Open-weight evidence

PhysicsLens reasons only on Meta's open-weight models. This page lists exactly what runs where, so anyone can check it.

| Order | Model | Served by | Provider and model id recorded in results |
|---|---|---|---|
| 1 (primary) | **Muse Spark Open** (816B mixture of experts, 42B active) | Baseten, OpenAI-compatible Chat Completions | `"provider": "baseten", "model": "meta-models/muse-spark-open-1026"` |
| 2 (fallback, and fully offline path) | **Muse Glimmer 30B** (Apache 2.0) | our own llama.cpp server | `"provider": "glimmer", "model": "muse-glimmer-30b"` |

The published gallery results so far ran on Muse Glimmer 30B (second section below). Results made on Muse Spark Open carry its label instead.

## Muse Spark Open on Baseten

- Model id `meta-models/muse-spark-open-1026` (the hackathon deployment; Baseten's catalog lists the same weights as `meta-models/muse-spark-open`, Hugging Face id `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`).
- Baseten is one of the hackathon's inference partners, so its usage logs for our key independently show the calls.
- Requests: `reasoning_effort: "low"` on every call (the model defaults to `xhigh`), sampling left at its defaults. The model always reasons; Baseten returns that reasoning in `reasoning_content`, and every result keeps it next to the answer.
- If Baseten fails, the next model tried is Muse Glimmer on our own server (only when one is configured), and the decision trail says so. There is no third option: in demo and production the app then shows a "busy" message instead of reasoning on a closed model (`agent/llm.py`, `provider_chain`).

## Muse Glimmer 30B, self-hosted

### The weights

Downloaded from the official repository [`meta-models/Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF), pinned to revision `b44f1072cf75e918a1b82a3a17b45c4389b076e7` in `scripts/gpu_server/serve.py`. The SHA-256 values are the ones Hugging Face publishes for these files at that revision:

| File | Role | Size (bytes) | SHA-256 |
|---|---|---|---|
| `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` | the model (Q4_K_M) | 16756683904 | `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e` |
| `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | vision encoder (keyframes, track checks) | 1400328928 | `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6` |
| `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` | speculative-decoding drafter (speed only; outputs are the model's) | 1631208128 | `b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef` |

To check a local copy: `sha256sum Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`.

### The runtime

- llama.cpp release `b11936` (official prebuilt CUDA 12.8 binaries from `github.com/ggml-org/llama.cpp/releases`), `llama-server` with the model's own chat template (`--jinja`).
- Served under the alias `muse-glimmer-30b` (`-a muse-glimmer-30b`), bound to localhost, with sampling at the model card's defaults (temperature 1.0, top_p 0.95, top_k 64) and `reasoning_strength: "low"`.
- Hardware: two NVIDIA T4 GPUs (16 GB each).
- The exact command is in `scripts/gpu_server/serve.py` and in the README under "Serve Muse Glimmer".

## How to see it in the results

- Every step of every result records who made it. In each `result.json`, the `trail` items and the `llm_calls` list carry the provider and model id from the table above for reasoning steps (the current gallery shows `"provider": "glimmer", "model": "muse-glimmer-30b"`) (scene, hypotheses, track check, agent decision). Verification steps carry `"model": "simulate-and-compare"`: numerical code, not a language model.
- `llm_calls` also keeps each step's reasoning text as the model returned it, plus any provider switch (`fallbacks`).
- The app shows the same labels on screen: every step of the decision trail names "Muse Spark Open (open weights, served by Baseten)", "Muse Glimmer 30B (open weights)" or the numerical method, with the model's reasoning one click away.

## What is not open weights, and why

- **SAM 3.1** (tracking and segmentation) runs on the Meta Model API.
- **Muse Voice Transcribe** (spoken questions and settings) runs on the Meta Model API.
- **Muse Image** draws requested experiment setups onto the user's own frame. It never reasons about or measures physics.
- **Muse Spark 1.3** was used only during early development; the app refuses to reason on it in demo and production, and none of its results are published.

In demo and production the app refuses to reason on anything but these open-weight models: if none answers it says it is busy instead of switching to a closed model (`agent/llm.py`).

The same open weights are also available from a hosted provider listed in Meta's Glimmer deploy guide (Together AI, model `meta-models/Muse-Glimmer-30B`); a run made there is labeled with that model id and provider.
