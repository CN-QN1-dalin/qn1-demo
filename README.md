# QN1 Demo: Small Model + Smart Memory

**0.5B parameters. 88 tok/s. Near-large-model recall accuracy.**

## Watch the Demo

```
python3 qn1_demo.py
```

[![asciicast](https://asciinema.org/a/placeholder.svg)](demo.cast)

## What It Proves

A 0.5B language model can't remember what it said 3 sentences ago. But add **SignalField** — an O(1) cosine-similarity memory — and it correctly recalls facts from earlier in the conversation.

| Test | Raw 0.5B | + SignalField |
|------|:---:|:---:|
| Describe Mars | ✓ | ✓ |
| Atmosphere composition | ✗ (helium!) | ✓ (CO2) |
| Atlas robot | ✓ | ✓ |
| **Recall: which planet?** | ✗ (solar system) | ✓ (Mars) |
| **Recall: who made robot?** | ✗ (Sony) | ✓ (Boston Dynamics) |

**The model is the same. The memory makes the difference.**

## How It Works

```
User asks → SignalField searches memory → finds similar past Q&A
    → injects into prompt → model sees context → answers correctly

Without it:
User asks → model has no context → guesses → wrong
```

## Run It Yourself

```bash
# 1. Get the model (Qwen2.5-0.5B-Instruct MLX 4-bit)
python3 -m mlx_lm.convert \
    --hf-path Qwen/Qwen2.5-0.5B-Instruct \
    -q --q-bits 4 \
    -m /tmp/mlx_qwen_05b_4bit

# 2. Install dependencies
pip install mlx-lm transformers

# 3. Run
python3 qn1_demo.py
```

Requirements: macOS with Apple Silicon (M1/M2/M3/M4).

## QN1 Pro

This demo uses a basic cosine-retrieval memory. **QN1 Pro** adds:

- **SignalField**: Learned attention-guided retrieval (cos ≥ 0.95 vs full attention)
- **归元SSM**: 99% KV cache compression with semantic retrievability
- **灵芽**: Model adaptation with 36% fewer parameters than LoRA

→ [qn1.ai](https://qn1.ai)

## Related

- [RingBuffer: O(1) KV Cache for llama.cpp](https://github.com/CN-QN1-dalin/ringbuffer)
- [PR #23743](https://github.com/ggml-org/llama.cpp/pull/23743)
- [QN1 Benchmarks](https://github.com/CN-QN1-dalin/benchmarks)

MIT License

Made with 🔥 by CN_SJZ-QN1-大林
