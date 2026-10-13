# Colab feasibility and runtime

**Observed on 13 October 2026: all four exercises passed on a Google Colab Tesla T4.** This was actual Colab execution using the supplied data, pinned model revisions and provided training helper. Reference answers and raw evaluation outputs remain private.

## Measured core pipeline

| Operation | Wall time |
|---|---:|
| Dependency installation | 23.45 s |
| CLIP loss numerical/gradient check | 0.23 s |
| EmbeddingGemma 2 load/download, 17-image index and four queries | 50.05 s |
| SmolVLM load/download | 7.39 s |
| Forward pass and short generation on three photographs | 4.17 s |
| Baseline generation on 100 test digits | 32.13 s |
| 100 connector-training steps including validation every 25 steps | 250.56 s |
| Adapted generation on the same 100 test digits | 33.08 s |
| Authored concept intervention check | 0.01 s |

Core stages totalled **377.62 s (6 min 18 s)**. Installation is additional. The image index itself took 9.08 s; four text queries took 0.08–0.18 s each after loading. A real click on the notebook widget returned a ranked five-image gallery; its subsequent query computation took 0.053 s, excluding browser rendering.

The student `infer` function also runs an explicit diagnostic forward pass before generation on each example, so its test loops take longer than the generation-only loops above. A follow-up check of the five completed student functions passed; its baseline and adapted evaluations took 43.41 s and 43.15 s and reproduced all 100 respective answers exactly. Notebook plotting, reading and interaction are also additional. Allow **10–15 minutes of machine time** for a clean run; this is a planning allowance, not a measured guarantee. GPU allocation wait time is excluded and can dominate. Model download speed, Colab images and GPU availability vary.

## Resources and fixed settings

The pilot used Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 5.19.0, sentence-transformers 6.1.0, Matplotlib 3.10.0, Pillow 11.3.0, ipywidgets 8.1.7 and a T4 with 15.64 GB GPU memory (decimal GB). The maximum recorded live PyTorch allocation was 1.92 GB across the sequential core stages. This is **not** total process VRAM or reserved-memory accounting. Both models ran in float32; each was released before the other loaded. No quantization, LoRA, paid inference endpoint or Hugging Face token was used.

The VLM has 256,484,928 parameters. Only its 7,077,888 connector parameters were trainable. The fixed 200/50/100 train/validation/test split, answer-only loss and validation-based checkpoint selection are documented in the assignment. Do not tune on test results or treat this small sample as a general benchmark.

## Student effort

**Estimated, not measured with students:** 3–5 hours including implementation, debugging and explanations. The five assessed functions are deliberately small; loading, preprocessing, training and display scaffolding are provided. All work, including the app, stays in Colab. The native OS app idea is outside the required work so that every student has the same environment.

## If a run fails

Confirm that a T4 is connected, rerun setup after a requested restart, and run sections in order. Do not replace float32 with float16 for EmbeddingGemma 2. Keep a saved notebook copy: Colab session files are temporary. If GPU allocation fails, retry later rather than silently treating CPU timing as comparable. An honest failure report and valid analysis are preferable to fabricated outputs.
