# Small multimodal experiments

An ES 667 Deep Learning assignment developed by Nipun Batra, IIT Gandhinagar.

**20 marks · one Google Colab notebook · five small functions to complete.**

[Open the student notebook in Colab](https://colab.research.google.com/github/nipunbatra/dl-colab-assignment/blob/main/assignment3.ipynb) · [Questions and marking breakdown](ASSIGNMENT.md) · [Measured runtime](RUNTIME.md) · [Data and credits](DATA.md)

1. Implement symmetric CLIP loss using basic tensor operations — 4 marks.
2. Build a small image-search app with frozen EmbeddingGemma 2 — 4 marks.
3. Run SmolVLM-256M and compare digit recognition before and after a short connector fine-tune — 8 marks.
4. Correct one concept in an illustrative concept bottleneck — 4 marks.

An instructor reference implementation passed a real Colab T4 pilot on 13 October 2026. Core computation took about 6 minutes 18 seconds, plus setup and notebook interaction. Allow 10–15 minutes of runtime and an estimated 3–5 hours for student coding and explanations. GPU availability and download speeds vary. See the runtime note for measurement scope.

The assignment is ready for instructor review; the instructor will set its release date and deadline.

## Start

Open the Colab link, save a copy to Drive, select **Runtime → Change runtime type → T4 GPU**, and run cells in order. Complete each `YOUR CODE` function before running its section. Use the supplied data and settings. Submit the notebook with measured outputs and short explanations.

The app runs inside Colab on any OS. No native installer, web deployment, paid API or Hugging Face token is required. Colab itself requires a Google account. Save the four static search galleries before moving to Q3; the widget is disabled when its embedding model is released.

## Contents

- `assignment3.ipynb`: student notebook with unimplemented assessed functions and no saved outputs.
- `ASSIGNMENT.md`: questions, marks and submission requirements.
- `data/`: 17-image gallery, fixed MNIST subsets (200/50/100), CLIP vectors and four authored concept examples.
- `support.py`: supplied model loading, preprocessing, plotting and training utilities.
- `requirements.txt`: pinned libraries; Colab supplies PyTorch and NumPy.
- `scripts/build_notebook.py`: regenerates the student notebook.
- `scripts/check_student_release.py`: validates notebook syntax/schema, unanswered stubs, empty outputs, data checksums and release hygiene.

For a local release check, use Python with `nbformat` and `numpy`, then run `python scripts/check_student_release.py`. Students do not need these development tools.

Instructor solutions, outputs and checkpoints are maintained outside this Git repository. No completed assessed functions or executed instructor notebooks belong here. The supplied training loop is scaffolding.

## References

- [Course schedule](https://nipunbatra.github.io/dl-2026/schedule.html) and [Assignment 2](https://nipunbatra.github.io/dl-2026/Assignments/assignment2.html).
- [EmbeddingGemma 2 model](https://huggingface.co/google/embeddinggemma-2) and [multimodal usage](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers).
- [SmolVLM-256M-Instruct model](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct).
- [CLIP paper](https://arxiv.org/abs/2103.00020) and [Concept Bottleneck Models](https://arxiv.org/abs/2007.04612).

## License and attribution

Original code and question text: MIT (see LICENSE). Image and dataset terms remain those listed in DATA.md; the code license does not relicense third-party data. Authored synthetic examples are explicitly labelled.
