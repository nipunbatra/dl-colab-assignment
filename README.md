# Small multimodal experiments

An ES 667 Deep Learning assignment developed by Nipun Batra, IIT Gandhinagar.

**Instructor-review draft.** Four small exercises, all inside Google Colab:

1. Implement the symmetric CLIP loss using basic tensor operations.
2. Search a small image gallery with frozen EmbeddingGemma 2 embeddings.
3. Run a 256M-parameter VLM and compare handwritten-digit predictions before and after a short connector-only fine-tune.
4. Correct one concept in an explicitly illustrative concept bottleneck.

[Assignment questions](ASSIGNMENT.md) · [Data and credits](DATA.md)

The student notebook and measured runtime note are being finalized during the instructor pilot. Do not treat provisional training settings as a validated release until the pilot status is updated.

## Contents

- `data/`: a 17-image gallery, fixed MNIST subsets, numerical CLIP examples and four authored concept examples.
- `support.py`: supplied model-loading, preprocessing, plotting and training utilities.
- `requirements.txt`: pinned library versions; Colab supplies PyTorch and NumPy.

No instructor solutions, executed instructor notebooks or student answers belong in this repository. The provided training loop is scaffolding; assessed functions are completed in the student notebook.

## License and attribution

Original code and question text: MIT (see LICENSE). Image and dataset licenses remain those listed in DATA.md; the code license does not relicense third-party data. Authored synthetic examples are explicitly labelled.
