# Student data pack

All supplied files are data, with no computed exercise answers or trained models. The pack is approximately 4.54 MB and can be loaded in Colab using NumPy, Pillow, and Python's JSON library.

- `data/digits.npz`: untouched grayscale MNIST pixels (`uint8`, 0–255, 28×28), with `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images`, `test_labels`, and `train_indices`, `val_indices`, `test_indices`. Indices are zero-based original source positions (`int64`); labels are `uint8`.
- `data/digits_provenance.json`: canonical source URLs, compressed-file MD5 and SHA-256 checksums, seed, RNG, and source split names.
- `data/gallery/`: 17 teaching images, `manifest.json`, and the verbatim original `CREDITS.md`. Each manifest entry explicitly identifies `image_kind` as `photograph` or `generated`. The photograph-style retriever remains labelled generated.
- `data/clip_vectors.json`: four distinct hand-authored image/text vector pairs in four dimensions, supplied unnormalized, with `tau=0.1` and permutation `[2,0,3,1]`. Rows align before permutation; the permutation indexes text rows. These are illustrative input data, not CLIP outputs. No similarity, probability, loss, or recovered pairing outputs are supplied.
- `data/cbm.json` and `data/concepts/*.png`: four authored RGB shapes, public ground-truth concepts and task labels, and illustrative binary concept predictions explicitly marked **NOT trained model output**. Concept order is `[is_red, is_circle]`; the task is red-circle versus other. The strict head consumes only concepts, using supplied weights `[1,1]`, bias `-1.5`, and threshold `0`. These parameters are assignment input data. The first example has a single incorrect red bit, allowing a concept correction to change the task decision. No task predictions or intervention results are supplied.
- `data/checksums.json`: SHA-256 for every delivered data file except the checksum file itself.

## MNIST split protocol

The builder checks the four canonical compressed MNIST MD5 checksums and records SHA-256. It verifies IDX magic numbers, exact payload lengths, 60,000 official training examples, 10,000 official test examples, 28×28 image dimensions, and labels 0–9. Local canonical gzip files were reused for this build, with no download needed.

A single NumPy `Generator(PCG64(42))` is used. For each digit in ascending order, shuffle its official training indices, select the first 20 for training and the next 5 for validation, then shuffle its official test indices and select the first 10 for test. Finally shuffle each assembled split in train/validation/test order using the same generator. Thus training has 20 examples per digit, validation 5, and test 10. Training and validation source indices are disjoint. Test indices belong to a separate official source namespace and must not be compared numerically with training indices as if they shared a source. Pixels are never normalized, augmented, resized, or recolored.

## Provenance and reuse terms

Gallery source: `/Users/nipun/git/dl-teaching/attention-followups/clip/lab/clip-mini-gallery`; its published provenance page is <https://nipunbatra.github.io/clip-lab/sources.html>. All original titles, captions, credits, source filenames, source SHA-256 hashes, and crop-panel/quadrant metadata are retained. `original_gallery_sha256` records the input gallery file hash; `sha256` records the delivered copy. Original dimensions and modification descriptions document conversion to RGB, aspect-preserving Lanczos resizing (maximum 512 pixels; no upscaling), and re-encoding.

The inherited gallery credits identify Oxford-IIIT Pet images as CC BY-SA 4.0, coffee (Rachel Michetti), rocket (SpaceX), and Chelsea (Stefan van der Walt) as CC0, and astronaut (NASA) as public domain. These attributions and terms are preserved from the source gallery, not independently re-licensed. Resized Oxford-IIIT Pet copies retain CC BY-SA 4.0; retain attribution and the modification notice when redistributing. See <https://creativecommons.org/licenses/by-sa/4.0/> and the copied credits for source links.

Ten gallery assets are labelled AI-generated teaching images from the OpenAI image tool, dated 29 September 2026, with original composite hashes and crop provenance retained. The source gallery supplies attribution but no explicit redistribution license for these assets; do not infer CC0 or a photographic source from their appearance.

MNIST source files are the canonical dataset files hosted at <https://storage.googleapis.com/cvdf-datasets/mnist/>; dataset attribution: Yann LeCun, Corinna Cortes, and Christopher J. C. Burges, MNIST, derived from NIST handwritten digits. See <https://yann.lecun.com/exdb/mnist/>. The gzip files contain no license text; this pack does not invent an additional license grant.

The paired vectors, colored shapes, concept labels, illustrative concept predictions, and fixed head are newly hand-authored teaching data in `scripts/build_data.py`. They contain no external images or model outputs. No additional license grant is asserted by this builder.

## Rebuild and verification

Requires Python with NumPy and Pillow; no additional packages. The local build used `/opt/homebrew/bin/python3.9` (NumPy 2.0.2, Pillow 11.3.0).

```sh
python scripts/build_data.py --gallery /path/to/clip-mini-gallery --mnist-cache /path/to/mnist-cache
```

Missing canonical gzip files are downloaded to the chosen cache. Use a cache outside delivered `data/` to keep the pack compact. The default gallery path points to the original local teaching project; rebuilding elsewhere requires providing that source gallery. The default cache is `data/.mnist-cache`; cached gzip files are not needed by students. Byte-identical image encodings across different Pillow versions are not guaranteed; the checked-in checksums describe these delivered files. NPZ ZIP timestamps are fixed, and sampling and JSON ordering are deterministic.

Each build validates gallery hashes/dimensions, canonical MNIST checksums/structure, split balance/index uniqueness/disjointness, image dtype/shape/range, vector dimensions/finiteness/nonzero norms, permutation validity, and CBM concept/label dimensions and shape hashes. Run with the same inputs and runtime to reproduce the delivered hashes.
