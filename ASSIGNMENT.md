# Assignment 3: Small multimodal experiments

**Draft for instructor review. 20 marks. All work is in one Google Colab notebook.**

Use the supplied data and settings. Explain what you observe, including failures. There is no accuracy target and no hyperparameter search. Submit `a3_<rollno>.ipynb` with outputs visible and a short AI-use disclosure. No website, native application, paid API, or external deployment is required.

Select **Runtime → Change runtime type → T4 GPU**. Complete each `YOUR CODE` function and run its section. The provided helpers handle downloading, preprocessing, plotting and training. Keep the fixed splits and seeds. Do not train on test examples.

## Q1. Match images and captions (4 marks)

Use the four paired vectors and temperature in `data/clip_vectors.json`. These are authored numerical examples, not outputs from a pretrained model. Row i in each matrix is a matching pair.

**(a) Implement the symmetric CLIP loss (2 marks).** Normalize each vector to unit length, construct the image–text dot-product matrix, and divide it by temperature. Compute the average negative log probability of the diagonal once across rows and once across columns; average these two losses. Use tensor operations and `torch.logsumexp`. Do not call a ready-made cross-entropy or contrastive-loss function inside your implementation.

**(b) Inspect the matches (1 mark).** Print the similarity matrix and the loss. Repeat after applying the supplied permutation only to the text vectors, while still treating diagonal entries as the targets.

**(c) Explain (1 mark).** In two or three sentences, explain why the diagonal represents the matching pairs and why the second experiment changes the loss. This implements a CLIP-style objective; it does not reproduce EmbeddingGemma's training procedure.

## Q2. Find a picture with words (4 marks)

Use the frozen `google/embeddinggemma-2` checkpoint and supplied 17-image gallery. Credits identify photographs and AI-generated teaching images. Embed the images once using the provided setup; captions are display metadata and must not be used instead of images when building the index.

**(a) Implement search (2 marks).** Embed a text query, normalize query and image vectors, compute cosine similarities, and return the five highest-scoring image indices and their scores.

**(b) Make a small app in Colab (1 mark).** Connect your function to the supplied text box, button and image gallery. Change the query without re-embedding the gallery.

**(c) Try it (1 mark).** Save outputs for four queries: two straightforward descriptions, one query about a visual detail, and one query for something absent from the gallery. Describe one limitation in two or three sentences. A nearest result need not be a correct match, and cosine similarity is not a calibrated probability.

## Q3. A small VLM before and after adaptation (8 marks)

Use `HuggingFaceTB/SmolVLM-256M-Instruct`. We provide model loading and image preprocessing. Run Q2 and Q3 sequentially, releasing the embedding model before loading the VLM.

**(a) Forward pass and generation (2 marks).** Complete the inference function. On three supplied photographs, run a single forward pass, print the logits' shape, and generate a short answer. Explain in two sentences why a forward pass and generating a complete answer are different operations.

**(b) Before adaptation (2 marks).** Ask the model to read the supplied handwritten digits. Use the exact provided prompt and deterministic greedy generation. Evaluate all 100 fixed test images. Report exact one-digit accuracy and the fraction of outputs that contain exactly one digit. Extra text counts as an invalid response; do not silently extract a digit from a sentence.

**(c) Adapt and compare (3 marks).** Run the supplied training helper once: 200 training images, 50 validation images, seed 42, 100 optimizer steps, batch size 4, AdamW learning rate 0.0002, zero weight decay, gradient clipping 1.0. Only the image-to-language connector is trainable. Validate every 25 steps and restore the lowest validation answer-token loss (earlier step wins ties). The prompt, image tokens and padding are excluded from the training loss. Do not change these settings based on test results.

Repeat the same test evaluation. Plot training loss, report before/after accuracy and valid-response fraction, and show the first ten test images with true, before and after answers. Keep the same preprocessing, prompt, generation settings and ordering.

**(d) Explain (1 mark).** In three or four sentences, describe what changed, what remained wrong, and whether better answer formatting explains any improvement. Improvement is not required for credit. This small sample is a classroom experiment, not a general VLM benchmark.

## Q4. Correct a concept (4 marks)

The supplied toy pipeline is `image → [is_red, is_circle] → red-circle / other`. The concept scores are **hand-authored illustrative values**, not measured predictions from a trained vision model. The supplied fixed class head receives only the two concept scores.

**(a) Forward pass (1 mark).** Use the provided weights and bias to compute the class score and prediction for each example.

**(b) Intervene (2 marks).** For the designated first example, copy its concept vector, replace only the incorrect `is_red` value with its supplied ground-truth value, and recompute the class prediction. Print the concept vector and prediction before and after. Keep the image and head parameters fixed.

**(c) Explain (1 mark).** Explain why this intervention can change the prediction, and why editing a separately displayed explanation would not necessarily change a black-box classifier. This exercise illustrates the routing of a strict concept bottleneck; it does not establish real-world interpretability or intervention success.

## Before submission

- Use the supplied splits, model revisions, seed and settings.
- Save outputs and short explanations in the notebook.
- Distinguish your measured outputs from authored examples.
- Cite any borrowed code and disclose AI assistance. Be prepared to explain your functions.
- No screenshots of private files or personal data are needed.

Release and deadline will be set by the instructor. The VLM settings and runtime estimates remain provisional until the instructor pilot is complete.
