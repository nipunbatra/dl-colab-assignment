"""Build the student notebook; assessed functions intentionally remain empty."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]

def build():
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(source).strip() + '\n',
                'id': f'cell-{len(cells):03d}'}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    def md(s): add('markdown', s)
    def code(s): add('code', s)
    md('''
    # Assignment 3 · Small multimodal experiments
    **20 marks · Q1 4 / Q2 4 / Q3 8 / Q4 4 · Google Colab**

    Choose **Runtime → Change runtime type → T4 GPU**. Run in order, completing the five small
    `YOUR CODE` functions. Helpers supply preprocessing, display and training. Keep fixed settings;
    no test-set tuning or accuracy target. Record failures honestly. See `RUNTIME.md` for measured Colab timings.

    **Name:** …  **Roll number:** …
    ''')
    md('''
    ## Setup
    Fetch the current public repository and install its pinned requirements before importing models.
    On a fresh Colab runtime this cell runs once; rerunning it refreshes the public files.
    If Colab requests a runtime restart after installation, restart and run setup again.
    ''')
    code('''
    from pathlib import Path
    import subprocess, sys

    REPO = Path('/content/dl-colab-assignment')
    URL = 'https://github.com/nipunbatra/dl-colab-assignment'
    if not REPO.exists():
        subprocess.run(['git', 'clone', '--depth', '1', URL, str(REPO)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
    print('Repository revision:', subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                    str(REPO / 'requirements.txt')], check=True)
    ''')
    code('''
    import json, time, gc
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image
    from IPython.display import display
    sys.path.insert(0, str(REPO))
    from support import (seed_all, sync, load_gallery, show_gallery, load_embedding_model,
                         load_vlm, make_inputs, load_digits, evaluate_digits, train_connector,
                         DIGIT_PROMPT, EMBED_REVISION, VLM_REVISION)
    seed_all(42)
    timings = {}
    print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')
    print('Pinned model revisions:', EMBED_REVISION, VLM_REVISION)
    print((REPO / 'ASSIGNMENT.md').read_text())
    ''')
    md('''
    ## Q1 · Match images and captions — 4 marks
    **(a) 2 marks:** return `(scaled_similarity_matrix, scalar_loss)` using basic torch operations
    and `torch.logsumexp`. Normalize vectors, divide dot products by temperature, and average
    diagonal negative log probabilities in both directions. No ready-made cross-entropy/contrastive loss.
    These are authored vectors, not pretrained embeddings.
    ''')
    code('''
    def clip_symmetric_loss(image_vectors, text_vectors, tau):
        """YOUR CODE: return the scaled matrix and symmetric diagonal loss."""
        raise NotImplementedError('Complete Q1(a)')
    ''')
    code('''
    clip_data = json.loads((REPO / 'data/clip_vectors.json').read_text())
    image_vectors = torch.tensor(clip_data['image_vectors'], dtype=torch.float32)
    text_vectors = torch.tensor(clip_data['text_vectors'], dtype=torch.float32)
    for label, texts in [('paired', text_vectors),
                         ('permuted text', text_vectors[clip_data['permutation']])]:
        matrix, loss = clip_symmetric_loss(image_vectors, texts, clip_data['tau'])
        print(label, '\\nScaled similarity matrix:\\n', matrix, '\\nLoss:', float(loss))
    ''')
    md('''
    **Q1 output and explanation · (b) 1 mark / (c) 1 mark**
    Keep both printed matrices and losses. In 2–3 sentences, explain diagonal targets and the effect
    of permuting only text rows. This CLIP-style objective does not reproduce EmbeddingGemma training.

    *Your explanation:* …
    ''')
    md('''
    ## Q2 · Find a picture with words — 4 marks
    The frozen EmbeddingGemma 2 loader is supplied. **Use float32 on T4**: float16 is unsupported.
    Embed all 17 images once, batch size 1, with 280 soft tokens. Captions are display metadata only.
    [Image API reference](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers).
    ''')
    code('''
    records, gallery_images = load_gallery()
    assert len(gallery_images) == 17
    for start in range(0, len(records), 5):
        show_gallery(gallery_images, range(start, min(start + 5, len(records))), records=records)
    print((REPO / 'data/gallery/CREDITS.md').read_text())
    sync(); started = time.perf_counter()
    embedding_model = load_embedding_model()
    image_embeddings = embedding_model.encode(
        [{'image': image} for image in gallery_images], batch_size=1,
        processing_kwargs={'image': {'max_soft_tokens': 280}},
        normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
    sync(); timings['q2_index_seconds'] = time.perf_counter() - started
    print('Image index:', tuple(image_embeddings.shape), timings)
    ''')
    md('''
    **(a) 2 marks:** embed a query with `embedding_model.encode(query, prompt_name='SearchQuery', convert_to_tensor=True)`;
    normalize query and image vectors, compute cosine similarities, return the five best indices and
    corresponding scores as `(indices, scores)`. Do not re-embed the images.
    ''')
    code('''
    def semantic_search(query, embedding_model, image_embeddings):
        """YOUR CODE: return five ranked image indices and their cosine scores."""
        raise NotImplementedError('Complete Q2(a)')
    ''')
    code('''
    def display_search(query):
        sync(); started = time.perf_counter()
        indices, scores = semantic_search(query, embedding_model, image_embeddings)
        sync(); elapsed = time.perf_counter() - started
        print(f'Query: {query!r} | {elapsed:.2f} seconds')
        # show_gallery expects scores indexed by the original gallery index.
        gallery_scores = np.full(len(gallery_images), np.nan)
        for index, score in zip(indices, scores):
            gallery_scores[int(index)] = float(score)
        show_gallery(gallery_images, indices, scores=gallery_scores, records=records)
        return elapsed

    import ipywidgets as widgets
    from google.colab import output
    output.enable_custom_widget_manager()
    query_box = widgets.Text(description='Query:', placeholder='Describe an image',
                             layout=widgets.Layout(width='80%'))
    search_button = widgets.Button(description='Find five images', button_style='primary')
    search_output = widgets.Output()
    def on_search(_):
        with search_output:
            search_output.clear_output(wait=True)
            if query_box.value.strip():
                display_search(query_box.value.strip())
            else:
                print('Enter a query first.')
    search_button.on_click(on_search)
    display(widgets.VBox([query_box, search_button, search_output]))
    ''')
    md('''
    **Q2 outputs · (b) 1 mark / (c) 1 mark**
    Try the widget, then save four static result galleries below: two straightforward descriptions,
    one visual detail and something absent. Replace the blank strings. Nearest results need not be
    correct; cosine similarity is not a calibrated probability. No deployment is required.
    ''')
    code('''
    queries = ['', '', '', '']  # YOUR QUERIES: descriptions, detail, absent item
    assert all(query.strip() for query in queries), 'Enter all four queries'
    timings['q2_query_seconds'] = [display_search(query) for query in queries]
    ''')
    md('''
    *Your limitation (2–3 sentences):* … Discuss the absent query explicitly: was the requested thing
    present, and what did the nearest results actually show? Do not assume retrieval succeeded.
    ''')
    md('''
    ## Q3 · A small VLM before and after adaptation — 8 marks
    Release Q2's model before loading SmolVLM. Finish Q2 first; its widget stops working after release.
    ''')
    code('''
    search_button.disabled = True
    search_button.on_click(on_search, remove=True)
    del embedding_model, image_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sync(); started = time.perf_counter()
    processor, vlm = load_vlm()
    sync(); timings['q3_load_seconds'] = time.perf_counter() - started
    ''')
    md('''
    **(a) 2 marks:** complete `infer`. Use supplied `make_inputs(processor, model, [image], prompt)`.
    In eval mode and without gradients, call the model's forward pass first. Obtain its logits shape
    and the greedy next token from the last position. Then call `generate` with `do_sample=False`,
    `max_new_tokens=8`; slice off the input tokens before decoding. Return
    `{'answer': str, 'logits_shape': tuple, 'next_token': str}`. Keep these settings for every evaluation.
    ''')
    code('''
    def infer(processor, model, image, prompt):
        """YOUR CODE: forward diagnostics, then deterministic generation of new tokens only."""
        raise NotImplementedError('Complete Q3(a)')

    def predict_digit(processor, model, image, prompt):
        return infer(processor, model, image, prompt)['answer']
    ''')
    code('''
    photo_indices = [i for i, record in enumerate(records)
                     if record['image_kind'] == 'photograph'][:3]
    show_gallery(gallery_images, photo_indices, records=records)
    for i in photo_indices:
        result = infer(processor, vlm, gallery_images[i], 'Describe this photograph briefly.')
        print(records[i]['title'], '| logits shape:', result['logits_shape'],
              '| next token:', repr(result['next_token']), '| answer:', repr(result['answer']))
    ''')
    md('''
    *Forward vs generation (2 sentences):* …

    **(b) 2 marks:** evaluate the fixed 100 test images before adaptation. Exact one-digit accuracy
    and valid-response fraction use the entire stripped answer; extra text is invalid. Do not extract
    a digit from a sentence. The prompt below is fixed.
    ''')
    code('''
    train_images, train_labels = load_digits('train')
    val_images, val_labels = load_digits('val')
    test_images, test_labels = load_digits('test')
    assert (len(train_images), len(val_images), len(test_images)) == (200, 50, 100)
    print('Fixed prompt:', DIGIT_PROMPT)
    sync(); started = time.perf_counter()
    before = evaluate_digits(processor, vlm, test_images, test_labels, predict_digit)
    sync(); timings['q3_before_seconds'] = time.perf_counter() - started
    before_first10 = list(before['answers'][:10])
    print('Before:', {k: before[k] for k in ('accuracy', 'valid_response_fraction')})
    ''')
    md('''
    **(c) 3 marks:** run the helper once. Only the connector is trainable; backbones are frozen.
    Settings: seed 42, 100 steps, batch 4, AdamW lr 0.0002, weight decay 0, clipping 1.0.
    Validate every 25 steps; restore lowest validation answer-token loss (earlier ties win).
    Prompt, image tokens and padding are excluded from loss. Never tune on test results.
    ''')
    code('''
    training = train_connector(processor, vlm, train_images, train_labels, val_images, val_labels,
                               steps=100, batch_size=4, lr=2e-4, seed=42)
    timings['q3_training_seconds'] = training['seconds']
    history = training['history']
    plt.plot([r['step'] for r in history], [r['train_loss'] for r in history], label='Training')
    validation = [r for r in history if 'val_loss' in r]
    plt.plot([r['step'] for r in validation], [r['val_loss'] for r in validation], 'o-', label='Validation')
    plt.xlabel('Optimizer step'); plt.ylabel('Answer-token loss'); plt.legend(); plt.show()
    print('Restored validation loss:', training['best_val_loss'])
    sync(); started = time.perf_counter()
    after = evaluate_digits(processor, vlm, test_images, test_labels, predict_digit)
    sync(); timings['q3_after_seconds'] = time.perf_counter() - started
    after_first10 = list(after['answers'][:10])
    print('Stage      Accuracy    Valid-response fraction')
    for name, result in [('Before', before), ('After', after)]:
        print(f"{name:10} {result['accuracy']:.3f}       {result['valid_response_fraction']:.3f}")
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(test_images[i]); ax.axis('off')
        ax.set_title(f'True: {test_labels[i]}\\nBefore: {before_first10[i]!r}\\nAfter: {after_first10[i]!r}',
                     fontsize=9, wrap=True)
    plt.tight_layout(); plt.show()
    print('Measured timings (seconds):', timings)
    ''')
    md('''
    **Q3 explanation · (d) 1 mark**
    In 3–4 sentences, describe measured changes, remaining errors and whether formatting explains
    improvement. Improvement is not required. This small classroom sample is not a general benchmark.

    *Your explanation:* …
    ''')
    md('''
    ## Q4 · Correct a concept — 4 marks
    `image → [is_red, is_circle] → red-circle / other`. Values are **hand-authored binary teaching
    scores**, not trained vision-model predictions. The supplied fixed AND head receives only these
    two values: weights `[1, 1]`, bias `−1.5`, threshold `0`. Keep image and head fixed.
    **(a) 1 mark:** compute scores and predictions. **(b) 2 marks:** copy the first vector, correct only
    `is_red` using its supplied ground truth, and recompute.
    ''')
    code('''
    cbm = json.loads((REPO / 'data/cbm.json').read_text())
    concepts = torch.tensor([r['concept_predictions'] for r in cbm['examples']], dtype=torch.float32)
    weights = torch.tensor(cbm['head']['weights'], dtype=torch.float32)
    bias = cbm['head']['bias']
    threshold = cbm['head']['decision_threshold']
    concept_images = [Image.open(REPO / 'data' / r['file']).convert('RGB') for r in cbm['examples']]
    show_gallery(concept_images, range(len(concept_images)))
    print('Concept names:', cbm['concepts'], '| authored teaching vectors:', concepts)

    def cbm_head(concepts, weights, bias, threshold):
        """YOUR CODE: return (scores, binary predictions) from only the concept inputs."""
        raise NotImplementedError('Complete Q4(a)')

    def intervene(concept_vector, true_is_red):
        """YOUR CODE: return an independent copy with only is_red corrected."""
        raise NotImplementedError('Complete Q4(b)')
    ''')
    code('''
    scores, predictions = cbm_head(concepts, weights, bias, threshold)
    print('All scores:', scores, '| all predictions:', predictions)
    original = concepts[0].clone()
    corrected = intervene(original, cbm['examples'][0]['concept_labels'][0])
    for label, vector in [('Before', original), ('After', corrected)]:
        score, prediction = cbm_head(vector, weights, bias, threshold)
        print(label, '| concepts:', vector, '| score:', score, '| prediction:', prediction)
    ''')
    md('''
    **Q4 explanation · (c) 1 mark**
    Explain why this intervention can change the prediction, and why editing a separately displayed
    explanation need not change a black-box classifier. This toy routing exercise does not establish
    real-world interpretability or intervention success.

    *Your explanation:* …

    ## Submission
    Save as **`a3_<rollno>.ipynb`** with measured outputs and short explanations visible.
    Check all four sections, four saved search galleries, the loss plot and ten-image comparison.
    Keep supplied splits, seed, revisions and settings. Cite borrowed code and be ready to explain yours.

    **AI-use disclosure and code sources:** …
    ''')
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'name': 'assignment3.ipynb'},
        'accelerator': 'GPU'}, 'cells': cells}

if __name__ == '__main__':
    (ROOT / 'assignment3.ipynb').write_text(json.dumps(build(), indent=1, ensure_ascii=False) + '\n')
