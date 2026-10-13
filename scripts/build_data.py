#!/usr/bin/env python3
"""Build supplied student data; requires only NumPy and Pillow."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import urllib.request
import zipfile
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MNIST = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}
BASE = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

def digest(path, algorithm='sha256'):
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def read_idx(path, count, images):
    raw = gzip.decompress(path.read_bytes())
    header = 16 if images else 8
    expected = (2051, count, 28, 28) if images else (2049, count)
    assert struct.unpack('>' + 'I' * (header // 4), raw[:header]) == expected
    assert len(raw) == header + count * (784 if images else 1)
    a = np.frombuffer(raw[header:], dtype=np.uint8).copy()
    return a.reshape(count, 28, 28) if images else a

def save_npz(path, arrays):
    # Fixed ZIP timestamps and array order make the archive byte reproducible.
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for key, value in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, value, allow_pickle=False)
            info = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            z.writestr(info, buf.getvalue(), compresslevel=9)

def build(args):
    out = ROOT / 'data'
    gallery = out / 'gallery'
    gallery.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.gallery / 'manifest.json').read_text())
    assert len(manifest) == 17
    for entry in manifest:
        source = args.gallery / entry['file']
        assert digest(source) == entry['sha256'], source
        entry['original_gallery_sha256'] = entry['sha256']
        entry['image_kind'] = 'generated' if 'AI-generated' in entry['credit'] else 'photograph'
        with Image.open(source) as im:
            entry['original_gallery_size'] = list(im.size)
            im = im.convert('RGB')
            im.thumbnail((512, 512), Image.Resampling.LANCZOS)
            target = gallery / entry['file']
            if target.suffix.lower() == '.jpg':
                im.save(target, quality=90, optimize=True, progressive=False)
            else:
                im.save(target, optimize=True)
            entry['size'] = list(im.size)
        entry['sha256'] = digest(target)
        entry['modification'] = 'RGB copy; aspect-preserving Lanczos resize to at most 512 pixels; re-encoded'
    write_json(gallery / 'manifest.json', manifest)
    shutil.copyfile(args.gallery / 'CREDITS.md', gallery / 'CREDITS.md')
    args.mnist_cache.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, md5 in MNIST.items():
        path = args.mnist_cache / name
        if not path.exists():
            urllib.request.urlretrieve(BASE + name, path)
        assert digest(path, 'md5') == md5, 'Canonical MNIST checksum mismatch: ' + name
        provenance.append({'file': name, 'url': BASE + name, 'md5': md5, 'sha256': digest(path)})
    train_x = read_idx(args.mnist_cache / 'train-images-idx3-ubyte.gz', 60000, True)
    train_y = read_idx(args.mnist_cache / 'train-labels-idx1-ubyte.gz', 60000, False)
    test_x = read_idx(args.mnist_cache / 't10k-images-idx3-ubyte.gz', 10000, True)
    test_y = read_idx(args.mnist_cache / 't10k-labels-idx1-ubyte.gz', 10000, False)
    assert set(train_y.tolist()) == set(test_y.tolist()) == set(range(10))
    rng = np.random.Generator(np.random.PCG64(42))
    train_idx, val_idx, test_idx = [], [], []
    for label in range(10):
        idx = rng.permutation(np.flatnonzero(train_y == label))
        train_idx.extend(idx[:20]); val_idx.extend(idx[20:25])
        test_idx.extend(rng.permutation(np.flatnonzero(test_y == label))[:10])
    arrays = {}
    for split, indices, x, y in [('train', train_idx, train_x, train_y), ('val', val_idx, train_x, train_y), ('test', test_idx, test_x, test_y)]:
        indices = rng.permutation(np.array(indices, dtype=np.int64))
        arrays[split + '_images'] = x[indices]
        arrays[split + '_labels'] = y[indices]
        arrays[split + '_indices'] = indices
    save_npz(out / 'digits.npz', arrays)
    write_json(out / 'digits_provenance.json', {'dataset': 'MNIST', 'seed': 42, 'rng': 'NumPy PCG64', 'source_files': provenance, 'index_base': 0, 'train_val_source': 'official train', 'test_source': 'official test'})
    write_json(out / 'clip_vectors.json', {
        'provenance': 'Hand-authored illustrative vectors; not CLIP model outputs.',
        'pair_ids': ['red_circle', 'blue_square', 'green_triangle', 'yellow_star'],
        'image_vectors': [[1, .1, 0, 0], [0, 1, .1, 0], [0, 0, 1, .1], [.1, 0, 0, 1]],
        'text_vectors': [[1, .05, 0, 0], [0, 1, .05, 0], [0, 0, 1, .05], [.05, 0, 0, 1]],
        'tau': .1, 'permutation': [2, 0, 3, 1],
        'pairing': 'Image and text rows with the same index are paired before permutation.',
        'permutation_convention': 'Reorder text rows as text_vectors[permutation]; vectors are supplied unnormalized.'})
    concepts = out / 'concepts'
    concepts.mkdir(exist_ok=True)
    examples = []
    for name, red, circle, prediction in [('red_circle', 1, 1, [0, 1]), ('red_square', 1, 0, [1, 0]), ('blue_circle', 0, 1, [0, 1]), ('blue_square', 0, 0, [0, 0])]:
        im = Image.new('RGB', (128, 128), 'white')
        draw = ImageDraw.Draw(im)
        color = (220, 40, 40) if red else (40, 90, 220)
        (draw.ellipse if circle else draw.rectangle)((24, 24, 104, 104), fill=color)
        target = concepts / (name + '.png')
        im.save(target, optimize=True)
        examples.append({'id': name, 'file': 'concepts/' + target.name, 'concept_labels': [red, circle], 'task_label': int(red and circle), 'concept_predictions': prediction, 'sha256': digest(target)})
    write_json(out / 'cbm.json', {'concepts': ['is_red', 'is_circle'], 'task': 'red-circle vs other', 'task_classes': {'0': 'other', '1': 'red-circle'}, 'provenance': 'Hand-authored illustrative binary concept predictions; NOT trained model output. Shape images rendered with Pillow.', 'head': {'weights': [1, 1], 'bias': -1.5, 'decision_threshold': 0, 'decision_rule': 'class 1 if dot(weights, concepts) + bias >= decision_threshold; otherwise class 0', 'input': 'only the two binary concept values; strict concept bottleneck'}, 'examples': examples})
    validate(out)
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'checksums.json')
    write_json(out / 'checksums.json', {str(p.relative_to(out)): digest(p) for p in files})
    print('Validated: 17 gallery images; balanced MNIST 200/50/100; disjoint train/val indices; 4 paired vectors; 4 CBM shapes. ' + str(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())) + ' bytes.')

def validate(out):
    for e in json.loads((out / 'gallery/manifest.json').read_text()):
        p = out / 'gallery' / e['file']
        assert digest(p) == e['sha256']
        with Image.open(p) as im:
            assert max(im.size) <= 512 and min(im.size) > 0
    with np.load(out / 'digits.npz', allow_pickle=False) as a:
        for split, n, source_n in [('train', 200, 60000), ('val', 50, 60000), ('test', 100, 10000)]:
            x, y, idx = (a[split + '_' + k] for k in ['images', 'labels', 'indices'])
            assert x.shape == (n, 28, 28) and x.dtype == np.uint8
            assert y.shape == idx.shape == (n,) and y.dtype == np.uint8
            assert np.array_equal(np.bincount(y, minlength=10), np.full(10, n // 10))
            assert len(set(idx.tolist())) == n and idx.min() >= 0 and idx.max() < source_n
            assert 0 <= int(x.min()) <= int(x.max()) <= 255
        assert not set(a['train_indices']) & set(a['val_indices'])
    v = json.loads((out / 'clip_vectors.json').read_text())
    for key in ['image_vectors', 'text_vectors']:
        x = np.asarray(v[key]); assert x.shape == (4, 4) and np.isfinite(x).all() and (np.linalg.norm(x, axis=1) > 0).all()
    assert sorted(v['permutation']) == list(range(4)) and v['tau'] > 0
    c = json.loads((out / 'cbm.json').read_text())
    assert len(c['head']['weights']) == len(c['concepts']) == 2
    for e in c['examples']:
        assert len(e['concept_predictions']) == len(e['concept_labels']) == 2
        assert set(e['concept_predictions'] + e['concept_labels']) <= {0, 1}
        assert e['task_label'] == int(all(e['concept_labels']))
        p = out / e['file']; assert digest(p) == e['sha256']
        with Image.open(p) as im:
            x = np.asarray(im); assert x.shape == (128, 128, 3) and x.dtype == np.uint8 and x.min() >= 0 and x.max() <= 255
    assert c['examples'][0]['concept_predictions'] == [0, 1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, default=Path('/Users/nipun/git/dl-teaching/attention-followups/clip/lab/clip-mini-gallery'))
    parser.add_argument('--mnist-cache', type=Path, default=ROOT / 'data/.mnist-cache')
    build(parser.parse_args())
