"""Provided plumbing. Assessed CLIP loss, retrieval, and intervention are not here."""
from pathlib import Path
import json, random, time, re, copy
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
VLM_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
VLM_REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
EMBED_ID = 'google/embeddinggemma-2'
EMBED_REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
DIGIT_PROMPT = 'What digit is shown? Answer with exactly one digit from 0 to 9.'

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def load_gallery():
    records = json.loads((ROOT/'data/gallery/manifest.json').read_text())
    images = [Image.open(ROOT/'data/gallery'/r['file']).convert('RGB') for r in records]
    return records, images

def show_gallery(images, indices, scores=None, records=None):
    import matplotlib.pyplot as plt
    indices = [int(i) for i in indices]
    fig, axes = plt.subplots(1,len(indices),figsize=(3*len(indices),3.6),squeeze=False)
    for ax,i in zip(axes.flat,indices):
        ax.imshow(images[i]); ax.axis('off')
        title = records[i]['title'] if records else str(i)
        if scores is not None: title += f'\n{float(scores[i]):.3f}'
        ax.set_title(title,fontsize=9)
    plt.tight_layout(); plt.show()

def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    # float16 is NOT supported by EmbeddingGemma 2. Use float32 on T4/CPU.
    return SentenceTransformer(EMBED_ID, revision=EMBED_REVISION,
        config_kwargs={'audio_config':None}, model_kwargs={'torch_dtype':torch.float32},
        device='cuda' if torch.cuda.is_available() else 'cpu')

def load_vlm():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(VLM_ID,revision=VLM_REVISION)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(VLM_ID,revision=VLM_REVISION,
        torch_dtype=torch.float32,attn_implementation='sdpa')
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return processor,model

def digit_image(array):
    # Fixed preprocessing: white paper with a black handwritten digit, RGB.
    return ImageOps.invert(Image.fromarray(array.astype('uint8'))).resize((224,224)).convert('RGB')

def load_digits(split):
    with np.load(ROOT/'data/digits.npz',allow_pickle=False) as d:
        return [digit_image(x) for x in d[f'{split}_images']],d[f'{split}_labels'].astype(int)

def make_inputs(processor,model,images,prompt=DIGIT_PROMPT,answers=None):
    texts=[]; prefix_lengths=[]
    for i in range(len(images)):
        message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        prefix=processor.apply_chat_template(message,add_generation_prompt=True)
        if answers is None: texts.append(prefix)
        else:
            # Mask all user/image/padding tokens. Only assistant answer and EOS train.
            prefix_batch=processor(text=prefix,images=[images[i]],return_tensors='pt')
            prefix_lengths.append(prefix_batch['input_ids'].shape[1])
            message.append({'role':'assistant','content':[{'type':'text','text':str(answers[i])}]})
            texts.append(processor.apply_chat_template(message,add_generation_prompt=False))
    batch=processor(text=texts,images=[[im] for im in images],padding=True,return_tensors='pt')
    if answers is not None:
        labels=batch['input_ids'].clone()
        for i,n in enumerate(prefix_lengths): labels[i,:n]=-100
        labels[batch['attention_mask']==0]=-100
        assert (labels!=-100).any(dim=1).all(), 'No supervised answer tokens'
        batch['labels']=labels
    return {k:v.to(model.device) if hasattr(v,'to') else v for k,v in batch.items()}

def parse_digit(text):
    text=text.strip()
    return int(text) if re.fullmatch(r'[0-9]',text) else -1

def evaluate_digits(processor,model,images,labels,predict_fn):
    # predict_fn is written by the student; generation sees no true answers.
    answers=[predict_fn(processor,model,im,DIGIT_PROMPT) for im in images]
    pred=np.array([parse_digit(s) for s in answers])
    return {'accuracy':float(np.mean(pred==labels)),
            'valid_response_fraction':float(np.mean(pred>=0)),
            'answers':answers,'predictions':pred.tolist(),'labels':labels.tolist()}

def train_connector(processor,model,train_images,train_labels,val_images,val_labels,
                    steps=100,batch_size=4,lr=2e-4,seed=42):
    """Provided training loop; update only the image-to-language connector."""
    seed_all(seed)
    for p in model.parameters(): p.requires_grad_(False)
    params=[]
    for name,p in model.named_parameters():
        if '.connector.' in name:
            p.requires_grad_(True);params.append(p)
    if not params: raise RuntimeError('Expected named connector parameters were not found')
    print('Trainable connector parameters:',sum(p.numel() for p in params),flush=True)
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.0)
    rng=np.random.default_rng(seed);history=[];best_loss=float('inf');best=None
    order=rng.permutation(len(train_images));cursor=0
    def validation_loss():
        model.eval();total=0.;count=0
        with torch.no_grad():
            for start in range(0,len(val_images),batch_size):
                b=make_inputs(processor,model,val_images[start:start+batch_size],
                    answers=val_labels[start:start+batch_size])
                n=int((b['labels'][:,1:]!=-100).sum())
                total+=float(model(**b,use_cache=False).loss)*n;count+=n
        return total/count
    sync();t0=time.perf_counter()
    for step in range(1,steps+1):
        if cursor+batch_size>len(order):order=rng.permutation(len(train_images));cursor=0
        ix=order[cursor:cursor+batch_size];cursor+=batch_size
        # Frozen backbones remain in eval mode, preserving the base inference behavior.
        model.eval()
        batch=make_inputs(processor,model,[train_images[i] for i in ix],answers=train_labels[ix])
        opt.zero_grad(set_to_none=True)
        loss=model(**batch,use_cache=False).loss
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
        row={'step':step,'train_loss':float(loss.detach())}
        if step%25==0 or step==steps:
            row['val_loss']=validation_loss()
            if row['val_loss']<best_loss:
                best_loss=row['val_loss'];best={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            sync();print({**row,'elapsed_s':round(time.perf_counter()-t0,1)},flush=True)
        history.append(row)
    if best is None: raise RuntimeError('No validation checkpoint')
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in best:p.copy_(best[n].to(p.device))
    model.eval();sync()
    return {'history':history,'seconds':time.perf_counter()-t0,
            'best_val_loss':best_loss,'trainable_parameters':sum(p.numel() for p in params)}
