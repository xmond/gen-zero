from pathlib import Path
import torch, math, json, hashlib
from safetensors import safe_open
p=Path('/tmp/clmrepro/heads/best_head.pt')
x=torch.load(p,map_location='cpu',weights_only=True)
print('CLM file',p,'sha256',hashlib.sha256(p.read_bytes()).hexdigest())
print('keys',list(x))
for key,val in x.items():
 if isinstance(val,dict):
  tensors={k:v for k,v in val.items() if isinstance(v,torch.Tensor)}
  if tensors:print('tensor_group',key,'tensors',len(tensors),'parameters',sum(v.numel() for v in tensors.values()))
  elif key in ('cfg','config'):print(key,val)
root=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots'
for p in sorted(root.glob('*/model.safetensors')):
 with safe_open(p,framework='pt',device='cpu') as f:
  keys=list(f.keys());n=sum(math.prod(f.get_slice(k).get_shape()) for k in keys)
 print('Qwen file',p,'tensors',len(keys),'parameters',n,'bytes',p.stat().st_size)
 cfg=json.loads((p.parent/'config.json').read_text())
 print('config',{k:cfg.get(k) for k in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','tie_word_embeddings')})
