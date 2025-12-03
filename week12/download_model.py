from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def download_model(ckpt_dir: Path):
    repo_id = "gpt-omni/mini-omni"
    snapshot_download(repo_id, local_dir=ckpt_dir, revision="main")
    
    state_dict = torch.load(ckpt_dir / "lit_model.pth")
    for key in list(state_dict.keys()):
        if "whisper" in key:
            state_dict.pop(key)
        if "transformer.h" in key:
            state_dict[key.replace("transformer.h", "transformer")] = state_dict[key]
            state_dict.pop(key)
        if "transformer.ln_f" in key:
            state_dict[key.replace("transformer.ln_f", "ln")] = state_dict[key]
            state_dict.pop(key)
    
    heads = state_dict['lm_head.weight'].split([152000]+[4160]*7)
    for i, head in enumerate(heads):
        state_dict[f'lm_heads.{i}.weight'] = head
    state_dict.pop('lm_head.weight')

    embs = state_dict['transformer.wte.weight'].split([152000]+[4160]*7)
    for i, emb in enumerate(embs):
        state_dict[f'embeds.{i}.weight'] = emb
    state_dict.pop('transformer.wte.weight')
    
    torch.save(state_dict, ckpt_dir / "model.pth")