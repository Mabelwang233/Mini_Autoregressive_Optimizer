import math, random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam(params, lr=1e-4, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=3e-5, momentum=0.9, weight_decay=weight_decay)
    elif name == 'lion':
        return Lion(params, lr=2e-4, betas=(0.9, 0.99), weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def bpb_from_loss(loss):           # nats → bits/token
    return loss / math.log(2.0)

def ppl_from_loss(loss):           # nats → perplexity
    return float(math.exp(loss))

def grad_global_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float().view(-1)
            total_sq += float(torch.dot(g, g))
    return total_sq ** 0.5

def param_global_norm(model):
    with torch.no_grad():
        total_sq = 0.0
        for p in model.parameters():
            if p.requires_grad:
                v = p.detach().float().view(-1)
                total_sq += float(torch.dot(v, v))
        return total_sq ** 0.5

# Sharpness proxy via adversarial input-embedding perturbation (FGSM)
def sharpness_proxy(model, x, y, eps=1e-2):
    model.eval()
    emb = model.get_input_embeddings()
    with torch.enable_grad():
        e = emb(x).detach().requires_grad_(True)
        clean = model(inputs_embeds=e, labels=y).loss
        clean.backward()
        adv_e = e + eps * e.grad.sign()
        with torch.no_grad():
            adv = model(inputs_embeds=adv_e, labels=y).loss
    return float((adv - clean).item()), float(clean.item()), float(adv.item())

def eval_sharpness_on_val(model, val_loader, batches=2, eps=1e-2):
    it = iter(val_loader)
    deltas = []
    for _ in range(batches):
        x, y = next(it)
        d, *_ = sharpness_proxy(model, x, y, eps=eps)
        deltas.append(d)
    if len(deltas) == 1:
        return float(deltas[0]), 0.0
    return float(np.mean(deltas)), float(np.std(deltas, ddof=1))
