  # mini-optim-autoregressive (SGD vs Adam vs Lion) — Perplexity Focus

A small PyTorch + Hugging Face project to compare **SGD**, **Adam**, and **Lion** on an **autoregressive** LM task, using **perplexity** as the primary metric.  
It also logs **generalization gap** (val_loss − train_loss), **gradient/parameter norms**, and a **sharpness proxy** via input-embedding FGSM.

## Features
- HF-style dataset: `bigcode/the-stack-smol` (Python subset) with a tokenizer (default `flax-community/gpt-neo-125M-code-clippy`)
- Fresh model init from config for fair optimizer comparison
- **Perplexity-first** logging (plus bits-per-token)
- Convergence plots (loss, perplexity, and gap)
- Multi-seed runner for **run-to-run variance**

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run convergence (SGD/Adam/Lion)
```
python -m src.train --cfg configs/default.yaml --task convergence
```

### Outputs:
```
out/
  adam_seed1337/   # logs + best.pt + summary.txt
  sgd_seed1337/
  lion_seed1337/
  convergence_loss.png
  convergence_ppl.png
  convergence_gap_loss.png
```

## Run multi-seed variance summary
```
python -m src.train --cfg configs/default.yaml --task multiseed
```
Writes `out/multi_seed_summary.csv` with mean/std of final val loss and perplexity.

## Resuls and Analysis

see the full write-up in [analysis.md](analysis.md)


## References

- Chen, G., Chen, T., Zhang, H., Narang, S., Gao, J., Zhao, T., & Keutzer, K. (2023). *Symbolic Discovery of Optimization Algorithms*. arXiv:2302.06675. https://arxiv.org/abs/2302.06675  
  > Source of the **Lion** optimizer we reimplemented in `src/optim_lion.py`.

- (2023). *arXiv preprint* arXiv:2306.00204. https://arxiv.org/pdf/2306.00204  
  > Related to optimizer analyses; we took inspiration for parts of our robustness-style checks.

