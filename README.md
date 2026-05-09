# English to Sinhala Translator

A transformer model built from scratch using PyTorch to translate English to Sinhala (සිංහල), based on the ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper.

## Model
- 6 encoder/decoder layers, 8 attention heads, d_model=512
- Trained on [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) (en-si) — ~979k sentence pairs
- 20 epochs, final loss ~2.3, trained on NVIDIA RTX 3090

## Files
| File | Description |
|---|---|
| `model.py` | Transformer architecture |
| `train.py` | Training script |
| `dataset.py` | Dataset and tokenization |
| `config.py` | Configuration |
| `translate.py` | Inference |