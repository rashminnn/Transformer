from datasets import load_dataset
from tokenizers import Tokenizer

# load your already built tokenizers
tokenizer_src = Tokenizer.from_file("../tokenizers/tokenizer_en.json")
tokenizer_tgt = Tokenizer.from_file("../tokenizers/tokenizer_si.json")

ds = load_dataset('Helsinki-NLP/opus-100', 'en-si', split='train')

en_lens = [len(tokenizer_src.encode(item['translation']['en']).ids) for item in ds]
si_lens = [len(tokenizer_tgt.encode(item['translation']['si']).ids) for item in ds]

import numpy as np
print(f"EN - max: {max(en_lens)}, 95th percentile: {np.percentile(en_lens, 95):.0f}")
print(f"SI - max: {max(si_lens)}, 95th percentile: {np.percentile(si_lens, 95):.0f}")