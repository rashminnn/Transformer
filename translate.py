import torch
from model import build_transformer
from tokenizers import Tokenizer
from config import get_config, get_weights_file_path
from dataset import causal_mask


def translate(sentence, epoch, config, tokenizer_src, tokenizer_tgt, device):
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'], config['seq_len'], config['d_model']
    ).to(device)

    # load specific epoch weights
    state = torch.load(get_weights_file_path(config, f'{epoch:02d}'), weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # encode input
    src_ids = tokenizer_src.encode(sentence).ids
    src_ids = src_ids[:config['seq_len'] - 2]  # truncate if too long
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = (src != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        decoder_input = torch.tensor([[tokenizer_tgt.token_to_id('[SOS]')]]).to(device)

        while decoder_input.size(1) < config['seq_len']:
            decoder_mask = causal_mask(decoder_input.size(1)).to(device)
            out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=-1)
            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # remove SOS token from output before decoding
    output_ids = decoder_input.squeeze(0).cpu().numpy()[1:]
    return tokenizer_tgt.decode(output_ids)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config()

    print(f"Using device: {device}")
    print(f"Loading tokenizers...")

    tokenizer_src = Tokenizer.from_file('../tokenizers/tokenizer_en.json')
    tokenizer_tgt = Tokenizer.from_file('../tokenizers/tokenizer_si.json')

    # test sentences — add or change as you like
    sentences = [
        "I love you.",
        "Where are you going?",
        "What is your name?",
        "Please help me.",
        "He doesn't know anything.",
        "The cat is on the table.",
        "Good morning!",
        "I am very happy today.",
    ]

    # which epochs to compare — change as needed
    epochs_to_check = [0, 5, 10, 15, 19]

    print("\n" + "="*80)

    for sentence in sentences:
        print(f"\nEN: {sentence}")
        print("-" * 60)
        for epoch in epochs_to_check:
            try:
                result = translate(sentence, epoch, config, tokenizer_src, tokenizer_tgt, device)
                print(f"  Epoch {epoch:02d}: {result}")
            except FileNotFoundError:
                print(f"  Epoch {epoch:02d}: not saved yet")
            except Exception as e:
                print(f"  Epoch {epoch:02d}: error — {e}")
        print()

    print("="*80)


if __name__ == "__main__":
    main()