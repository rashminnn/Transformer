import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from dataset import BilingualDataset,causal_mask
from model import build_transformer
from config import get_config,get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from pathlib import Path
from tqdm import tqdm
import warnings

def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx=tokenizer_tgt.token_to_id('[SOS]')
    eos_idx=tokenizer_tgt.token_to_id('[EOS]')

    #precompute the encoder output and reuse it for every step of the decoding
    encoder_output = model.encode(source,source_mask)

    #Initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) #(1,1)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        #build mask for the tearget
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) #(1,seq_len,seq_len)

        #calculate output
        out = model.decode(encoder_output,source_mask,decoder_input,decoder_mask) #(1,seq_len,d_model)

        #get the next token
        prob=model.project(out[:,-1]) 
        
        #select the token with max prob(greedy search)
        _,next_word = torch.max(prob,dim=-1) #(1,)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0) # return a list of token ids

def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_step,writer,num_examples=2):
    model.eval()
    count = 0

    console_width = 80

    #size of the control window
    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input = batch['encoder_input'].to(device) #(batch_size,seq_len)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "validation batch size should be 1"

            model_output = greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_output_text}")

            if count == num_examples:
                break

    if writer:
        writer.add_scalar('validation loss', count, global_step)
        writer.flush()

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config,ds,lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer= WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # ds_raw=load_dataset("helsinki_nlp/opus100", "en-si")
    ds_raw = load_dataset('Helsinki-NLP/opus-100', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    print(ds_raw[0])

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    # kepp 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    # setting up the max length for source and target sentences, for that we calculate the id length of each sentence and keep the maximum one, this is just for information, we will use config['seq_len'] as the max length for both source and target sentences
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))

    print(f'Max length of source sentences: {max_len_src}')
    print(f'Max length of target sentences: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #for val batch size 1 because want to validate sentence one by one 

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):

    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])

    return model


def train_model(config):
    #define the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

    #Tensorboard writer
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    scaler = GradScaler()

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Preloading model weights from {model_filename}")

        state = torch.load(model_filename, weights_only=True)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"),label_smoothing=0.1).to(device) # label smoothing is a regularization technique that replaces the hard 0 and 1(high probability) labels with smoothed values, this can help prevent the model from becoming overconfident and can improve generalization. 


    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) #(batch_size,seq_len)
            decoder_input = batch['decoder_input'].to(device) #(batch_size,seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(batch,1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) #(batch,1,seq_len,seq_len)

            # run tensors through the transformer
            optimizer.zero_grad() 
            with autocast():
                encoder_output = model.encode(encoder_input,encoder_mask) #(batch,seq_len,d_model)
                decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) # (batch,seq_len,d_model)

                projection_output = model.project(decoder_output) #(batch,seq_len,tgt_vocab_size)

                label = batch['label'].to(device) #(batch,seq_len)

                #(batch,seq_len,tgt_vocab_size) -> (batch*seq_len,tgt_vocab_size)
                loss = loss_fn(projection_output.view(-1,tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            #log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            #Backpropagate the loss
            scaler.scale(loss).backward()

            #update the weights
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        model_filename = get_weights_file_path(config,f'{epoch:02d}')
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            },
            model_filename
        )

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    print("DEBUG: Loaded config keys:", config.keys())
    train_model(config)