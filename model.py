import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model) # after embedding when we given a number which provide a same vector for that number

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model) # we multiply by sqrt(d_model) to scale the embeddings, which helps with training stability.
    
class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int , seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)

        #create a vector of shape (seq_len, 1) to hold the position indices
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 10000^(2i/d)  =  exp(2i * log(10000) / d)

        #apply sin to even positions 
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) # add a batch dimension
        self.register_buffer('pe',pe) # register pe as a buffer, so it will be saved and loaded with the model, but not updated during training

    def forward(self,x):
        x  = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) # add the positional encoding to the input embeddings, and ensure that the positional encodings are not updated during training
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6) -> None:  # FIXED: added features param, fixed eps (was 10e-6 = 1e-5)
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # FIXED: per-feature shape [features] — learnable scaling parameter which can be multiplied with the normalized output to allow the model to learn the optimal scale for the normalized activations.
        self.bias = nn.Parameter(torch.zeros(features)) # FIXED: per-feature shape [features] — learnable bias parameter which can be added to the normalized output to allow the model to learn the optimal shift for the normalized activations.

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)

        return self.alpha*(x-mean)/(std+self.eps) + self.bias 
    
class FeedForwardBlock(nn.Module):

    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) # W2 and B2
    
    def forward(self,x):
        #(batch,seq_len,d_model) ---> (batch,seq_len,d_ff) ---> (batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model % h == 0, "d_model must be divisible by h"
         
        self.d_k = d_model // h # dimension of each head
        self.w_q = nn.Linear(d_model,d_model) # Wq
        self.w_k = nn.Linear(d_model,d_model) # Wk
        self.w_v = nn.Linear(d_model,d_model) # Wv
         
        self.w_o = nn.Linear(d_model,d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod # scores = MultiHeadAttention.attention(q, k, v, mask) we can use like this 
    def attention(query,key,value,mask,dropout=nn.Dropout):
        d_k = query.shape[-1]
        # (batch,h,seq_len,d_k) @ (batch,h,d_k,seq_len) ---> (batch,h,seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k) 
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask==0,-1e4) 
        attention_scores = attention_scores.softmax(dim=-1) #(batch,h,seq_len,seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores # (batch,h,seq_len,d_k) , (batch,h,seq_len,seq_len)

    def forward(self,q,k,v,mask): #the reason to use mask is we want some words not to intereact with other words so we use mask 
        query = self.w_q(q) # (batch,seq_len,d_model) ---> (batch,seq_len,d_model)
        key = self.w_k(k) # (batch,seq_len,d_model) ---> (batch,seq_len,d_model)
        value = self.w_v(v) # (batch,seq_len,d_model) ---> (batch,seq_len,d_model)

        # (batch,seq_len,d_model) ---> (batch,seq_len,h,d_k) ---> (batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2) # before view query look like this query.shape  →  [batch, seq_len, d_model] [32,10,512] imagine if we have 512 chocolate in a single row this reshape into 8 of 64 each
            # "I"        = [a0,a1,a2,a3, a4,a5,a6,a7]
            # "love"     = [b0,b1,b2,b3, b4,b5,b6,b7]
            # "deep"     = [c0,c1,c2,c3, c4,c5,c6,c7]
            # "learning" = [d0,d1,d2,d3, d4,d5,d6,d7]

            # # BEFORE transpose [1, 4, 2, 4] — organised by WORD:

            # word0 ("I")     → head0[a0,a1,a2,a3]  head1[a4,a5,a6,a7]
            # word1 ("love")  → head0[b0,b1,b2,b3]  head1[b4,b5,b6,b7]
            # word2 ("deep")  → head0[c0,c1,c2,c3]  head1[c4,c5,c6,c7]
            # word3 ("learn") → head0[d0,d1,d2,d3]  head1[d4,d5,d6,d7]

            # # AFTER transpose  [1, 2, 4, 4] — organised by HEAD:

            # head0 → word0[a0,a1,a2,a3]  word1[b0,b1,b2,b3]  word2[c0,c1,c2,c3]  word3[d0,d1,d2,d3]
            # head1 → word0[a4,a5,a6,a7]  word1[b4,b5,b6,b7]  word2[c4,c5,c6,c7]  word3[d4,d5,d6,d7]

            # so we can see after transpose each heads getting an idea about each word in different perspective 
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2) 
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2) 

        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout) 

        #(batch,h,seq_len,d_k) ---> (batch,seq_len,h,d_k) ---> (batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
    
        return self.w_o(x) # (batch,seq_len,d_model) ---> (batch,seq_len,d_model)

class Residualconnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:  # FIXED: added features param to pass to LayerNormalization
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)  # FIXED: pass features
    
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # we apply layer normalization to the input x before passing it through the sublayer, and then add the original input x to the output of the sublayer after applying dropout. This helps to stabilize training and allows the model to learn residual connections effectively.

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, d_model: int) -> None:  # FIXED: added d_model param
        super().__init__() 
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([Residualconnection(d_model, dropout) for _ in range(2)])  # FIXED: pass d_model

    def forward(self,x,src_msk):
        x= self.residual_connections[0](x,lambda x:self.self_attention_block(x, x, x, src_msk))
        x= self.residual_connections[1](x,self.feed_forward_block)

        return x

class Encoder(nn.Module): # in paper there are Nx encoder blocks therefore we can use nn.ModuleList to create a list of encoder blocks

    def __init__(self, layers: nn.ModuleList, d_model: int) -> None:  # FIXED: added d_model param
        super().__init__()
        self.layers=layers
        self.norm = LayerNormalization(d_model)  

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, d_model: int) -> None:  # FIXED: added d_model param
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([Residualconnection(d_model, dropout) for _ in range(3)])  # FIXED: pass d_model
    
    def forward(self,x,encoder_output,src_msk,tgt_msk):
        x = self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,tgt_msk))
        x = self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_msk))
        x = self.residual_connections[2](x,self.feed_forward_block)

        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, d_model: int) -> None:  
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)  

    def forward(self,x,encoder_output,src_msk,tgt_msk):
        for layer in self.layers:
            x=layer(x,encoder_output,src_msk,tgt_msk)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int,vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size) 

    def forward(self,x):
        return self.proj(x) # (batch,seq_len,d_model) ---> (batch,seq_len,vocab_size)    
    
class Transformer(nn.Module):

    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self,src,src_msk):
        src=self.src_embed(src)
        src=self.src_pos(src)

        return self.encoder(src,src_msk)

    def decode(self,encoder_output,src_msk,tgt,tgt_msk):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)

        return self.decoder(tgt,encoder_output,src_msk,tgt_msk)

    def project(self,x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048)->Transformer:
    # creating the embedding layers
    src_embed=InputEmbeddings(d_model,src_vocab_size)
    tgt_embed=InputEmbeddings(d_model,tgt_vocab_size)

    # Creating the positional encoding layers
    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)

    # create the encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout,d_model)  # FIXED: pass d_model               
        encoder_blocks.append(encoder_block)
    
    # create the decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout,d_model)  # FIXED: pass d_model               
        decoder_blocks.append(decoder_block)

    #create the encoder and decoder
    encoder=Encoder(nn.ModuleList(encoder_blocks),d_model)  # FIXED: pass d_model
    decoder=Decoder(nn.ModuleList(decoder_blocks),d_model)  # FIXED: pass d_model

    #create the projection layer
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    #create the transformer
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    # initialize the parameters of the model
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    
    return transformer