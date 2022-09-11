
from attention import MultiHeadAttention
import math
import torch.nn as nn 
import torch


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(emb_dim, d_k, d_v, n_heads)
        #self.pos_ffn = PoswiseFeedForwardNet()
        self.fc1 = nn.Linear(emb_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, emb_dim)


    def forward(self, enc_inputs, enc_self_attn_mask,  n_heads, d_k, d_v):
        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, n_heads, d_k, d_v)   
        #enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.fc2(gelu(self.fc1(enc_outputs))) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
    
    