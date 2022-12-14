
import torch.nn as nn
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def ScaledDotProductAttention(Q, K, V, attn_mask, d_k):
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
    scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
    attn = nn.Softmax(dim=-1)(scores)
    context = torch.matmul(attn, V)
    return scores, context, attn 
    
   


class MultiHeadAttention(nn.Module):
    def __init__(self,emb_dim, d_k, d_v, n_heads ):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(emb_dim, d_k * n_heads)
        self.W_K = nn.Linear(emb_dim, d_k * n_heads)
        self.W_V = nn.Linear(emb_dim, d_v * n_heads)
        self.emb_dim = emb_dim
    def forward(self, Q, K, V, attn_mask, n_heads, d_k, d_v):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores, context, attn = ScaledDotProductAttention(q_s, k_s, v_s, attn_mask, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        
        #if device == torch.device("cuda"):
        context.to(device)
        L_model = nn.Linear(n_heads*d_v, self.emb_dim).to(device)
        output = L_model(context)
        Ly_norm =  nn.LayerNorm(self.emb_dim).to(device)
        
        return Ly_norm(output + residual), attn # output: [batch_size x len_q x d_model]