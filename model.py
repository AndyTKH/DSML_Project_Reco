from embedding import Embedding
from encoder import EncoderLayer
import math
import torch.nn as nn 
import torch

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class BERT(nn.Module):
    def __init__(self, vocab_size, maxlen, emb_dim, n_segments, d_ff, n_layers, d_k, d_v, n_heads ):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, maxlen, emb_dim, n_segments)
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos, device, n_heads, d_k, d_v):
        
        
        output = self.embedding(input_ids, segment_ids)
        output.to(device)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask, n_heads, d_k, d_v)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
     
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model] # Output is decided by first token (CLS)
        h_pooled2 = self.activ1(self.fc(torch.mean(output, dim=1))) # Output is decided by the mean of the tokens
        
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]
        logits_clsf2 = self.classifier(h_pooled2)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]


        return logits_lm, logits_clsf, logits_clsf2
