
import torch.nn as nn 
import torch

class Embedding(nn.Module):
    def __init__(self, vocab_size, maxlen, emb_dim, n_segments):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, emb_dim)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, emb_dim) # position embedding
        self.seg_embed = nn.Embedding(n_segments, emb_dim)  # segment(token type) embedding
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos =  torch.arange(seq_len, dtype=torch.long)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
            
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
           
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
      
       
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        

        return self.norm(embedding)  