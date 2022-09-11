import torch

class RecDataset:
    def __init__(self, input_ids, segment_ids, masked_tok, masked_pos, isNext ):
        self.input = input_ids
        self.segment = segment_ids
        self.masked_tok = masked_tok
        self.masked_pos = masked_pos
        self.isNext = isNext
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, item):
        input_items = self.input[item]
        segment = self.segment[item]
        masked_tok = self.masked_tok[item]
        masked_pos= self.masked_pos[item]
        isNext = self.isNext[item]
        

        return {
            "input_ids": torch.tensor(input_items, dtype=torch.long),
            "seg_ids": torch.tensor(segment, dtype=torch.long),
            "masked_tok": torch.tensor(masked_tok, dtype=torch.long),
            "masked_pos": torch.tensor(masked_pos, dtype=torch.long),
            "isNext": torch.tensor(isNext, dtype=torch.long),
        }