import random
from random import randrange




def make_pos_neg(train, train_target, special_token, max_pred, int_to_vocab, maxlen, pos_neg_samples ):

    batch = []
    positive = negative = 0
    input_ls, seg_ls, mastok_ls, maspos_ls, isNext_ls = [],[],[],[],[] 
    
    while positive != pos_neg_samples or negative != pos_neg_samples:
        

        prob = random.uniform(0,1)
        if prob <0.5:
            tokens_a_index, tokens_b_index = randrange(len(train)), randrange(len(train))
        else:
            tokens_a_index = randrange(len(train))
            tokens_b_index = tokens_a_index
          
        
        tokens_a, tokens_b = train[tokens_a_index], train_target[tokens_b_index]
        
        # For 100 sequence length, we take the last 92 tokens from one sequence of training set, 
        # combine with the 5 tokens from one sequence of training target set, 
        # plus 3 special tokens: 1 [CLS], and 2 [SEP] tokens.   
        #tokens_a = tokens_a[-92:]
        tokens_a = tokens_a[-(maxlen-8):]
        
        input_ids = [special_token['[CLS]']] + tokens_a + [special_token['[SEP]']] + tokens_b + [special_token['[SEP]']]

        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        #MASK LM
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.20)))) # 15 % of tokens in one sentence
        #n_pred = max(1, int(round(len(input_ids) * 0.20)))
        
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != special_token['[CLS]'] and token != special_token['[SEP]'] and token != special_token['[PAD]'] ]
        random.shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            prob = random.uniform(0,1)
            if prob  < 0.8:  # 80%
                input_ids[pos] = special_token['[MASK]'] # make mask
            elif prob < 0.9:  # 10%
                #index = random.randint(0, len(seq_map) - 1) # random index in vocabulary
                #input_ids[pos] = list(seq_map.keys())[index] # replace
                input_ids[pos] = random.randint(1, max(int_to_vocab.keys()))

        
        
        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

    #     # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index == tokens_b_index and positive < pos_neg_samples:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            input_ls.append(input_ids)
            seg_ls.append(segment_ids)
            mastok_ls.append(masked_tokens) 
            maspos_ls.append(masked_pos)
            isNext_ls.append(True)
            positive += 1
        elif tokens_a_index != tokens_b_index and negative < pos_neg_samples:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            input_ls.append(input_ids)
            seg_ls.append(segment_ids)
            mastok_ls.append(masked_tokens) 
            maspos_ls.append(masked_pos)
            isNext_ls.append(False)
            negative += 1
  
    return batch, input_ls, seg_ls, mastok_ls, maspos_ls, isNext_ls

        