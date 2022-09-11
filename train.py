import torch.optim as optim
import torch
import torch.nn as nn 
from tqdm import trange
from tqdm import tqdm
from model import BERT
import pandas as pd
import numpy as np
import mysamples 
from dataset import RecDataset
from embedding import Embedding
from attention import MultiHeadAttention 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
    
def preprocess_data(samples):
    
    userid, qq_item_seq, kandian_item_seq, userprofile  = [], [], [], []
    
    for i in range(len(samples)):
    
        userid.append(int(samples[i].split(",,")[0]))

        qq_item_seq_temp = [int(digit) for digit in samples[i].split(",,")[1].split(",")]
        qq_item_seq.append(qq_item_seq_temp)

        kandian_item_seq_temp = [int(digit) for digit in samples[i].split(",,")[2].split(",")]
        kandian_item_seq.append(kandian_item_seq_temp)

        userprofile_temp = [int(digit) for digit in samples[i].split(",,")[3].split(",")]
        userprofile.append(userprofile_temp)
        
    return userid, qq_item_seq, kandian_item_seq, userprofile

def train_split(userid, qq_item_seq):
    train, train_target, val, val_target = {}, {}, {}, {}
    length_list = []
    user_index = 0
    val_user_index = 0
    vocab_to_int, int_to_vocab = {}, {}
    total_vocab = []
    
    for user in range(len(userid)):
        qq_seq = qq_item_seq[user]
        for vocab in qq_seq:
            total_vocab.append(vocab)
    
    for index, vocab  in enumerate(set(total_vocab)):
        vocab_to_int[vocab] = index+1
        int_to_vocab[index+1] = vocab

  
    for user in range(len(userid)):
        qq_seq = qq_item_seq[user]
        qq_seq_int = []
        for vocab in qq_seq:
            qq_seq_int.append(vocab_to_int[vocab])

        length_list.append(len(qq_seq_int))
        
        # Last 5 items as the training taret sequence
        if user < 0.8*len(userid):
            train[user_index], train_target[user_index] = qq_seq_int[:-5], qq_seq_int[-5:] 
            user_index += 1 
        else:
            val[val_user_index], val_target[val_user_index] = qq_seq_int[:-5], qq_seq_int[-5:]
            val_user_index += 1      
        
    return int_to_vocab, train, train_target, val, val_target

def compute_accuracy():
   
    masked_list, predict_masked_list, isNext_list, predict_isNext_list = [], [], [], []
    for data in val_data_loader:
        # note: below will load the saved model, uncomment if required to load model
        #model = BERT()
        #model.load_state_dict(torch.load("recmodel.bin"0)


        input_ids, seg_ids, masked_tok, masked_pos, isNext = data['input_ids'].to(device), data['seg_ids'].to(device), data['masked_tok'].to(device), data['masked_pos'].to(device), data['isNext'].to(device)
        
        logits_lm, logits_clsf, logits_clsf2 = model(input_ids, seg_ids, masked_pos, device, n_heads, d_k, d_v)
        
    
        
    
        # True masked tokens list 
        lm_max_index = torch.argmax(logits_lm, axis=2)
        for mt in masked_tok:
            for value in mt.cpu().data.numpy().flatten():
                masked_list.append(value)


        # Predicted masked tokens list
        for pmt_value in lm_max_index.cpu().data.numpy().flatten():
            predict_masked_list.append(pmt_value)



        # True isNext list
        for x in isNext:
            if x:
                isNext_list.append(True)
            else:
                isNext_list.append(False)

        
        # Predicted isNext list
        clsf_max_index = torch.argmax(logits_clsf, axis=1)

        for x in clsf_max_index:
            if x==1:
                predict_isNext_list.append(True)
            else: 
                predict_isNext_list.append(False)

    
            
    
    print("Show 10 examples of input:", input_ids[0:10] )
    #print("Input ids:",  input_ids[-1])
    print('true masked tokens list : ',masked_list[0:10])
    print('predicted masked tokens list : ',predict_masked_list[0:10])
    print('isNext : ', isNext_list[0:10])
    print('predicted isNext : ', predict_isNext_list[0:10])
    print( "\n")
    print("----------------------------------------------")
    print("Overall masked tokens accuracy score: ",int(accuracy_score(masked_list,predict_masked_list)*100 ),"%")
    print("Overall isNext accuracy score: ",int(accuracy_score(isNext_list,predict_isNext_list)*100 ),"%")
    print("----------------------------------------------")
   


def evaluation (val_data_loader, device, n_heads, d_k, d_v):
    masked_list, predict_masked_list, isNext_list, predict_isNext_list = [], [], [], []
    model.to(device)
    model.eval()
    total_loss = 0
    for data in val_data_loader:
        input_ids, seg_ids, masked_tok, masked_pos, isNext = data['input_ids'].to(device), data['seg_ids'].to(device), data['masked_tok'].to(device), data['masked_pos'].to(device), data['isNext'].to(device)
        logits_lm, logits_clsf, logits_clsf2 = model(input_ids, seg_ids, masked_pos, device, n_heads, d_k, d_v)
        
        
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tok) # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf2, isNext) # for sentence classification
        loss_clsf1 = criterion(logits_clsf, isNext) 
        
      
        loss = loss_clsf1 + loss_lm
        
        total_loss += loss.detach() 
        
        
    avg_loss = (total_loss/(len(val_data_loader.dataset)))
    
    return avg_loss


def plot_loss(train_losses, val_losses, epochs):
    epochs_plot = range(1, epochs+1)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    axes.plot(epochs_plot, train_losses, 'g', label='Training Loss')
    axes.plot(epochs_plot, val_losses, 'b', label='Validation Loss')
    axes.set(ylabel='Loss')
    axes.set(xlabel='Epochs')
    axes.legend(loc='upper right')
    axes.set_title('Training and Validation Loss')
    #plt.axis([1, 200, 0.6, 5])
    plt.savefig('RecoBERTloss.jpg')
    
    
    
    

    
if __name__ == "__main__":
    
    all_val_losses = []
    all_train_losses = []
    
    maxlen = 100 # maximum of length
    epochs = 1000 # number of epochs
    max_pred = 40  # max tokens of prediction
    n_layers = 1 # number of Encoder of Encoder Layer
    n_heads = 4 # number of heads in Multi-Head Attention
    emb_dim = 128 # Embedding Size
    d_ff = 128*4   # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2 # number of segments dfv
    pos_neg_samples = 5000 # # number of positive and negative samples, 100 means generating 100 pos and 100 neg <------- CHANGE THIS TO YOUR DESIRED POS/NEG SAMPLES
    learn_rate = 0.00001 #Adam optimizer learn rate

    # Open tensen dataset which contains 2k user sequences
    samples = list(open('desen_random_2k.txt', "r").readlines())
   
    userid, qq_item_seq, kandian_item_seq, userprofile = preprocess_data(samples)

    # Split each sequence into two, first half as qq sequence, another half as target sequence 
    int_to_vocab, train, train_target, val, val_target = train_split(userid, qq_item_seq)
    #seq_map = {s: i for i, s in enumerate(set(flat_seq))}
    vocab_size = max(int_to_vocab)+4

    # Define special token for padding, separation, and masking 
    special_token = {'[PAD]': 0, '[CLS]': max(int_to_vocab)+1, '[SEP]': max(int_to_vocab)+2, '[MASK]': max(int_to_vocab)+3}

    # Define embedding
    emb = Embedding(vocab_size, maxlen, emb_dim, n_segments)

    # Generate Positive and Negative Training samples
    batch, input_ls, seg_ls, mastok_ls, maspos_ls, isNext_ls = mysamples.make_pos_neg(train, train_target, special_token, max_pred, int_to_vocab, maxlen, pos_neg_samples)

    # Generate Positive and Negative Validation samples
    val_batch, val_input_ls, val_seg_ls, val_mastok_ls, val_maspos_ls, val_isNext_ls = mysamples.make_pos_neg(val, val_target, special_token, max_pred, int_to_vocab, maxlen, pos_neg_samples)

    # Define Dataset 
    Dataset = RecDataset(input_ids=input_ls, segment_ids= seg_ls, masked_tok=mastok_ls,
                         masked_pos= maspos_ls, isNext= isNext_ls)


    val_Dataset = RecDataset(input_ids=val_input_ls, segment_ids= val_seg_ls, masked_tok=val_mastok_ls, masked_pos= val_maspos_ls, isNext= val_isNext_ls)

    # Define training Dataloader 
    train_data_loader = torch.utils.data.DataLoader(Dataset, batch_size=10, num_workers=4)

    val_data_loader = torch.utils.data.DataLoader(val_Dataset, batch_size=10, num_workers=4)


    # Define RecoBert model
    model = BERT(vocab_size, maxlen, emb_dim, n_segments, d_ff, n_layers, d_k, d_v, n_heads)
    model.to(device)
    #model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    best_loss = np.inf
    min_train_loss = np.inf
    min_val_loss = np.inf

    train_losses, val_losses = [], []

    # Run training process
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for data in train_data_loader:
            optimizer.zero_grad()
            input_ids, seg_ids, masked_tok, masked_pos, isNext = data['input_ids'].to(device), data['seg_ids'].to(device), data['masked_tok'].to(device), data['masked_pos'].to(device), data['isNext'].to(device)

            logits_lm, logits_clsf, logits_clsf2 = model(input_ids, seg_ids, masked_pos, device, n_heads, d_k, d_v )
           
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tok) # for masked LM
            loss_lm = (loss_lm.float()).mean()
            loss_clsf_avg = criterion(logits_clsf2, isNext) # for sentence classification
            loss_clsf = criterion(logits_clsf, isNext)
        
            loss = loss_lm + loss_clsf
                
            total_loss += loss.detach() 

            loss.backward()
            optimizer.step()



        # Divide the total loss by the length of the positive/negaive examples, scale by 100
        train_loss = (total_loss/(len(train_data_loader.dataset)))

        val_loss = evaluation(val_data_loader, device, n_heads, d_k, d_v)

        train_loss_temp =train_loss.detach().cpu().numpy()
        val_loss_temp =val_loss.detach().cpu().numpy()


        train_losses.append(train_loss_temp)
        val_losses.append(val_loss_temp)

        if (epoch + 1) % 10 == 0:
            print('Epoch', '%01d' % (epoch + 1), 'cost =', '{:.6f}'.format(train_loss))
            print('Epoch', '%01d' % (epoch + 1), 'cost =', '{:.6f}'.format(val_loss))

        if val_loss < best_loss:
            torch.save(model.state_dict(), "recmodel.bin")
            best_loss = val_loss

        if train_loss < min_train_loss:
            min_train_loss = train_loss

        if val_loss < min_val_loss:
            min_val_loss = val_loss


    # Compute accuracy
    compute_accuracy()
    print("Minimum Training Loss :",  min_train_loss)
    print("Minimum Validation Loss:",  min_val_loss) 
    plot_loss(train_losses, val_losses, epochs)
    
    
    





    
    
    
    
    
    
    
    
    



    
    
    
    
    
