#train dev test 前の章で作ったものを使う
#full_data_tokened.txtは3つの全部が入ってる
#mozes でtトークナイズ済み
import os 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence,pack_sequence
from sklearn.metrics import accuracy_score
import random
import gensim


class SimpleRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,pre_trained_embedding):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(pre_trained_embedding))
        self.rnn = nn.RNN(embedding_dim,hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim,tagset_size)

        # if pre_trained_embedding:


    def forward(self,sentence,word_seq_lengths):
        embeds = self.word_embeddings(sentence)  
        #手書きでpaddingしたのでpackしてRNNに入れて戻す
        packed_words = pack_padded_sequence(embeds, word_seq_lengths.cpu().numpy(), batch_first=True) 
        hidden = None
        output, h_n = self.rnn(packed_words,hidden)
        #h_nは各時刻の情報がpackされている情報がくる
        #そのまま使うとpackの最後の情報がピンポイントで使えて便利
        tag_space = self.hidden2tag(h_n[0])
 
        tag_scores = F.log_softmax(tag_space,dim=1)
        return tag_scores

#vocab 次元の辞書を作る 80番の内容
def make_vocab(data_path):
    vocab = {}
    with open(data_path) as data:
        for line in data:
            text_data = line.split("\t")[0]
            words = text_data.split()

            for word in  words:
                if word in vocab:
                    vocab[word] +=1
                else:
                    vocab[word] = 1

    #(単語,頻度)の順にタプルで入ってる
    vocab_sorted = sorted(vocab.items(),key=lambda x:x[1])[::-1]

    vocab_with_id = {}
    #idに変換
    for item in vocab_sorted:
        #default_dictを使って一行で書く
        if item[1] >= 2:
            vocab_with_id[item[0]] = len(vocab_with_id.items())+1
        else:
            vocab_with_id[item[0]] = 0

    return vocab_with_id

#train,test,devを読み込む関数
def data_import(mode,vocab_with_id):
    cu_path = os.getcwd()
    name = mode
    data_path = os.path.join(cu_path,"data",name+".txt")

    text_data_list = []
    label_list = []
    with open(data_path) as data:
        for line in data:
            text_data,label  = line.strip().split("\t")
            words = text_data.split()
            text_data_list.append(words)
            label_list.append(label)
    
    return text_data_list,label_list

#単語列 -> id に変換する
#ここでID errorの時に0に処理するプログラムに変える
def prepare_sequence(seqs,with_id,mode):
    bacth_id_list = []
    for seq in seqs:
        idxs = [with_id[w] if (w in with_id) else 0 for w in seq]
        bacth_id_list.append(idxs)
    # print(bacth_id_list)
    # return torch.tensor(idxs, dtype=torch.long)
    return bacth_id_list


#torch.saveはnumpyでもできる
#data => torch.saveでsave それをtorch.loadで読み込むと早くなるかも

#データセットをある程度長さごとに固めてバッチを作ると系列長が揃って処理時間が短くなる
#Allen_NLPを使うと便利

#結局手で書いてしまった
#系列長が違う生でlist > tensor の変換がうまくいかないので0埋めを手動でやった
def sequence2padded_tesnsor(seqs,labels,device):    
    batch_size = len(seqs)
    word_seq_lengths = torch.LongTensor(list(map(len, seqs)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=True).long()
    label_seq_tensor = torch.zeros((batch_size), requires_grad=True).long()
    # mask = torch.zeros((batch_size, max_seq_len), requires_grad=True).byte()
    for idx, (seq, label,seqlen) in enumerate(zip(seqs, labels,word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx] = torch.LongTensor(label)
        # mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
    
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]#長さごとにに並べ替え
    label_seq_tensor = label_seq_tensor[word_perm_idx]

    if device == "cuda":
        word_seq_tensor = word_seq_tensor.to(device)
        label_seq_tensor = label_seq_tensor.to(device)
        word_seq_lengths = word_seq_lengths.to(device)

    return word_seq_tensor,label_seq_tensor,word_seq_lengths

def init_embedding(vocab_with_id,EMBEDDING_DIM):
    from gensim.models import KeyedVectors
    #作ったvocabのindexに紐付ける
    matrix = np.zeros((len(vocab_with_id),EMBEDDING_DIM))
    path = 'data/GoogleNews-vectors-negative300.bin'
    vectors = KeyedVectors.load_word2vec_format(path,binary=True)
    for word,id in vocab_with_id.items():
        if word in vectors.vocab:
            matrix[id] = vectors[word]

    matrix[0] = np.zeros(EMBEDDING_DIM)

    return matrix
        
#fastText使うと一行でやってくれる

def main():
    #paraameter
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 50
    BATCH_SIZE = 32
     
    cu_path = os.getcwd()
    data_path = os.path.join(cu_path,'data','train.txt')
    vocab_with_id = make_vocab(data_path)
    pre_trained_embedding = init_embedding(vocab_with_id,EMBEDDING_DIM)
    
    train_X, train_y = data_import("train",vocab_with_id)

    label_with_id = {"b":0,"t":1,"e":2,"m":3}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SimpleRNN(EMBEDDING_DIM, 
                      HIDDEN_DIM, 
                      len(vocab_with_id), 
                      len(label_with_id),
                      pre_trained_embedding).float()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    model.to(device)



    for epoch in range(10):
        model.train()
        print(epoch)
        train_num = len(train_X)
        batch_size = BATCH_SIZE

        #shuffle
        combined=list(zip(train_X,train_y))
        random.shuffle(combined)
        train_X,train_y=zip(*combined)

        if train_num % batch_size == 0:
            total_batch = train_num // batch_size 
        else:
            total_batch = train_num // batch_size + 1
        
        loss_total = 0
        gold_list = np.array([-1])
        pred_list = np.array([-1])
        
        for batch_id in range(total_batch):
            #それぞれのsizeが違ってdata_loaderの使いかたがわからなかった
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            sentences = train_X[start:end]
            tags = train_y[start:end]

            model.zero_grad()

            sentence_in = prepare_sequence(sentences, vocab_with_id,"X")
            targets = prepare_sequence(tags, label_with_id,"y")
            sentence_in, targets, word_seq_lengths = sequence2padded_tesnsor(sentence_in,targets,device)

            tag_scores = model(sentence_in,word_seq_lengths)
            loss = loss_function(tag_scores, targets)
            
            # print(tag_scores)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                pred_y = torch.argmax(tag_scores,dim=1).cpu().numpy()
            else:
                pred_y = torch.argmax(tag_scores,dim=1).numpy()
            pred_list = np.insert(pred_list,-1,pred_y)
        
            gold_y = np.array(targets.cpu())
            gold_list = np.insert(gold_list,-1,gold_y)
            
        pred_list = np.delete(pred_list,-1)
        gold_list = np.delete(gold_list,-1)
        
        train_acc = accuracy_score(gold_list, pred_list)

        print("train_acc :",train_acc)        
        print("train_loss :",loss_total)

        #========================================================
        #dev
        dev_X, dev_y = data_import("dev",vocab_with_id)
        train_num = len(dev_X)
        batch_size = BATCH_SIZE

        if train_num % batch_size == 0:
            total_batch = train_num // batch_size 
        else:
            total_batch = train_num // batch_size + 1

        dev_loss_total = 0
        gold_list = np.array([-1])
        pred_list = np.array([-1])

        model.eval()
        for batch_id in range(total_batch):
            #それぞれのsizeが違ってdata_loaderの使いかたがわからなかった
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            sentences = dev_X[start:end]
            tags = dev_y[start:end]

            model.zero_grad()

            sentence_in = prepare_sequence(sentences, vocab_with_id,"X")
            targets = prepare_sequence(tags, label_with_id,"y")
            sentence_in, targets, word_seq_lengths = sequence2padded_tesnsor(sentence_in,targets,device)

            tag_scores = model(sentence_in,word_seq_lengths)
            loss = loss_function(tag_scores, targets)
            
            # print(tag_scores)
            dev_loss_total += loss.item()
            
            if torch.cuda.is_available():
                pred_y = torch.argmax(tag_scores,dim=1).cpu().numpy()
            else:
                pred_y = torch.argmax(tag_scores,dim=1).numpy()
            pred_list = np.insert(pred_list,-1,pred_y)
        
            gold_y = np.array(targets.cpu())
            gold_list = np.insert(gold_list,-1,gold_y)
            
        pred_list = np.delete(pred_list,-1)
        gold_list = np.delete(gold_list,-1)
        
        dev_acc = accuracy_score(gold_list, pred_list)

        print("dev_acc :",dev_acc)        
        print("devloss :",dev_loss_total)

if __name__ == '__main__':
    main()
