import torch 
import torch.nn as nn
import torch.nn.functional as F

#71



class TextClassification(nn.Module):
    def __init__(self,embedding_dim,label_num):
        super().__init__()
        dropout_late = 0.05
        middle_dim = 600
        # self.W = nn.Parameter(torch.randn(embedding_dim, label_num))
        self.W = nn.Parameter(torch.randn(embedding_dim, middle_dim))
        self.middle2tag = nn.Linear(middle_dim, label_num)
        self.bias = nn.Parameter(torch.ones(label_num))
        #バイアスはLInearにデフォであるのでいらない
        self.dropout = nn.Dropout(dropout_late)
        #一個しかないのでdim=0  
        self.softmax = nn.Softmax(dim=0)
        if torch.cuda.is_available():
            print("we can use GPU")
            self.softmax = self.softmax.cuda()
            self.dropout = self.dropout.cuda()
            self.bias = self.bias.cuda()
            self.middle2tag = self.middle2tag.cuda()
        else:
            print("we cannot use GPU")
        
    # #from pytorch official tutorial
    # def init_weights(self):
    #     initrange = 0.5
    #     self.W.weight.data.uniform_(-initrange, initrange)
    #     self.W.bias.data.zero_()

    def forward(self,sent_vector):
        sent_vector = torch.from_numpy(sent_vector).float()
        output = torch.matmul(sent_vector, self.W)
        output = F.relu(output)
        # print(output)
        output = self.middle2tag(output)
        # print(output)
        output = output + self.bias
        output = self.softmax((output))
        
        
        if torch.cuda.is_available():
            sent_vector = sent_vector.cuda()
            output = output.cuda()
        

        return output





        
        