"""构建模型"""
import torch.nn as nn
import config
import torch.nn.functional as F

class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel,self).__init__()

        embedding_dim = 200

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)
        self.fc = nn.Linear(config.max_len*embedding_dim,2)
    
    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        #print("input shape: ", input.shape) # torch.Size([512, 50])

        input_embeded = self.embedding(input) #input embeded :[batch_size,max_len,200]

        #print("1 input_embeded shape: ", input_embeded.shape) # torch.Size([512, 50, 200])

        #变形
        input_embeded_viewed = input_embeded.view(input_embeded.size(0),-1)

        #print("2 input_embeded_viewed shape: ", input_embeded_viewed.shape) # torch.Size([512, 10000])

        #全连接
        out = self.fc(input_embeded_viewed)
        #print("out shape: ", out.shape) # torch.Size([512, 2])

        #print("out : ", out)
        #print("out softmax: ", F.log_softmax(out,dim=-1))


        return F.log_softmax(out,dim=-1)


class ImdbModelRNN(nn.Module):
    def __init__(self):
        super(ImdbModelRNN,self).__init__()

        embedding_dim = 200
        hidden_size = 256
        bidirectional = True
        scale = 2 if bidirectional else 1

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)

        self.rnn = nn.RNN(batch_first=True, input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True, nonlinearity="tanh", bidirectional=bidirectional)  
        #torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity="tanh", bias=True)

        self.rnn_fc1 = nn.Linear(config.max_len*hidden_size*scale,hidden_size)
        self.rnn_fc2 = nn.Linear(hidden_size,2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        input_embeded = self.embedding(input)

        rnn_res, h = self.rnn(input_embeded)
        # print("rnn_res: ", rnn_res.shape) #torch.Size([512, 50, 512])

        input_embeded_viewed = rnn_res.contiguous().view(rnn_res.size(0),-1)
        #print("rnn input_embeded_viewed : ", input_embeded_viewed.shape) #torch.Size([512, 25600])

        out1 = self.rnn_fc1(input_embeded_viewed)
        #print("out1 shape: ", out1.shape) #torch.Size([512, 512])

        out2 = self.rnn_fc2(out1)
        #print("out2 shape: ", out2.shape) #torch.Size([512, 2])

        return F.log_softmax(out2,dim=-1)


class ImdbModelLSTM(nn.Module):
    def __init__(self):
        super(ImdbModelLSTM,self).__init__()

        embedding_dim = 200
        hidden_size = 256
        bidirectional = True
        scale = 2 if bidirectional else 1

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)

        self.lstm = nn.LSTM(batch_first=True, input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True,  bidirectional=bidirectional) 

        self.rnn_fc1 = nn.Linear(config.max_len*hidden_size*scale,hidden_size)
        self.rnn_fc2 = nn.Linear(hidden_size,2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        input_embeded = self.embedding(input)

        rnn_res, h = self.lstm(input_embeded)
        # print("rnn_res: ", rnn_res.shape) #torch.Size([512, 50, 512])

        input_embeded_viewed = rnn_res.contiguous().view(rnn_res.size(0),-1)
        #print("rnn input_embeded_viewed : ", input_embeded_viewed.shape) #torch.Size([512, 25600])

        out1 = self.rnn_fc1(input_embeded_viewed)
        #print("out1 shape: ", out1.shape) #torch.Size([512, 512])

        out2 = self.rnn_fc2(out1)
        #print("out2 shape: ", out2.shape) #torch.Size([512, 2])

        return F.log_softmax(out2,dim=-1)



class ImdbModelGRU(nn.Module):
    def __init__(self):
        super(ImdbModelGRU,self).__init__()

        embedding_dim = 200
        hidden_size = 256
        bidirectional = True
        scale = 2 if bidirectional else 1

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)

        self.lstm = nn.GRU(batch_first=True, input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True,  bidirectional=bidirectional) 

        self.rnn_fc1 = nn.Linear(config.max_len*hidden_size*scale,hidden_size)
        self.rnn_fc2 = nn.Linear(hidden_size,2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        input_embeded = self.embedding(input)

        rnn_res, h = self.lstm(input_embeded)
        # print("rnn_res: ", rnn_res.shape) #torch.Size([512, 50, 512])

        input_embeded_viewed = rnn_res.contiguous().view(rnn_res.size(0),-1)
        #print("rnn input_embeded_viewed : ", input_embeded_viewed.shape) #torch.Size([512, 25600])

        out1 = self.rnn_fc1(input_embeded_viewed)
        #print("out1 shape: ", out1.shape) #torch.Size([512, 512])

        out2 = self.rnn_fc2(out1)
        #print("out2 shape: ", out2.shape) #torch.Size([512, 2])

        return F.log_softmax(out2,dim=-1)