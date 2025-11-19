import torch, torch.nn as nn
class EncoderBiLSTM(nn.Module):
    def __init__(self,input_dim,emb_dim,hid_dim,n_layers=2,dropout=0.3,pad_idx=0):
        super().__init__()
        self.embedding=nn.Embedding(input_dim,emb_dim,padding_idx=pad_idx)
        self.rnn=nn.LSTM(emb_dim,hid_dim,num_layers=n_layers,bidirectional=True,dropout=dropout,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.hid_dim=hid_dim; self.n_layers=n_layers
    def forward(self,src):
        emb=self.dropout(self.embedding(src))
        return self.rnn(emb)
