import torch, torch.nn as nn
class DecoderLSTM(nn.Module):
    def __init__(self,out_dim,emb_dim,hid_dim,n_layers=4,dropout=0.3,pad_idx=0):
        super().__init__()
        self.embedding=nn.Embedding(out_dim,emb_dim,padding_idx=pad_idx)
        self.rnn=nn.LSTM(emb_dim+hid_dim*2,hid_dim,num_layers=n_layers,dropout=dropout,batch_first=True)
        self.fc=nn.Linear(hid_dim,out_dim); self.drop=nn.Dropout(dropout)
    def forward(self,x,h,c,ctx):
        x=self.drop(self.embedding(x.unsqueeze(1)))
        out,(h,c)=self.rnn(torch.cat([x,ctx],2),(h,c))
        return self.fc(out.squeeze(1)),h,c
