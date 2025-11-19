import torch, torch.nn as nn
class Seq2Seq(nn.Module):
    def __init__(self,enc,dec,device):
        super().__init__(); self.enc=enc; self.dec=dec; self.device=device
        self.bridge=nn.Linear(enc.hid_dim*2,dec.hid_dim)
    def forward(self,src,trg,tf=0.5):
        b,trg_len=src.size(0),trg.size(1)
        out=torch.zeros(b,trg_len,self.dec.fc.out_features).to(self.device)
        enc_out,(h,c)=self.enc(src)
        h=torch.cat([h[-2],h[-1]],1); h=torch.tanh(self.bridge(h)).unsqueeze(0)
        c=torch.zeros_like(h)
        x=trg[:,0]
        for t in range(1,trg_len):
            ctx=enc_out.mean(1,keepdim=True)
            pred,h,c=self.dec(x,h,c,ctx)
            out[:,t]=pred
            x=trg[:,t] if torch.rand(1)<tf else pred.argmax(1)
        return out
