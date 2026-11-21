import numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
# Small illustrative parallel dataset; token-level model demonstrates encoder-decoder workflow.
pairs=[('hello','bonjour'),('good','bien'),('morning','matin'),('good morning','bonjour'),('thank you','merci'),('yes','oui'),('no','non'),('cat','chat'),('dog','chien'),('book','livre')]
words=set(' '.join(a+' '+b for a,b in pairs).split()); vocab={'<pad>':0,'<s>':1,'</s>':2}; vocab.update({w:i+3 for i,w in enumerate(sorted(words))}); inv={v:k for k,v in vocab.items()}
def enc(s): return [vocab[w] for w in s.split()]+[2]
def pad(seq): return seq+[0]*(max_len-len(seq))
max_len=max(max(len(enc(a)),len([1]+enc(b))) for a,b in pairs)
X=torch.tensor([pad(enc(a)) for a,b in pairs]); Y=torch.tensor([pad([1]+enc(b)) for a,b in pairs]);
class Model(nn.Module):
 def __init__(self):
  super().__init__(); self.emb=nn.Embedding(len(vocab),24); self.encoder=nn.LSTM(24,48,batch_first=True); self.decoder=nn.LSTM(24,48,batch_first=True); self.out=nn.Linear(48,len(vocab))
 def forward(self,x,y):
  _,(h,c)=self.encoder(self.emb(x)); z,(h,c)=self.decoder(self.emb(y), (h,c)); return self.out(z)
model=Model(); opt=torch.optim.Adam(model.parameters(),lr=.03); lossfn=nn.CrossEntropyLoss(ignore_index=0)
losses=[]
for epoch in range(180):
 opt.zero_grad(); logits=model(X,Y[:,:-1]); loss=lossfn(logits.reshape(-1,len(vocab)),Y[:,1:].reshape(-1)); loss.backward(); opt.step(); losses.append(loss.item())
plt.plot(losses); plt.title('Toy Seq2Seq LSTM training loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.tight_layout(); plt.savefig('65_output.png'); print('Final loss:',losses[-1])
