import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch, torch.nn as nn
torch.manual_seed(2); np.random.seed(2)
# Long-range synthetic sequence: target is the value introduced at the first timestep.
N=500; seq_len=30; X=np.random.randn(N,seq_len,1).astype('float32'); y=X[:,0,0]
class Net(nn.Module):
 def __init__(self,kind):
  super().__init__(); self.r=nn.RNN(1,24,batch_first=True) if kind=='RNN' else nn.LSTM(1,24,batch_first=True); self.fc=nn.Linear(24,1)
 def forward(self,x): z=self.r(x)[0]; return self.fc(z[:,-1,:]).squeeze()
results={}
for kind in ['RNN','LSTM']:
 model=Net(kind); opt=torch.optim.Adam(model.parameters(),lr=.015); loss=nn.MSELoss(); xt=torch.tensor(X); yt=torch.tensor(y)
 for ep in range(120): opt.zero_grad(); l=loss(model(xt),yt); l.backward(); opt.step()
 results[kind]=float(loss(model(xt),yt).item())
plt.bar(results.keys(),results.values()); plt.ylabel('MSE'); plt.title('RNN vs LSTM: recall of first-step signal'); plt.tight_layout(); plt.savefig('66_output.png'); print(results)
