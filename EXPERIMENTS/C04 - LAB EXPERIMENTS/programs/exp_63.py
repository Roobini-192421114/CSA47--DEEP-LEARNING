import numpy as np
from common import rnn_features, fit_readout, sigmoid
print("EXPERIMENT 63: CUSTOMER MESSAGE CLASSIFICATION")
messages=["refund please","money returned","bad service","cancel order","great support","thank you","excellent help","very poor"]
labels=np.array([0,0,1,1,2,2,2,1]) # refund, complaint, appreciation
# Numeric sequence tokens: negative/neutral/positive indicators
mapping={"refund":-1,"returned":0,"money":0,"bad":-1,"service":0,"cancel":-1,"order":0,
         "great":1,"support":1,"thank":1,"you":1,"excellent":1,"help":1,"very":0,"poor":-1,"please":0}
seq=[[mapping.get(w,0) for w in m.split()] for m in messages]
F=rnn_features(seq,hidden=8,seed=63)
# One-vs-rest linear readout with simple least-squares fitting
Y=np.eye(3)[labels]; W=fit_readout(F,Y)
scores=np.c_[F,np.ones(len(F))]@W
pred=scores.argmax(axis=1)
names=["Refund","Complaint","Appreciation"]
for m,p in zip(messages,pred): print(f"{m:20s} -> {names[p]}")
for c,name in enumerate(names):
    tp=np.sum((pred==c)&(labels==c)); fp=np.sum((pred==c)&(labels!=c)); fn=np.sum((pred!=c)&(labels==c))
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    print(f"{name}: precision={precision:.2f}, recall={recall:.2f}, F1={f1:.2f}")
