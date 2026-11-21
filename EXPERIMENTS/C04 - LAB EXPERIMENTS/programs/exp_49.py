import numpy as np
from common import rnn_features, fit_readout, sigmoid

print("EXPERIMENT 49: RNN SENTIMENT CLASSIFICATION")
reviews=["good product", "excellent quality", "happy with purchase", "bad product", "poor quality", "not happy"]
labels=np.array([1,1,1,0,0,0],dtype=float)
# Convert each review into a small sequence: positive/negative word indicators.
positive={"good":1.0,"excellent":1.0,"happy":1.0,"quality":0.2,"purchase":0.2}
negative={"bad":-1.0,"poor":-1.0,"not":-0.8,"product":0.1}
seq=[]
for review in reviews:
    seq.append([positive.get(w,negative.get(w,0.0)) for w in review.split()])
features=rnn_features(seq,hidden=6,seed=4)
weights=fit_readout(features,labels)
scores=sigmoid(features@weights[:-1]+weights[-1])
for review,score,label in zip(reviews,scores,labels):
    print(f"{review:22s} -> {'Positive' if score>=0.5 else 'Negative'} (score={score:.2f}, actual={int(label)})")
print("Note: Small demonstration dataset; use a larger labelled review dataset for real evaluation.")
