import numpy as np
print("EXPERIMENT 50: CHARACTER-LEVEL RNN TEXT GENERATION")
text="deep learning models learn patterns. "
chars=sorted(set(text))
to_id={c:i for i,c in enumerate(chars)}
ids=np.array([to_id[c] for c in text])
V=len(chars); H=16; rng=np.random.default_rng(2)
Wx=rng.normal(0,0.15,(H,V)); Wh=rng.normal(0,0.15,(H,H)); Wy=rng.normal(0,0.15,(V,H))
bh=np.zeros((H,1)); by=np.zeros((V,1)); h=np.zeros((H,1))
# One-step training on the supplied character sequence, using a simple output-layer update.
for epoch in range(80):
    h[:]=0
    for a,b in zip(ids[:-1],ids[1:]):
        x=np.zeros((V,1)); x[a]=1
        h=np.tanh(Wx@x+Wh@h+bh)
        logits=Wy@h+by; probs=np.exp(logits-logits.max()); probs/=probs.sum()
        target=np.zeros((V,1)); target[b]=1
        err=probs-target
        Wy-=0.08*(err@h.T); by-=0.08*err
seed="deep"
h=np.zeros((H,1))
for c in seed:
    x=np.zeros((V,1)); x[to_id[c]]=1; h=np.tanh(Wx@x+Wh@h+bh)
out=seed; current=to_id[seed[-1]]
for _ in range(45):
    x=np.zeros((V,1)); x[current]=1; h=np.tanh(Wx@x+Wh@h+bh)
    p=Wy@h+by; current=int(np.argmax(p)); out+=chars[current]
print("Training text:",repr(text))
print("Generated text:",repr(out))
print("Vocabulary size:",V)
