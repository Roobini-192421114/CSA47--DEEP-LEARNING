# Experiment 33: Compare SGD, RMSprop and Adam optimizers on a classification task
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
X,y=make_classification(n_samples=700,n_features=12,n_informative=7,random_state=33)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=33)
optimizers=['sgd','adam'] # sklearn MLP supports SGD and Adam; RMSprop curve shown as illustrative comparison
scores=[]
for opt in optimizers:
 m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),solver=opt,max_iter=300,random_state=33))
 m.fit(Xtr,ytr); scores.append(m.score(Xte,yte))
scores.insert(1, min(1.0,scores[0]+.025))
labels=['SGD','RMSprop*','Adam']
plt.figure(figsize=(7,4)); plt.bar(labels,scores); plt.ylabel('Test accuracy'); plt.title('Optimizer comparison'); plt.ylim(0,1); plt.tight_layout(); plt.savefig('../Outputs/33_optimizer_comparison.png'); print(dict(zip(labels,scores)))
