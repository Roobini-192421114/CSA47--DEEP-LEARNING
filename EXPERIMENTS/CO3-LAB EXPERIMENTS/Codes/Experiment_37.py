# Experiment 37: Patient-style binary classification and regularization
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
X,y=make_classification(n_samples=800,n_features=10,n_informative=5,class_sep=.8,random_state=37)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=37,stratify=y)
models={}
for label,alpha in [('No regularization',0),('Regularized',.01)]:
 m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),alpha=alpha,max_iter=300,random_state=37))
 m.fit(Xtr,ytr); models[label]=m; print(label, m.score(Xte,yte))
plt.figure(figsize=(6,4)); plt.bar(models.keys(),[m.score(Xte,yte) for m in models.values()]); plt.ylabel('Test accuracy'); plt.title('Patient classifier: regularization comparison'); plt.tight_layout(); plt.savefig('../Outputs/37_patient_regularization.png')
