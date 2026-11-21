# Experiment 36: L2 regularization on a classification model
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
X,y=make_classification(n_samples=600,n_features=20,random_state=36)
Xtr,Xte,ytr,yte=train_test_split(X,y,random_state=36)
scores=[]
for alpha in [0.00001,0.001,0.1,1.0]:
 m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(40,),alpha=alpha,max_iter=300,random_state=36))
 m.fit(Xtr,ytr); scores.append(m.score(Xte,yte))
print('L2 alpha scores:',scores)
plt.figure(figsize=(7,4)); plt.plot([.00001,.001,.1,1],scores,marker='o'); plt.xscale('log'); plt.xlabel('Regularization strength (alpha)'); plt.ylabel('Test accuracy'); plt.title('L2 regularization'); plt.tight_layout(); plt.savefig('../Outputs/36_l2_regularization.png')
