import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
X,y=make_classification(n_samples=900,n_features=12,n_informative=7,n_redundant=3,random_state=16); a,c,b,d=train_test_split(X,y,test_size=.25,random_state=16,stratify=y); sc=StandardScaler().fit(a); a=sc.transform(a); c=sc.transform(c)
m=MLPClassifier(hidden_layer_sizes=(48,24),max_iter=500,random_state=16,early_stopping=True).fit(a,b); pred=m.predict(c); acc=accuracy_score(d,pred)
ConfusionMatrixDisplay(confusion_matrix(d,pred)).plot(); plt.title(f'Synthetic healthcare classifier (accuracy={acc:.2f})'); plt.tight_layout(); plt.savefig('80_output.png'); print('Accuracy:',acc)
