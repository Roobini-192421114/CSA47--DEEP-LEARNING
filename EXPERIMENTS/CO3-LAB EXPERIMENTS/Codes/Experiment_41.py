# Experiment 41: Cats vs dogs image classification demonstration (synthetic feature data)
# A locally runnable demonstration of binary image-classification workflow.
import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
rng=np.random.default_rng(41); X=rng.normal(size=(600,24)); y=np.repeat([0,1],300)
X[y==0,:8]+=1.2; X[y==1,8:16]+=1.2
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=41,stratify=y)
m=MLPClassifier(hidden_layer_sizes=(32,),max_iter=250,random_state=41); m.fit(Xtr,ytr); pred=m.predict(Xte)
print('Demo accuracy:',accuracy_score(yte,pred),'(synthetic features; not actual image data)')
ConfusionMatrixDisplay.from_predictions(yte,pred,display_labels=['Cat','Dog']); plt.title('Cats vs dogs workflow demo'); plt.tight_layout(); plt.savefig('../Outputs/41_cat_dog_demo.png')
