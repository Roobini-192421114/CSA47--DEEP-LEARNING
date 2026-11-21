# Experiment 38: CNN-inspired handwritten cheque digit classifier demo
# Uses sklearn's 8x8 digits dataset as a local, lightweight substitute for MNIST.
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X,y=load_digits(return_X_y=True); X=X/16.
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=38,stratify=y)
m=MLPClassifier(hidden_layer_sizes=(100,),max_iter=350,random_state=38); m.fit(Xtr,ytr); pred=m.predict(Xte)
print('Digit accuracy (MLP baseline):',accuracy_score(yte,pred))
fig,ax=plt.subplots(1,5,figsize=(10,2))
for i,a in enumerate(ax): a.imshow(Xte[i].reshape(8,8),cmap='gray'); a.set_title('Pred '+str(pred[i])); a.axis('off')
plt.suptitle('Digit sample predictions'); plt.tight_layout(); plt.savefig('../Outputs/38_digit_predictions.png')
