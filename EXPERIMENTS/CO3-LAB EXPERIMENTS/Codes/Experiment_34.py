# Experiment 34: Handwritten digit recognition (sklearn digits dataset)
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
X,y=load_digits(return_X_y=True); X=X/16.0
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=34,stratify=y)
model=MLPClassifier(hidden_layer_sizes=(64,),max_iter=350,random_state=34); model.fit(a,c); pred=model.predict(b)
print('Test accuracy:',accuracy_score(d,pred))
ConfusionMatrixDisplay.from_predictions(d,pred); plt.title('Digit recognition confusion matrix'); plt.tight_layout(); plt.savefig('../Outputs/34_digit_confusion_matrix.png')
