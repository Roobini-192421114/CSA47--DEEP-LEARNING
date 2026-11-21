# Experiment 43: Normal/abnormal medical-image classification workflow demonstration
# Synthetic feature dataset used because a labelled medical image dataset is not bundled.
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
from sklearn.neural_network import MLPClassifier
X,y=make_classification(n_samples=500,n_features=16,n_informative=8,class_sep=1.1,random_state=43)
a,b,c,d=train_test_split(X,y,test_size=.3,random_state=43,stratify=y)
m=MLPClassifier(hidden_layer_sizes=(32,),max_iter=250,random_state=43); m.fit(a,c); pred=m.predict(b)
print(classification_report(d,pred,target_names=['Normal','Abnormal']))
ConfusionMatrixDisplay.from_predictions(d,pred,display_labels=['Normal','Abnormal']); plt.title('Normal vs abnormal demo'); plt.tight_layout(); plt.savefig('../Outputs/43_medical_classification.png')
