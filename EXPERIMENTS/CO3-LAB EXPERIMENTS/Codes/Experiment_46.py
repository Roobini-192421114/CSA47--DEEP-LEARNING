# Experiment 46: Transfer-learning workflow illustration using local sklearn data
# This demonstrates the evaluation pipeline. Real ResNet/EfficientNet weights require TensorFlow and internet/dataset access.
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
X,y=load_digits(return_X_y=True); X=X/16.
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=46,stratify=y)
m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)); m.fit(a,c); score=m.score(b,d)
print('Local baseline accuracy:',score,'| ResNet/EfficientNet transfer learning requires image data and pretrained weights.')
plt.figure(figsize=(5,4)); plt.bar(['Local baseline'],[score]); plt.ylabel('Accuracy'); plt.title('Transfer-learning workflow baseline'); plt.ylim(0,1); plt.tight_layout(); plt.savefig('../Outputs/46_transfer_learning_baseline.png')
