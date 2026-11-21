import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X,y=make_classification(n_samples=700,n_features=20,n_informative=7,n_redundant=5,random_state=4); a,c,b,d=train_test_split(X,y,test_size=.25,random_state=4)
sc=StandardScaler().fit(a); a=sc.transform(a); c=sc.transform(c); p=PCA(.95).fit(a); ap=p.transform(a); cp=p.transform(c)
vals=[]
for tr,te in [(a,c),(ap,cp)]: m=RandomForestClassifier(n_estimators=120,random_state=4).fit(tr,b); vals.append(accuracy_score(d,m.predict(te)))
plt.bar(['Original','PCA 95%'],vals); plt.ylabel('Test accuracy'); plt.title('Classification before and after PCA'); plt.tight_layout(); plt.savefig('70_output.png'); print(vals, 'components:',p.n_components_)
