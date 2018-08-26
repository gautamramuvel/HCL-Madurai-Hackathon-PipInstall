import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("C://Users//gauta//Desktop//Guvi//CKD//ckd.csv",sep=',', na_values=['?'])
print("Attribute list is {}".format(data.columns.values))

print (" ")
#No of people with chronic kidney disease
n_ckd = len(data[data['class']=='ckd'])

#No of people without chronic kidney disease
n_notckd = len(data[data['class']=='notckd'])

print (" ")

print ("Number of people detected with chronic kidney disease: {}".format(n_ckd))
print ("Number of people not detected with chronic kidney diesease: {}".format(n_notckd))
display(data.describe())

from pandas.tools.plotting import scatter_matrix
scatter_matrix(data, alpha=0.6, figsize=(12, 12), diagonal='kde')
plt.show()

import seaborn as sns
plt.figure(figsize=(30,20))
sns.heatmap(data.corr(), annot=True)
plt.show()


X = pd.DataFrame(data)
fill = pd.Series([X[c].value_counts().index[0]
        if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
        index=X.columns)
new_data=X.fillna(fill) 
print(new_data)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data=new_data.copy()
for items in data:
    if data[items].dtype == np.dtype('O'):
        data[items]=le.fit_transform(data[items])

print(data.dtypes)
print(data)

from sklearn.preprocessing import RobustScaler
target_class = data['class']
features = data.drop('class', axis = 1)
data_robust = pd.DataFrame(RobustScaler().fit_transform(features), columns=features.columns)


from sklearn.decomposition import PCA
pca = PCA()
data_pca = pd.DataFrame(pca.fit_transform(data_robust), columns=data_robust.columns)
print(data_robust.shape)
print(pca.explained_variance_ratio_)


pca = PCA(n_components=11)
pca.fit(data_robust)
reduced_data = pca.transform(data_robust)
reduced_data = pd.DataFrame(reduced_data, columns = ['dim1','dim2','dim3','dim4','dim5','dim6','dim7','dim8','dim9','dim10','dim11'])
print(reduced_data.head())


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reduced_data,target_class, test_size=0.25, random_state=42)


from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from time import time
clf_A=GaussianNB()
clf_B=tree.DecisionTreeClassifier(random_state=0)
clf_C=SVC()
clf_D=RandomForestClassifier()
clf_E=KNeighborsClassifier()
clf_F=MLPClassifier(random_state=1)

for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F]:
    
    print("\n{}: \n".format(clf.__class__.__name__))
    start = time()
  
    clf.fit(X_train, y_train)
    
    end = time()
    print("Trained model in {:.4f} milliseconds".format((end - start)*1000))
    
for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F]:
    y_pred = clf.predict(X_test)
    f1_scorer=f1_score(y_test, y_pred, average=None)
    print ("\n{}: \n".format(clf.__class__.__name__))
    print ("f1 score for test test is {}".format(f1_scorer))
    

from sklearn.metrics import f1_score
from sklearn.svm import SVC
parameters = [{'kernel':['rbf','sigmoid'],'C': [.1,1,10], 'gamma': [0.001,10,1000]},]
clf = SVC()
f1_scorer = make_scorer(f1_score,pos_label=0)
sss = StratifiedShuffleSplit( y_train, n_iter=10, test_size=0.25)
grid_obj = GridSearchCV(clf,parameters,cv = sss,scoring=f1_scorer)
grid_obj = grid_obj.fit(X_train,y_train)
clf = grid_obj.best_estimator_
#print clf
y_pred = clf.predict(X_test)
f1_score_value=f1_score(y_test, y_pred, pos_label=0, average=None) # For testing
print ("F1 Score for test set: {}".format(f1_score_value))
print ("Confusion Matrix is : \n  {} ".format(confusion_matrix(y_test, y_pred)))
target_names = ['class 0', 'class 1']
print (" ")
print ("Classification report is : \n  ")
print(classification_report(y_test, y_pred, target_names=target_names))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
parameters = [{'n_estimators': [10,50,75],
               'max_features': ["auto","sqrt","log2"],'n_jobs': [1]}]
clf = RandomForestClassifier()
f1_scorer = make_scorer(f1_score,pos_label=0)
sss = StratifiedShuffleSplit( y_train, n_iter=10, test_size=0.25)
grid_obj = GridSearchCV(clf,parameters,cv = sss,scoring=f1_scorer)
grid_obj = grid_obj.fit(X_train,y_train)
clf = grid_obj.best_estimator_
y_pred = clf.predict(X_test)
f1_score_value=f1_score(y_test, y_pred, pos_label=0, average=None) # For testing
print ("F1 Score for test set: {}".format(f1_score_value))
print ("Confusion Matrix is : \n  {} ".format(confusion_matrix(y_test, y_pred)))
target_names = ['class 0', 'class 1']
print (" ")
print ("Classification report is : \n  ")
print(classification_report(y_test, y_pred, target_names=target_names))


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
parameters = [{'metric':['minkowski','euclidean'] ,
               'weights':['uniform'],'n_neighbors': [5,15,10]}]
clf = KNeighborsClassifier()
f1_scorer = make_scorer(f1_score,pos_label=0)
sss = StratifiedShuffleSplit( y_train, n_iter=10, test_size=0.25)
grid_obj = GridSearchCV(clf,parameters,cv = sss,scoring=f1_scorer)
grid_obj = grid_obj.fit(X_train,y_train)
clf = grid_obj.best_estimator_
print(clf)
y_pred = clf.predict(X_test)
f1_score_value=f1_score(y_test, y_pred, pos_label=0) # For testing
print ("F1 Score for test set: {}".format(f1_score_value))
print ("Confusion Matrix is : \n  {} ".format(confusion_matrix(y_test, y_pred)))
target_names = ['class 0', 'class 1']
print (" ")
print ("Classification report is : \n  ")
print(classification_report(y_test, y_pred, target_names=target_names))


from sklearn import tree
from sklearn.metrics import classification_report
parameters = [{'criterion':['entropy','gini'] ,'splitter':['best','random']}]
clf = tree.DecisionTreeClassifier(random_state=0)
f1_scorer = make_scorer(f1_score,pos_label=0)
sss = StratifiedShuffleSplit( y_train, n_iter=10, test_size=0.25)
grid_obj = GridSearchCV(clf,parameters,cv = sss,scoring=f1_scorer)
grid_obj = grid_obj.fit(X_train,y_train)
clf = grid_obj.best_estimator_
y_pred = clf.predict(X_test)
f1_score_value=f1_score(y_test, y_pred, pos_label=0) # For testing
print ("F1 Score for test set: {}".format(f1_score_value))
print ("Confusion Matrix is : \n  {} ".format(confusion_matrix(y_test, y_pred)))
target_names = ['class 0', 'class 1']
print (" ")
print ("Classification report is : \n  ")
print(classification_report(y_test, y_pred, target_names=target_names))


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

parameters = [{'solver':['lbfgs','sgd'],'alpha':[1e-6,1e-3],'hidden_layer_sizes':[3,6], }]
clf = MLPClassifier(random_state=42)
f1_scorer = make_scorer(f1_score,pos_label=0)
sss = StratifiedShuffleSplit( y_train, n_iter=10, test_size=0.25)
grid_obj = GridSearchCV(clf,parameters,cv = sss,scoring=f1_scorer)
grid_obj = grid_obj.fit(X_train,y_train)
clf = grid_obj.best_estimator_
y_pred = clf.predict(X_test)
f1_score_value=f1_score(y_test, y_pred, pos_label=0, average=None) # For testing
print ("F1 Score for test set: {}".format(f1_score_value))
print ("Confusion Matrix is : \n  {} ".format(confusion_matrix(y_test, y_pred)))
target_names = ['class 0', 'class 1']
print (" ")
print ("Classification report is : \n  ")
print(classification_report(y_test, y_pred, target_names=target_names))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)

f1_score_value=f1_score(y_test, y_pred, pos_label=0, average=None) # For testing
print ("F1 Score for test set: {}".format(f1_score_value))
print ("Confusion Matrix is : \n  {} ".format(confusion_matrix(y_test, y_pred)))
target_names = ['class 0', 'class 1']
print (" ")
print ("Classification report is : \n  ")
print(classification_report(y_test, y_pred, target_names=target_names))