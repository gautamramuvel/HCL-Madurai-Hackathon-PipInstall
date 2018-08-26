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

def label_encode_rbc(x):
    switcher = {
        "normal": 1,
        "abnormal":2
    }
    return switcher.get(x, 0)
def label_encode_pc(x):
    switcher = {
        "normal": 1,
        "abnormal":2
    }
    return switcher.get(x, 0)
def label_encode_pcc(x):
    switcher = {
        "present": 1,
        "notpresent":2
    }
    return switcher.get(x, 0)
def label_encode_ba(x):
    switcher = {
        "present": 1,
        "notpresent":2
    }
    return switcher.get(x, 0)
def label_encode_htn(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
def label_encode_dm(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
def label_encode_cad(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
def label_encode_appet(x):
    switcher = {
        "no": 1,
        "poor":2,
        "good":3
    }
    return switcher.get(x, 0)
def label_encode_pe(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
def label_encode_ane(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)

def encoding(data):
    for i in data["rbc"]:
        data["rbc"] = int(label_encode_rbc(str(i)))
    for i in data["pc"]:
        data["pc"] = int(label_encode_pc(str(i)))
    for i in data["pcc"]:
        data["pcc"] = int(label_encode_pcc(str(i)))
    for i in data["ba"]:
        data["ba"] = int(label_encode_ba(str(i)))
    for i in data["htn"]:
        data["htn"] = int(label_encode_htn(str(i)))
    for i in data["dm"]:
        data["dm"] = int(label_encode_dm(str(i)))
    for i in data["cad"]:
        data["cad"] = int(label_encode_cad(str(i)))
    for i in data["appet"]:
        data["appet"] = int(label_encode_appet(str(i)))
    for i in data["pe"]:
        data["pe"] = int(label_encode_pe(str(i)))
    for i in data["ane"]:
        data["ane"] = int(label_encode_ane(str(i)))
    return data

from sklearn.preprocessing import RobustScaler
target_class = new_data['class']
features = new_data.drop('class', axis = 1)
features = features.drop('id', axis = 1)
features = encoding(features)
data_robust = pd.DataFrame(RobustScaler().fit_transform(features), columns=features.columns)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_robust,target_class, test_size=0.25, random_state=42)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

clf = MLPClassifier(activation='relu', alpha=1e-06, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=3, learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
f1_score_value=f1_score(y_test, y_pred, pos_label=0, average=None) # For testing
print ("F1 Score for test set: {}".format(f1_score_value))
print ("Confusion Matrix is : \n  {} ".format(confusion_matrix(y_test, y_pred)))
target_names = ['class 0', 'class 1']
print (" ")
print ("Classification report is : \n  ")
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.externals import joblib

joblib.dump(clf, 'C://Users//gauta//Desktop//Guvi//CKD//1model.pkl')