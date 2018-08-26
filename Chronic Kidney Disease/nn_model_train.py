import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("C://Users//gauta//Desktop//Guvi//CKD//ckd.csv",sep=',', na_values=['?'])
#No of people with chronic kidney disease
n_ckd = len(data[data['class']=='ckd'])

#No of people without chronic kidney disease
n_notckd = len(data[data['class']=='notckd'])

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

from sklearn.preprocessing import RobustScaler
target_class = data['class']
features = data.drop('class', axis = 1)
data_robust = pd.DataFrame(RobustScaler().fit_transform(features), columns=features.columns)


from sklearn.decomposition import PCA
pca = PCA()
data_pca = pd.DataFrame(pca.fit_transform(data_robust), columns=data_robust.columns)


pca = PCA(n_components=11)
pca.fit(data_robust)
reduced_data = pca.transform(data_robust)
reduced_data = pd.DataFrame(reduced_data, columns = ['dim1','dim2','dim3','dim4','dim5','dim6','dim7','dim8','dim9','dim10','dim11'])

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='relu', alpha=1e-06, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=3, learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

clf.fit(reduced_data, target_class)

from sklearn.externals import joblib

joblib.dump(clf, 'model_train.pkl')