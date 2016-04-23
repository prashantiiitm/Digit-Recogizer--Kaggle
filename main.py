import pandas as pd;
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

train=pd.read_csv('data/train.csv');
#print train.info();
#print train.shape
#train=train.loc[(train != 0).any(axis=0),:]
#print train.shape
#print train.head(100);
#print train.info();
train_data=train.values;
#print train_data[900:1000,0:10];

#train_features=train_data[:,1:];
#train_labels=train_data[:,0:1];
#print train_labels

from sklearn.ensemble import RandomForestClassifier;

clf=RandomForestClassifier(n_estimators = 1000);

clf.fit(train_data[0:,1:], train_data[0:,0]);

test=pd.read_csv('data/test.csv')

test_data=test.values;
output=clf.predict(test_data)

res=np.c_[test_data[:,0].astype(int),output.astype(int)];
#print res
df_res=pd.DataFrame(res[:,0],columns=["Label"]);
df_res.index+=1;
print df_res
df_res.to_csv('results/chardata.csv',index_label="ImageId")
print "done";