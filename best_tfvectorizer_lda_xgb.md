
# Best


```python
import pandas as pd
from sklearn.externals import joblib
import numpy as np
from sklearn.cross_validation import KFold
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

df1 = pd.read_csv("data600.csv")
df2 = pd.read_csv("data600_labeled.csv")
df = pd.concat([df1,df2],0,ignore_index = True)

idx = df[df.Performance == 1].index
df.loc[idx,"BugsCrashes"] = 1
del df['Performance']

idx = df[df.Suggestion == 1].index
df.loc[idx,"Experience"] = 1
del df['Suggestion']

idx = df[df.None == 1].index
df.loc[idx,"Experience"] = 1
del df['None']

lda = joblib.load('lda2_30.pkl')
tf_vectorizer = joblib.load('tf_vectorizer.pkl')
target_name = ['BugsCrashes','Experience','Hardware','Pricing']
data = pd.concat([pd.DataFrame(lda.transform(tf_vectorizer.transform(df.Body.tolist()))),df.BugsCrashes,df.Experience, df.Hardware, df.Pricing], 1)
X = lda.transform(tf_vectorizer.transform(df.Body.tolist()))
y = df[['BugsCrashes','Experience', 'Hardware', 'Pricing']]
y = np.array(y)

full_clf_pred = np.empty((0,4))
full_y_test = np.empty((0,4))
k_fold = KFold(data.shape[0], n_folds=20, shuffle=True, random_state=40)
test_idx_list = np.empty(0)

for fold in k_fold:
    train_idx = fold[0] 
    test_idx = fold[1]
    
    X_train, y_train = X[train_idx,:], y[train_idx,:]
    X_test, y_test = X[test_idx, :], y[test_idx, :]
    
    clf = OneVsRestClassifier(xgb.XGBClassifier()).fit(X_train,y_train)
    clf_pred_p = clf.predict_proba(X_test)
#     clf_pred_p = np.hstack(map(lambda x: np.expand_dims(x[:,1],1), clf_pred_p))
    clf_pred_p = (clf_pred_p > 0.4).astype(np.int16)
    
    full_clf_pred = np.append(full_clf_pred,clf_pred_p, axis = 0)
    full_y_test = np.append(full_y_test,y_test, axis = 0)
    
    # save pred vs test df
    test_idx_list = np.append(test_idx_list,test_idx)
    
idx = df.iloc[test_idx_list,:].index
df_pred = pd.DataFrame(full_clf_pred, columns= ['BugsCrashes_pred', 'Experience_pred', 'Hardware_pred', 'Pricing_pred'], index=idx)
result = pd.concat([df.iloc[test_idx_list,:],df_pred], axis = 1)
    
print classification_report(full_y_test,full_clf_pred,target_names = target_name)
```

                 precision    recall  f1-score   support
    
    BugsCrashes       0.67      0.64      0.66       384
     Experience       0.67      0.89      0.76       656
       Hardware       0.80      0.78      0.79       227
        Pricing       0.66      0.59      0.62       262
    
    avg / total       0.69      0.76      0.72      1529
    

