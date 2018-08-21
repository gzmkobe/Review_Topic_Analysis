
# Best


```python
import pandas as pd
from sklearn.externals import joblib
import numpy as np
from sklearn.cross_validation import KFold
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid

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


lda = joblib.load('LDA_Topic35_Feature15000_length10max_df0.1min_df1e-05.pkl')
tf_vectorizer = joblib.load('TF_Vectorizer_Topic35_Feature15000_length10max_df0.1min_df1e-05.pkl')
target_name = ['BugsCrashes','Experience','Hardware','Pricing']
data = pd.concat([pd.DataFrame(lda.transform(tf_vectorizer.transform(df.Body.tolist()))),df.BugsCrashes,df.Experience, df.Hardware, df.Pricing], 1)
X = lda.transform(tf_vectorizer.transform(df.Body.tolist()))
y = df[['BugsCrashes','Experience', 'Hardware', 'Pricing']]
y = np.array(y)


# rf_grid = {'n_estimators':[1000,1500],
#     'max_depth': [10,20,30,None],
#  'max_features': ['sqrt'],
#  'min_samples_leaf': [1,2],
#  'min_samples_split': [2,3,10]}

rf_grid = {'n_estimators':[1000],
    'max_depth': [10],
 'max_features': ['sqrt'],
 'min_samples_leaf': [1],
 'min_samples_split': [2]}

for i in range(len(ParameterGrid(rf_grid))):
    n_estimators, min_samples_split,max_depth, max_features, min_samples_leaf = ParameterGrid(rf_grid)[i].values()
    print '\n'
    print 'n_estimators_' + str(n_estimators) + ',max_depth_' + str(max_depth) + ',max_features_' + str(max_features) + ',min_samples_leaf_'+str(min_samples_leaf) + ',min_samples_split_' + str(min_samples_split)
    full_clf_pred = np.empty((0,4))
    full_y_test = np.empty((0,4))
    k_fold = KFold(data.shape[0], n_folds=20, shuffle=True, random_state=40)
    test_idx_list = np.empty(0)
    for fold in k_fold:
        train_idx = fold[0] 
        test_idx = fold[1]
        X_train, y_train = X[train_idx,:], y[train_idx,:]
        X_test, y_test = X[test_idx, :], y[test_idx, :]

        clf = RandomForestClassifier(n_jobs = 8, random_state = 10, max_features = max_features, 
                                     min_samples_split = min_samples_split, max_depth = max_depth, 
                                     min_samples_leaf = min_samples_leaf, n_estimators = n_estimators).fit(X_train,y_train)
        
        clf_pred_p = clf.predict_proba(X_test)
        clf_pred_p = np.hstack(map(lambda x: np.expand_dims(x[:,1],1), clf_pred_p))
        clf_pred_p = (clf_pred_p > 0.45).astype(np.int16)

        full_clf_pred = np.append(full_clf_pred,clf_pred_p, axis = 0)
        full_y_test = np.append(full_y_test,y_test, axis = 0)

        # save pred vs test df
        test_idx_list = np.append(test_idx_list,test_idx)

    idx = df.iloc[test_idx_list,:].index
    df_pred = pd.DataFrame(full_clf_pred, columns= ['BugsCrashes_pred', 'Experience_pred', 'Hardware_pred', 'Pricing_pred'], index=idx)
    result = pd.concat([df.iloc[test_idx_list,:],df_pred], axis = 1)

    print classification_report(full_y_test,full_clf_pred,target_names = target_name, digits=4)
```

    
    
    n_estimators_1000,max_depth_10,max_features_sqrt,min_samples_leaf_1,min_samples_split_2
                 precision    recall  f1-score   support
    
    BugsCrashes     0.8356    0.7943    0.8144       384
     Experience     0.7872    0.8796    0.8308       656
       Hardware     0.8883    0.7709    0.8255       227
        Pricing     0.9043    0.7939    0.8455       262
    
    avg / total     0.8344    0.8273    0.8284      1529
    

