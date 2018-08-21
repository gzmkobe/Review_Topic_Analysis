# Review_topics

**labeled_data** is the dataset sampled from the original review dataset with our manual labels.

**LDA_script.py** is the main script used to train LDA model, train random forest classifiers, and tune hyper-parameters.

**grep.sh** is used to sample review data by searching keyword, this is how we create **labeled_data**

**LDA_Topic35_Feature15000_length10max_df0.1min_df1e-05.pkl** and **TF_Vectorizer_Topic35_Feature15000_length10max_df0.1min_df1e-05.pkl** are the best vectorizer and best LDA model.

**Random Forest with Hyper-parameters Tuning.md** is the source code to produce the best classification report below 
**Xgboost.md** is the source code to run Xgboost on training data. Since it has so many hyper-parameters, we can explore it in the future.

## Usage
There are two datasets, one is **data600.csv**, the other one is **data600_labeled.csv**. Don't be confused about the names, I actually merge then by column to create machine learning training data. Also, notice that in these two datasets, I have 7 topics, but later on, I only have 4 topics in output. I merge **Performance** to **BugsCrash**, **Suggestion** to **Experience**, and **None** to **Experience** in **LDA_script.py**. In future, if we have more labeled data we can do go back to 7 topics.

## Hyper-parameters
The overall parameters have been divided in 2 categories.

1. One category is LDA parameters, they determine the data quality. The most important two parameters in LDA are the length of sentence after I remove stopwords, which is 10 in this case. The other parameter is max_df, which is 0.05. The training data transfromed from LDA is fitted into Random Forest, and XgBoost, both obtain 0.81 f1 score as a baseline.

2. The other category is random forest/XgBoost parameters. These parameters will explore the upper limit of machine learning models, which now is 0.8284


## Best random forest model based on f1-score with adjusted threshold

```

    n_estimators_1000,max_depth_10,max_features_sqrt,min_samples_leaf_1,min_samples_split_2
                 precision    recall  f1-score   support
    
    BugsCrashes     0.8356    0.7943    0.8144       384
     Experience     0.7872    0.8796    0.8308       656
       Hardware     0.8883    0.7709    0.8255       227
        Pricing     0.9043    0.7939    0.8455       262
    
    avg / total     0.8344    0.8273    0.8284      1529
```

## Future Work##
1. Setup Xgboost hyper-parameter grid search pipeline
2. Manually label more data
3. Tune LDA pparameters for max_df around 0.1, increase the number of features starting at 10000, and number of topics from 30 to 75.
