# Review_topics

**labeled_data** is the dataset sampled from the original review dataset with our manual labels.

**LDA_script.py** is the main script used to train LDA model, train random forest classifiers, and tune hyper-parameters.

**grep.sh** is used to sample review data by searching keyword, this is how we create **labeled_data**

**lda2_30.pkl** and **tf_vectorizer.pkl** are the best vectorizer and best LDA model.

**best_tfvectorizer_lda_xgb.md** is the source code to produce the classification report below 

## Usage
There are two datasets, one is **data600.csv**, the other one is **data600_labeled.csv**. Don't be confused about the names, I actually merge then by column to create machine learning training data. Also, notice that in these two datasets, I have 7 topics, but later on, I only have 4 topics in output. I merge **Performance** to **BugsCrash**, **Suggestion** to **Experience**, and **None** to **Experience**. In future, if we have more labeled data we can do go back to 7 topics.
. 



## Best xgboost model based on f1-score with adjusted threshold

```

                 precision    recall  f1-score   support
    
    BugsCrashes       0.67      0.64      0.66       384
     Experience       0.67      0.89      0.76       656
       Hardware       0.80      0.78      0.79       227
        Pricing       0.66      0.59      0.62       262
    
    avg / total       0.69      0.76      0.72      1529
    

# Review_Topic_Analysis
