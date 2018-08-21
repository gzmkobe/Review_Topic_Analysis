import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,ParameterGrid
import pickle
from sklearn.cross_validation import KFold

### sampled review data
# sample_rev = pd.read_csv('sample4.tsv', sep='\t', error_bad_lines =False)
# sample_rev = sample[sample_rev.iloc[:,2].notnull()]
# sample_rev = sample_rev[sample_rev.iloc[:,2].str.count(' ') >= 15]
# sample_rev = sample_rev.sample(frac=1.)
# sample_list = sample_rev.iloc[:,2].tolist()
# with open("sample_list.txt", "wb") as f:
#     pickle.dump(sample_list, f)

######################################################################### Additional Functional Tools ############################################################
def print_top_words(model, feature_names, n_top_words):
	for topic_idx, topic in enumerate(model.components_):
		message = "Topic #%d: " % topic_idx
		message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
		print(message)
	print()


def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in xrange(0, len(l), n):
		yield l[i:i + n]


def process_data():
	'''
	This dataset is created from manually labeled datasets to a machine learning training data  
	'''
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
	return df


def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    del lines[6]
    for line in lines[2:-1]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', mode = 'a', index = False)

######################################################################### LDA part ############################################################
def LDA(review_data, df, n_features = 10000, length = 10, n_top_words = 25, max_df = 0.01, min_df = 0.00001, n_components = 30, 
		max_features = None, min_samples_split = None, max_depth = None, min_samples_leaf = None, myCsvRow = None):

	print "Start tf_vectorizer"
	tf_vectorizer = CountVectorizer(max_df = max_df, min_df=min_df,
									max_features=n_features,
									stop_words='english',
									token_pattern = r"(?u)\b[A-Za-z0-9]{3,}\b")
	tf = tf_vectorizer.fit_transform(review_data)
	tf = tf[np.array(tf.sum(1)).flatten() > length,:]
	tf_feature_names = tf_vectorizer.get_feature_names()

	test = tf[tf.shape[0]-100000:]
	tf = tf[:tf.shape[0]-100000]
	file1_name = 'TF_Vectorizer_' + 'Topic'+str(n_components) + '_Feature' + str(n_features)  + '_length' + str(length) + 'max_df'+str(max_df) + 'min_df' + str(min_df) + '.pkl'
	joblib.dump(tf_vectorizer, file1_name)	
	print "Finished tf_vectorizer"

	print "start LDA"
	lda = LatentDirichletAllocation(n_components=n_components,
									learning_method='online', verbose = 1, learning_decay=0.5, batch_size = 4096,
									learning_offset=64, total_samples = tf.shape[0],
									random_state=0, n_jobs=8)
	
	last_bound = 1000000
	for it in range(8):
		for i, ll in enumerate(chunks(range(tf.shape[0]), 100000)):
			lda.partial_fit(tf[ll])
		bound = lda.perplexity(test)
		print "preplexity:",bound
		if last_bound and last_bound - bound < 0.1:
			break
		last_bound = bound
	print_top_words(lda, tf_feature_names, n_top_words)


	file2_name = 'LDA_' + 'Topic'+str(n_components) + '_Feature' + str(n_features)  + '_length' + str(length) + 'max_df'+str(max_df) + 'min_df' + str(min_df) + '.pkl'
	joblib.dump(lda, file2_name)
	print "Finished LDA"

######################################################################### Machine Learning part ############################################################
	target_name = ['BugsCrashes','Experience','Hardware','Pricing']

	data = pd.concat([pd.DataFrame(lda.transform(tf_vectorizer.transform(df.Body.tolist()))),df.BugsCrashes,df.Experience, df.Hardware, df.Pricing], 1)

	X = lda.transform(tf_vectorizer.transform(df.Body.tolist()))
	y = df[['BugsCrashes','Experience', 'Hardware', 'Pricing']]
	y = np.array(y)

	full_rf_pred = np.empty((0,4))
	full_y_test = np.empty((0,4))
	k_fold = KFold(data.shape[0], n_folds=10, shuffle=True, random_state=40)
	for fold in k_fold:
		train_idx = fold[0] 
		test_idx = fold[1]
		X_train, y_train = X[train_idx,:], y[train_idx,:]
		X_test, y_test = X[test_idx, :], y[test_idx, :]

		rf = RandomForestClassifier(n_jobs = 8, random_state = 10, n_estimators = 300, max_features = max_features, min_samples_split = min_samples_split, 
									max_depth = max_depth, min_samples_leaf = min_samples_leaf).fit(X_train, y_train)
		rf_pred = rf.predict(X_test)

		full_rf_pred = np.append(full_rf_pred,rf_pred, axis = 0)
		full_y_test = np.append(full_y_test,y_test, axis = 0)
	print '############rf#############\n',classification_report(full_y_test, full_rf_pred, target_names = target_name, digits = 3) 

	with open('classification_report.csv', 'a') as csvfile:
		csvfile.write('\n')
		csvfile.write(myCsvRow)
		csvfile.write('\n')

	report = classification_report(full_y_test, full_rf_pred, target_names = target_name)
	classification_report_csv(report)

######################################################################### Tuning Hyper-parameters ############################################################
def grid_search(review_data, df):

	lda_grid = {'n_features':[15000],
	'n_components':[31,35,40],'length':[10], 
	'max_df': [0.04,0.05,0.06], 
	'min_df': [0.00001]}

	rf_grid = {'max_depth': [None],
	'max_features': ['sqrt'],
	'min_samples_leaf': [1],
	'min_samples_split': [2]}

	print "Start Grid-search"
	num_comb = len(ParameterGrid(lda_grid))*len(ParameterGrid(rf_grid))
	count = 0
	for i in range(len(ParameterGrid(lda_grid))):
		for j in range(len(ParameterGrid(rf_grid))):
			count += 1
			print "Now it is {} out of {} combinations".format(count, num_comb)
			max_df, length, n_features, n_components, min_df = ParameterGrid(lda_grid)[i].values()
			min_samples_split, max_features, max_depth, min_samples_leaf = ParameterGrid(rf_grid)[j].values()
			myCsvRow = "Parameters are: n_features = {}, n_components= {}, length= {}, max_df= {}, min_df= {},max_features= {}, min_samples_split= {}, max_depth= {}, min_samples_leaf= {}".format(n_features, n_components, length, max_df, min_df,max_features, min_samples_split, max_depth, min_samples_leaf)
			print myCsvRow
			LDA(review_data = sample_list, df = df,n_features = n_features, length = length, n_top_words = 25, max_df = max_df, min_df = min_df, n_components = n_components, 
				max_features = max_features, min_samples_split = min_samples_split, max_depth = max_depth, min_samples_leaf = min_samples_leaf, myCsvRow = myCsvRow)
	print "Finished Grid-search"
######################################################################### Shell part ############################################################
if __name__ == '__main__':
	with open("sample_list.txt", "rb") as fp:
		sample_list = pickle.load(fp)
		print "Review Data Loaded"

	df = process_data()
	print 'Label Data Loaded'

	grid_search(review_data = sample_list, df = df)


