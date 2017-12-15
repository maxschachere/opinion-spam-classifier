import os
import re
import pandas as pd
import nltk
from sklearn import linear_model, svm, neighbors, naive_bayes
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# UNCOMMENT THIS
# import enchant

import grammar_check
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk

DATA_SET_PATH = "Data Sets/op_spam_v1.4/"

# SPELLING_DICT = enchant.Dict("en_US")
GRAMMAR_CHECK = grammar_check.LanguageTool('en-US')

def main():
	raw_data = load_data()

	# split the data into train, validation and test
	training_data = raw_data.sample(frac=0.7)
	validation_data = raw_data.loc[set(raw_data.index) - set(training_data.index)].sample(frac=0.5)
	test_data = raw_data.loc[set(raw_data.index) - set(training_data.index) - set(validation_data.index)]

	training_data_featurized = featurize_data(training_data, training=None)
	# print training_data_featurized.shape
	validation_data_featurized = featurize_data(validation_data, training=training_data_featurized)
	# print validation_data_featurized.shape
	test_data_featurized = featurize_data(test_data, training=training_data_featurized)
	# print test_data_featurized.shape
	model = create_model(training_data_featurized)
	evaluate_model(model, validation_data_featurized)
	print("Performance on Test")
	evaluate_model(model, test_data_featurized)

	print("Moving to baseline")
	baseline_unigrams = get_baseline_unigrams(training_data)
	training_data_baseline = featurize_data_baseline(training_data, None)
	# print training_data_baseline.shape
	# print training_data_baseline.columns
	validation_data_baseline = featurize_data_baseline(validation_data, training_data_baseline)
	# print validation_data_baseline.shape
	test_data_baseline = featurize_data_baseline(test_data, training_data_baseline)
	# print test_data_baseline.shape
	model = create_model(training_data_baseline)
	evaluate_model(model, validation_data_baseline)
	print("Performance on Baseline Test")

def load_data():
	"""
	returns pandas dataframe with two columns: review, real
	"""
	print "Loading data..."
	data = []
	real_reviews_path = DATA_SET_PATH + "real/"
	fake_reviews_path = DATA_SET_PATH + "fake/"
	for filename in os.listdir(real_reviews_path):
		if filename.endswith(".txt"):
			with open(real_reviews_path + filename) as f:
				data.append({'review': f.readline(), 'real': True})
	for filename in os.listdir(fake_reviews_path):
		if filename.endswith(".txt"):
			with open(fake_reviews_path + filename) as f:
				data.append({'review': f.readline(), 'real': False})
	data = pd.DataFrame.from_dict(data)
	return data


def featurize_data(data, training):
	# training is either a DF or false (meaning that this is the training run)
	print "Featurizing the data..."

	if training is None:
		training_features = None
	else:
		training_features = list(training)

	processed_data = []
	for index, review in data.iterrows():
		text = review['review']
		review_vector = featurize_review(text, training_features)
		review_vector['real'] = 1 if review['real'] else 0
		processed_data.append(review_vector)

	processed_dataframe = pd.DataFrame.from_dict(processed_data)
	# standardize dimensions
	if training is not None:
		for training_feature in list(training):
			if training_feature not in list(processed_dataframe):
				processed_dataframe[training_feature] = 0
	processed_dataframe.fillna(0, inplace=True)
	return processed_dataframe


def featurize_data_baseline(data, training_features):
	# training is either a DF or false (meaning that this is the training run)
	print "Featurizing the baseline data..."

	processed_data = []
	for index, review in data.iterrows():
		text = review['review']
		review_vector = featurize_review_baseline(text, training_features)
		review_vector['real'] = 1 if review['real'] else 0
		processed_data.append(review_vector)

	processed_dataframe = pd.DataFrame.from_dict(processed_data)
	# this code standardizes dimensions across training, validation and test
	# THIS NEEDS TO BE RUN IF WORD AND POS BIGRAMS ARE BEING USED
	if training_features is not None:
		print("Standardizing Dimensions to training dimensions")
		for feature in training_features:
			if feature not in list(processed_dataframe):
				processed_dataframe[feature] = 0
	processed_dataframe.fillna(0, inplace=True)
	return processed_dataframe

# deprecated
def get_baseline_unigrams(training_data):
	unigrams = set()
	all_reviews = training_data['review'].values
	for review in all_reviews:
		tokens = review.split(" ")
		unigrams.update(tokens)
	return unigrams


def featurize_review_baseline(review, training_features):
	review_vector = {}
	tokens = nltk.word_tokenize(review)
	pos_tags = nltk.pos_tag(tokens)
	for i in range(len(pos_tags)):
		curr_word, curr_tag = pos_tags[i]
		if training_features is None:
			if curr_word in review_vector.keys():
				review_vector[curr_word] = review_vector[curr_word] + 1
			else:
				review_vector[curr_word] = 1
		else:
			if curr_word in training_features:
				if curr_word in review_vector.keys():
					review_vector[curr_word] = review_vector[curr_word] + 1
				else:
					review_vector[curr_word] = 1

	return review_vector

def featurize_review(review, training_features):
	# training features is list of columns
	review_vector = {}
	words = review.split(" ")
	tokens = nltk.word_tokenize(review)
	pos_tags = nltk.pos_tag(tokens)
	first_person_count = 0
	misspelled_words = 0
	review_vector['all_caps'] = 0
	review_vector['title'] = 0
	review_vector['number'] = 0
	review_vector['alphanumeric'] = 0

	for i in range(len(pos_tags)):
		curr_word, curr_tag = pos_tags[i]
		if i == 0:
			prev_word = ""
			prev_tag = ""
		else:
			prev_word, prev_tag = pos_tags[i - 1]


		if curr_word.lower() in ['i', 'we', 'me', 'us']:
			first_person_count = first_person_count + 1
		# if re.match('[a-zA-Z]', curr_tag) and SPELLING_DICT.check(curr_word) is not True:
		# 	misspelled_words = misspelled_words + 1

		if curr_tag in review_vector.keys():
			review_vector[curr_tag] = review_vector[curr_tag] + 1
		else:
			review_vector[curr_tag] = 1

		# check capitalization
		if curr_word.isupper():
			review_vector['all_caps'] = review_vector['all_caps'] + 1
		if curr_word.istitle():
			review_vector['title'] = review_vector['title'] + 1
		if curr_word.isdigit():
			review_vector['number'] = review_vector['number'] + 1
		if re.match('(([a-zA-Z]+[0-9]+([a-zA-Z]+)?)|([0-9]+[a-zA-Z]+))', curr_word):
			review_vector['alphanumeric'] += 1

	review_vector['length'] = len(review)
	review_vector['word_count'] = len(words)
	review_vector['average_word_length'] = sum(len(x) for x in words) / len(words)
	review_vector['first_person_freq'] = first_person_count / float(len(words))
	review_vector['misspell_freq'] = misspelled_words / float(len(words))

	review_vector['number_oov'] = 0
	word_set = set()

	for word in tokens:
		if tokens.count(word) == 1:
			review_vector['number_oov'] += 1
		word_set.add(word)
	review_vector['unique_word_count'] = len(word_set)
	# review_vector['bigram_word_OOV'] = 0
	# review_vector['bigram_pos_OOV'] = 0

	return review_vector

def create_model(data):
	print "Creating model..."
	x = data.loc[:, data.columns != 'real']
	y = data['real']
	logreg = linear_model.LogisticRegression()
	logreg.fit(x, y)

	# clf = svm.SVC()
	# clf.fit(x, y)

	# knn = neighbors.KNeighborsClassifier(15)
	# knn.fit(x, y)

	# nbc = naive_bayes.GaussianNB()
	# nbc.fit(x, y)

	return logreg
	# return clf
	# return knn
	# return nbc

def evaluate_model(model, data):
	print "Evaluating model..."
	x = data.loc[:, data.columns != 'real']
	y_true = data['real']
	y_pred = model.predict(x)
	y_pred_prob = model.predict_proba(x)

	print "Accuracy: " + str(accuracy_score(y_true, y_pred))
	print "F-score: " + str(f1_score(y_true, y_pred))
	print "AUC: " + str(roc_auc_score(y_true, y_pred_prob[:,1]))


if __name__ == "__main__": main()