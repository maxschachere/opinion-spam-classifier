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

# UNCOMMENT THIS
# SPELLING_DICT = enchant.Dict("en_US")
GRAMMAR_CHECK = grammar_check.LanguageTool('en-US')

def main():
	raw_data = load_data()
	bigram_set = generate_bigram_set(raw_data)
	unigram_set = generate_unigram_set(raw_data)
	# processed_data = featurize_data(raw_data, bigram_set, unigram_set, baseline_flag=False)
	processed_data_baseline = featurize_data(raw_data, bigram_set, unigram_set, baseline_flag=True)
	# processed_data.to_csv("data.csv")
	# training_data = processed_data.sample(frac=0.7)
	# validation_data = processed_data.loc[set(processed_data.index)-set(training_data.index)].sample(frac=0.5)
	# test_data = processed_data.loc[set(processed_data.index)-set(training_data.index)-set(validation_data.index)]


	# split the baseline data
	training_data_baseline = processed_data_baseline.sample(frac=0.7)
	validation_data_baseline = processed_data_baseline.loc[set(processed_data_baseline.index) - \
														   set(training_data_baseline.index)].sample(frac=0.5)
	test_data_baseline = processed_data_baseline.loc[set(processed_data_baseline.index) - \
											set(training_data_baseline.index) - set(validation_data_baseline.index)]

	print("Fitting and Evaluating Baseline Model")
	model_baseline = create_model(training_data_baseline)
	evaluate_model(model_baseline, validation_data_baseline)

	# training_data = processed_data.iloc[0:floor(7 * len(processed_data)/10)]
	# training_data = processed_data.iloc[floor(7 * len(processed_data)/10):floor(7 * len(processed_data)/10)]
	# training_data = processed_data.iloc[floor(7 * len(processed_data)/10):floor(7 * len(processed_data)/10)]
	# print("Fitting and Evaluating enhanced model")
	# model = create_model(training_data)
	# evaluate_model(model, validation_data)


def load_data():
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
	return data


def generate_bigram_set(raw_data):
	word_bigrams = set()
	pos_bigrams = set()
	for review_dict in raw_data:
		text = review_dict['review']
		tokens = word_tokenize(text)
		pos_tags = pos_tag(tokens)
		for i in range(1, len(pos_tags)):
			word_bigram = pos_tags[i-1][0] + '-' + pos_tags[i][0]
			pos_bigram = pos_tags[i-1][1] + '-' + pos_tags[i][1]
			word_bigrams.add(word_bigram)
			pos_bigrams.add(pos_bigram)
	return {"word_bigram_set": word_bigrams, "pos_bigram_set": pos_bigrams}

def generate_unigram_set(raw_data):
	word_unigrams = set()
	for review_dict in raw_data:
		text = review_dict['review']
		tokens = word_tokenize(text)
		for word in tokens:
			word_unigrams.add(word)
	return word_unigrams

def featurize_data(data, overall_bigram_set_dict, word_unigrams, baseline_flag):
	print "Featurizing the data..."
	processed_data = []
	for review in data:
		text = review['review']
		if baseline_flag:
			review_vector = featurize_review_baseline(text, word_unigrams)
		else:
			review_vector = featurize_review(text, overall_bigram_set_dict)
		review_vector['real'] = 1 if review['real'] else 0
		processed_data.append(review_vector)

	processed_data = pd.DataFrame.from_dict(processed_data)
	return processed_data

def featurize_review_baseline(text, word_unigram_set):
	review_vector = {}
	word_tokens = word_tokenize(text)
	for unigram in word_unigram_set:
		if unigram in word_tokens:
			review_vector[unigram] = 1
		else:
			review_vector[unigram] = 0
	return review_vector

def get_review_pos_word_bigrams(text):
	pos_bigrams = []
	word_bigrams = []
	tokens = word_tokenize(text)
	pos_tags = pos_tag(tokens)
	for i in range(1, len(pos_tags)):
		word_bigram = pos_tags[i-1][0] + '-' + pos_tags[i][0]
		pos_bigram = pos_tags[i - 1][1] + '-' + pos_tags[i][1]
		pos_bigrams.append(pos_bigram)
		word_bigrams.append(word_bigram)
	return {"word_bigrams": word_bigrams, "pos_bigrams": pos_bigrams}

def featurize_review(review, overall_bigram_dict):
	review_vector = {}
	words = review.split(" ")
	tokens = nltk.word_tokenize(review)
	pos_tags = nltk.pos_tag(tokens)
	noun_count = 0
	adjective_count = 0
	first_person_count = 0
	adverb_count = 0
	misspelled_words = 0
	VBD_count = 0
	CC_count = 0
	review_bigram_dictionary = get_review_pos_word_bigrams(review)
	review_word_bigrams = review_bigram_dictionary['word_bigrams']
	review_pos_bigrams = review_bigram_dictionary['pos_bigrams']

	overall_word_bigram_set = overall_bigram_dict["word_bigram_set"]
	overall_pos_bigram_set = overall_bigram_dict["pos_bigram_set"]

	for word_bigram in overall_word_bigram_set:
		review_vector[word_bigram] = review_word_bigrams.count(word_bigram)
	for pos_bigram in overall_pos_bigram_set:
		review_vector[pos_bigram] = review_pos_bigrams.count(pos_bigram)

	for tag in pos_tags:
		word, tag = tag
		if word.lower() in ['i', 'we', 'me', 'us']:
			first_person_count = first_person_count + 1

		# if re.match('[a-zA-Z]', tag) and SPELLING_DICT.check(word) is not True:
		# 	misspelled_words = misspelled_words + 1
		if 'NN' in tag:
			noun_count = noun_count + 1
		elif 'JJ' in tag:
			adjective_count = adjective_count + 1
		elif 'RB' in tag:
			adverb_count = adverb_count + 1
		elif 'VB' in tag:
			VBD_count = VBD_count + 1
		elif tag == 'CC':
			CC_count = CC_count + 1

	#orthographic features


	review_vector['number_oov'] = 0
	word_set = set()
	word_tokens = word_tokenize(review)
	for word in word_tokens:
		if word_tokens.count(word) == 1:
			review_vector['number_oov'] += 1
		word_set.add(word)
	review_vector['unique_word_count'] = len(word_set)

	review_vector['length'] = len(review)
	review_vector['word_count'] = len(words)
	review_vector['average_word_length'] = sum(len(x) for x in words)/len(words)
	review_vector['noun_freq'] = noun_count/float(len(words))
	review_vector['adjective_freq'] = adjective_count/float(len(words))
	review_vector['first_person_freq'] = first_person_count/float(len(words))
	review_vector['adverb_freq'] = adverb_count/float(len(words))
	review_vector['misspell_freq'] = misspelled_words/float(len(words))
	review_vector['VBD_freq'] = VBD_count/float(len(words))
	review_vector['CC_freq'] = CC_count/float(len(words))

	return review_vector

def get_named_entities():
	pass	

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
	print "AUC: " + str(roc_auc_score(y_true, y_pred_prob))
	# print model.coef_

if __name__ == "__main__": main()