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
	# processed_data = featurize_data(raw_data, generate_bigram_set(raw_data))
	
	training_data = raw_data.sample(frac=0.7)
	validation_data = raw_data.loc[set(raw_data.index)-set(training_data.index)].sample(frac=0.5)
	test_data = raw_data.loc[set(raw_data.index)-set(training_data.index)-set(validation_data.index)]
	
	training_data = featurize_data(training_data, training=None)
	print training_data.shape
	validation_data = featurize_data(validation_data, training=training_data)
	print validation_data.shape
	test_data = featurize_data(test_data, training=training_data)
	print test_data.shape
	model = create_model(training_data)
	evaluate_model(model, validation_data)


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
	data = pd.DataFrame.from_dict(data)
	return data

def featurize_data(data, training):
	#training is either a DF or false (meaning that this is the training run)
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

	processed_data = pd.DataFrame.from_dict(processed_data)

	# this code standardizes dimensions across training, validation and test
	# THIS NEEDS TO BE RUN IF WORD AND POS BIGRAMS ARE BEING USED
	if training_features is not None:
		for feature in training_features:
			if feature not in list(processed_data):
				processed_data[feature] = 0
	processed_data.fillna(0, inplace=True)
	return processed_data

def featurize_review(review, training_features):
	#training features is list of columns

	review_vector = {}
	words = review.split(" ")
	tokens = nltk.word_tokenize(review)
	pos_tags = nltk.pos_tag(tokens)
	first_person_count = 0
	misspelled_words = 0
	review_vector['all_caps'] = 0
	review_vector['title'] = 0
	review_vector['number'] = 0
	# review_bigram_dictionary = get_review_pos_word_bigrams(review)
	# review_word_bigrams = review_bigram_dictionary['word_bigrams']
	# review_pos_bigrams = review_bigram_dictionary['pos_bigrams']

	# overall_word_bigram_set = overall_bigram_dict["word_bigram_set"]
	# overall_pos_bigram_set = overall_bigram_dict["pos_bigram_set"]

	# for word_bigram in overall_word_bigram_set:
	# 	review_vector[word_bigram] = review_word_bigrams.count(word_bigram)
	# for pos_bigram in overall_pos_bigram_set:
	# 	review_vector[pos_bigram] = review_pos_bigrams.count(pos_bigram)
	for i in range(len(pos_tags)):
		curr_word, curr_tag = pos_tags[i]
		if i==0:
			prev_word = ""
			prev_tag = ""
		else:
			prev_word, prev_tag = pos_tags[i-1]

		word_bigram = prev_word + "-" + curr_word
		pos_bigram = prev_tag + "-" + curr_tag

		#THIS BIGRAM WAS NOT EFFECTIVE AT ALL
		if training_features is None:
			#this means we ARE training
			if word_bigram in review_vector.keys():
				review_vector[word_bigram] = review_vector[word_bigram] + 1
			else:
				review_vector[word_bigram] = 1
			if pos_bigram in review_vector.keys():
				review_vector[pos_bigram] = review_vector[pos_bigram] + 1
			else:
				review_vector[pos_bigram] = 1
		else:
			#this means we are NOT training
			if word_bigram in training_features:
				if word_bigram in review_vector.keys():
					review_vector[word_bigram] = review_vector[word_bigram] + 1
				else:
					review_vector[word_bigram] = 1
			if pos_bigram in training_features:
				if pos_bigram in review_vector.keys():
					review_vector[pos_bigram] = review_vector[pos_bigram] + 1
				else:
					review_vector[pos_bigram] = 1


		if curr_word.lower() in ['i', 'we', 'me', 'us']:
			first_person_count = first_person_count + 1
		# if re.match('[a-zA-Z]', curr_tag) and SPELLING_DICT.check(curr_word) is not True:
		# 	misspelled_words = misspelled_words + 1
		
		if curr_tag in review_vector.keys():
			review_vector[curr_tag] = review_vector[curr_tag] + 1
		else:
			review_vector[curr_tag] = 1

		#THIS UNIGRAM WAS NOT EFFECTIVE AT ALL:
		# if training_features is None:
		# 	if curr_word in review_vector.keys():
		# 		review_vector[curr_word] = review_vector[curr_word] + 1
		# 	else:
		# 		review_vector[curr_word] = 1
		# else:
		# 	if curr_word in training_features:
		# 		if curr_word in review_vector.keys():
		# 			review_vector[curr_word] = review_vector[curr_word] + 1
		# 		else:
		# 			review_vector[curr_word] = 1

		#check capitalization
		if curr_word.isupper():
			review_vector['all_caps'] = review_vector['all_caps'] + 1
		if curr_word.istitle():
			review_vector['title'] = review_vector['title'] + 1
		if curr_word.isdigit():
			review_vector['number'] = review_vector['number'] + 1

	# sentences = sent_tokenize(review)
	# review_vector['grammar_errors'] = 0
	# for sentence in sentences:
	# 	matches = GRAMMAR_CHECK.check(sentence)
	# 	review_vector['grammar_errors'] += len(matches)

	# review_vector['number_oov'] = 0
	# word_set = set()
	# word_tokens = word_tokenize(review)
	# for word in word_tokens:
	# 	if word_tokens.count(word) == 1:
	# 		review_vector['number_oov'] += 1
	# 	word_set.add(word)
	# review_vector['unique_word_count'] = len(word_set)

	review_vector['length'] = len(review)
	review_vector['word_count'] = len(words)
	review_vector['average_word_length'] = sum(len(x) for x in words)/len(words)
	review_vector['first_person_freq'] = first_person_count/float(len(words))
	review_vector['misspell_freq'] = misspelled_words/float(len(words))
	# review_vector['bigram_word_OOV'] = 0
	# review_vector['bigram_pos_OOV'] = 0

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
	# print model.classes_
	# print y_pred_prob
	print "Accuracy: " + str(accuracy_score(y_true, y_pred))
	print "F-score: " + str(f1_score(y_true, y_pred))
	print "AUC: " + str(roc_auc_score(y_true, y_pred_prob[:,1]))
	# print model.coef_

if __name__ == "__main__": main()