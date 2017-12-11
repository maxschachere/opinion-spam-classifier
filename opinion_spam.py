import os
import pandas as pd
import nltk
from sklearn import linear_model
from sklearn.metrics import accuracy_score

DATA_SET_PATH = "Data Sets/op_spam_v1.4/"

def main():
	raw_training_data = load_data()
	processed_data = featurize_data(raw_training_data)
	model = create_model(processed_data)
	evaluate_model(model, processed_data)

def load_data():
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

def featurize_data(data):
	processed_data = []
	for review in data:
		text = review['review']
		review_vector = featurize_review(text)
		review_vector['real'] = 'real' if review['real'] else 'fake'
		processed_data.append(review_vector)

	processed_data = pd.DataFrame.from_dict(processed_data)
	return processed_data

def featurize_review(review):
	review_vector = {}
	words = review.split(" ")
	review_vector['length'] = len(review)
	review_vector['word_count'] = len(words)
	review_vector['average_word_length'] = sum(len(x) for x in words)/len(words)
	tokens = nltk.word_tokenize(review)
	pos_tags = nltk.pos_tag(tokens)
	noun_count = 0
	adjective_count = 0
	first_person_count = 0
	for tag in pos_tags:
		word, tag = tag
		if 'NN' in tag:
			noun_count = noun_count + 1
		if 'JJ' in tag:
			adjective_count = adjective_count + 1
		if word.lower() in ['i', 'we', 'me', 'us']:
			first_person_count = first_person_count + 1
	review_vector['num_nouns'] = noun_count
	review_vector['num_adjectives'] = adjective_count
	review_vector['first_person'] = first_person_count
	return review_vector

def get_named_entities():
	pass	

def create_model(data):
	x = data.loc[:, data.columns != 'real']
	y = data['real']
	logreg = linear_model.LogisticRegression()
	logreg.fit(x, y)

	return logreg

def evaluate_model(model, data):
	x = data.loc[:, data.columns != 'real']
	y_true = data['real']
	y_pred = model.predict(x)
	print accuracy_score(y_true, y_pred)

if __name__ == "__main__": main()