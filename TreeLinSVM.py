# Clusters topic names using word2vec embeddings. Creates binary classifiers using 
#   composite topics of these clusters. Repeat on smaller composite topics until done.

from DataHandler import DataHandler

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.grid_search import GridSearchCV

from sltokenizer import StemmaLemmaTokenizer

import cPickle

import os


class TreeLinSVM:
	def __init__(self):
		self.composite_label_pairings = [
			('animals-sexuality-relationships', 
			'faith-family-fashion-food-meetup-military-misc-personal-pop_culture-questions-school-sports-tatoos-work'),
			('animals', 'sexuality-relationships'),
			('sexuality', 'relationships'),
			('misc', 'faith-family-fashion-food-meetup-military-personal-pop_culture-questions-school-sports-tatoos-work'),
			('meetup', 'faith-family-fashion-food-military-personal-pop_culture-questions-school-sports-tatoos-work'),
			('fashion-pop_culture-sports-tatoos', 'faith-family-food-military-personal-questions-school-work'),
			('fashion-sports-tatoos','pop_culture'),
			('tatoos','fashion-sports'),
			('fashion','sports'),
			('questions','faith-family-food-military-personal-school-work'),
			('faith','family-food-military-personal-school-work'),
			('food','family-military-personal-school-work'),
			('military','family-personal-school-work'),
			('school','family-personal-work'),
			('work','family-personal'),
			('personal','family'),
		]
		self.data_handler = DataHandler()

	# Translates categories to rough equivalents in word2vec model
	def model_category(self,category):
	    if category == 'lgbtq':
	        return 'sexuality'
	    elif category == 'qna':
	        return 'questions'
	    else:
	        return category

	# Returns a list of 2-tuples that the category will have to be classified according to.
	def which_classifiers(self,category):
		classifier_pairs = []
		for pairing in self.composite_label_pairings:
			if category in pairing[0].split('-') or category in pairing[1].split('-'):
				classifier_pairs.append(pairing)
		return classifier_pairs

	# Finds which classifiers the topic will go through and writes to the relevant data files to be used by those classifiers
	def write_data(self,category, sentence):
		classifiers = self.which_classifiers(category)
		for pairing in classifiers:
			filename = ''.join(['data/tree-linsvc/',pairing[0],'__',pairing[1],'.csv'])
			if not os.path.isfile(filename):
				write_file = open(filename,'w+')
				write_file.write('category,text\r\n')
				write_file.close()
			write_file = open(filename,'a')
			if category in pairing[0].split('-'):
				new_category = pairing[0]
			else:
				new_category = pairing[1]
			write_file.write(''.join([new_category,',',sentence.encode('utf8'),'\r\n']))


	# Reads in training data and calls other methods to write it to new composite-label datasets
	def build_new_data(self, text, targets):
		write_directory = 'data/tree-linsvc/'

		if not os.path.exists(write_directory):
			os.makedirs(write_directory)

		for idx, sentence in enumerate(text):
			self.write_data(self.model_category(targets[idx]),sentence)


	# Maps catgories to classifiers
	def category_to_classifier_name(self,category):
	    if category == 'animals-sexuality-relationships':
	        return 'animals__sexuality-relationships'
	    elif category == 'faith-family-fashion-food-meetup-military-misc-personal-pop_culture-questions-school-sports-tatoos-work':
	        return 'misc__faith-family-fashion-food-meetup-military-personal-pop_culture-questions-school-sports-tatoos-work'
	    elif category == 'faith-family-fashion-food-meetup-military-personal-pop_culture-questions-school-sports-tatoos-work':
	        return 'meetup__faith-family-fashion-food-military-personal-pop_culture-questions-school-sports-tatoos-work'
	    elif category == 'faith-family-fashion-food-military-personal-pop_culture-questions-school-sports-tatoos-work':
	        return 'fashion-pop_culture-sports-tatoos__faith-family-food-military-personal-questions-school-work'
	    elif category == 'fashion-pop_culture-sports-tatoos':
	        return 'fashion-sports-tatoos__pop_culture'
	    elif category == 'fashion-sports-tatoos':
	        return 'tatoos__fashion-sports'
	    elif category == 'fashion-sports':
	        return 'fashion__sports'
	    elif category == 'faith-family-food-military-personal-questions-school-work':
	        return 'questions__faith-family-food-military-personal-school-work'
	    elif category == 'faith-family-food-military-personal-school-work':
	        return 'faith__family-food-military-personal-school-work'
	    elif category == 'family-food-military-personal-school-work':
	        return 'food__family-military-personal-school-work'
	    elif category == 'family-military-personal-school-work':
	        return 'military__family-personal-school-work'
	    elif category == 'family-personal-school-work':
	        return 'school__family-personal-work'
	    elif category == 'family-personal-work':
	        return 'work__family-personal'
	    elif category == 'family-personal':
	        return 'personal__family'
	    elif category == 'sexuality-relationships':
	        return 'sexuality__relationships'
	    else:
	        return category

	# Loads a classifier that was pickled into the given file name
	def load_classifier(self,file_name):
		print "Loading model:" , file_name, '\r\n'
		classifier = cPickle.load(open(file_name,'rb'))
		data_file = ''.join(['tree-linsvc',file_name.replace('.csv.pkl','').replace('dumps','')])
		fit_data_text, fit_data_targets = self.data_handler.read_data(data_file)
		classifier.fit(fit_data_text,fit_data_targets)
		return classifier


	# Returns a collection of trained classifiers indexed by the composite file types fed into them.
	#   Note that dump files have this format: topic1__topic2.csv.pkl, so we index by topic1__topic2
	def init_classifiers(self,dump_files):
	    classifiers = {}
	    for dump in dump_files:
	        clf_name = dump.split('.')[0]
	        dump_name = '/'.join(['dumps',dump])
	        classifiers[clf_name] = self.load_classifier(dump_name)
	    return classifiers

	# Recursively applies classifiers until we get a non-composite category
	def composite_predict(self,sentence, classifiers, category):
	    if '-' not in category:
	        if category == 'sexuality':
	            return 'lgbtq'
	        elif category == 'questions':
	            return 'qna'
	        else:
	            return category
	    else:
	        classifier_key = self.category_to_classifier_name(category)
	        clf = classifiers[classifier_key]
	        new_category = clf.predict(sentence)
	        return self.composite_predict([sentence[0]],classifiers,new_category[0])


	# Sklearn SVM requires numbers for binary classifiers. File names for data are topic1__topic2.csv, so we split the filename by __
	# 	and see which topic we have.
	def binary_topic(self,category, list_of_cats):
	    if category == list_of_cats[0]:
	        return 0
	    else:
	        return 1

	# If there are less than 1000 samples, use a bayesian multinomial instead of an SVM
	def clf_type(self,num_samples):
		if num_samples >= 0:
			return ('clf',LinearSVC())
		else:
			return ('clf', MultinomialNB())

	# Given input and target data, trains a binary classifier and then persists it to the disk
	def train_classifier(self,filename, dump_folder):
		# Get data
		list_of_cats = filename.split('/')[2].split('__')
		reduced_filename = filename.replace('.csv','').replace('data/','')
		text_train, targets_train = self.data_handler.read_data(reduced_filename)
		targets_train = [self.binary_topic(x, list_of_cats) for x in targets_train]

		num_samples = len(text_train)
		clf_tuple = self.clf_type(num_samples)

		model = Pipeline([
			('tfidf', TfidfVectorizer(tokenizer=StemmaLemmaTokenizer())),
			clf_tuple,
		])

		model.fit(text_train, targets_train)

		print "Finished training model. \r\n"

		# Persist to disk
		print "Saving model to disk \r\n"
		dump_file = ''.join([dump_folder, '/',filename.split('/')[2],'.pkl'])
		cPickle.dump(model,open(dump_file, 'wb'))
		print "Finished saving model \r\n"

	# Train the tree of classifiers and save each of them to the disk
	def train(self):
		data_folder_name = 'data/tree-linsvc'
		dump_folder_name = 'dumps'
		data_files = os.listdir('data/tree-linsvc')

		if not os.path.exists(dump_folder_name):
			os.makedirs(dump_folder_name)

		for file_name in data_files: 
			print "Training model: ", file_name, '\r\n'
			full_name = '/'.join([data_folder_name, file_name])
			self.train_classifier(full_name, dump_folder_name)


	# Run the tree-classifier on the processed data.
	def run(self, text_test, targets_test):
		print "Running tree model."
		predictions = []

		dump_files = os.listdir('dumps')
		classifiers = self.init_classifiers(dump_files)

		top_level_category = 'animals-sexuality-relationships__faith-family-fashion-food-meetup-military-misc-personal-pop_culture-questions-school-sports-tatoos-work'

		for sentence in text_test:
			predictions.append(self.composite_predict([sentence], classifiers, top_level_category))

		print 'macro f1:', f1_score(targets_test,predictions, average='macro')
		print 'accuracy:', accuracy_score(targets_test, predictions)

		ls = ['animals','faith','family','fashion','food','lgbtq','meetup','military','misc','personal',
		'pop_culture','qna','relationships','school','sports','tatoos','work']
		conf_mat = confusion_matrix(targets_test, predictions, labels=ls)
		print 'confusion matrix: \r\n', conf_mat

	# Train and run the classifier
	def main(self):
		text_train, targets_train = self.data_handler.read_data('train')
		text_test, targets_test = self.data_handler.read_data('test')
		self.build_new_data(text_train,targets_train)
		self.train()
		self.run(text_test, targets_test)









