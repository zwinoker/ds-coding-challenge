# Wrapper class for all experiments. Allows for easy access to default settings for each classifier.

from DataHandler import DataHandler
from w2vops import W2VOps

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sltokenizer import StemmaLemmaTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

import TreeLinSVM

class Experiments(object):
	def __init__(self):
		self.data_handler = DataHandler()
		self.classifiers = {
			'svm': LinearSVC(),
			'baseline': LogisticRegression()
		}
		self.tokenizer = StemmaLemmaTokenizer()
		self.treeLinSVM = TreeLinSVM.TreeLinSVM()
		self.wv_database = '/Users/zacharywinoker/Downloads/GoogleNews-vectors-negative300.bin'
		
	# Initializes the word_vec_handler. Replace the argument with pretrained word2vec model of choice.
	def init_wv_handler(self):
		self.word_vec_handler = W2VOps(self.wv_database)

	# If expand==True, then generate expanded data using word vector similarity. Finally, feed expanded
	# 	data to the LinSVM model.
	# WARNING: TAKES VERY LONG. To illustrate how it works, 
	# 	change 'train' and 'test' to 'train-mini' and 'test-mini'. This will run on smaller datasets.
	def word_vector_expand(self, expand=False):
		self.init_wv_handler()

		# New data file names
		expanded_train_name = 'wv-expanded-train'
		expanded_test_name = 'wv-expanded-test'

		if expand:
			# Read in un-expanded data.
			text_train, targets_train = self.data_handler.read_data('train-mini')
			text_test, targets_test = self.data_handler.read_data('test-mini')

			# Expand the data.
			full_train_filename = ''.join(['data/', expanded_train_name, '.csv'])
			full_test_filename = ''.join(['data/', expanded_test_name, '.csv'])
			self.word_vec_handler.wv_expand_text(full_train_filename,text_train,targets_train)
			self.word_vec_handler.wv_expand_text(full_test_filename,text_train,targets_train)

		# Run SVM model with expanded data.
		self.run_model('svm', expanded_train_name, expanded_test_name)

	# Clusters the 'misc' and 'personal' categories with K-means on sentence-average word vectors.
	# 	Then runs LinSVM on the results. Afterwards, lifts subcategories to 'misc' and 'personal' to 
	# 	evaluate final results. 
	def word_vector_subcategory(self):
		self.init_wv_handler()
		train_data_with_subcategories = 'train-with-clusters'
		text_train, targets_train = self.data_handler.read_data('train')
		text_test, targets_test = self.data_handler.read_data('test')
		self.word_vec_handler.subcategorize_catchall(text_train, targets_train, train_data_with_subcategories)
		self.run_model('svm', train_data_with_subcategories, 'test', sub_categories=True)

	# Converts misc_i to misc and personal_i to personal if using subcategories.
	def lift_subcategories(self, cat):
		if 'misc' in cat:
			return 'misc'
		elif 'personal' in cat:
			return 'personal'
		else:
			return cat

	# Runs the linear SVM model with TF-IDF features.
	def run_LinSVM(self):
		self.run_model('svm','train','test')

	# Tells user how to run the CNN model. 
	def run_CNN(self):
		print "\r\nTo run the CNN model, execute the following commands in the console:"
		print "python CNN_process_data.py {path/to/word2vec/model}"
		print "THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec\r\n"

	def run_TreeSVM(self):
		self.treeLinSVM.main()

	# Runs the baseline model with TF-IDF features.
	def run_baseline(self):
		self.run_model('baseline','train','test')

	# Runs either the baseline or a linear-SVM model and prints results to console. If sub_categories == True,
	# 	then convert misc_i -> misc and personal_i -> personal in predicted categories
	def run_model(self, model_name, train_data_file, test_data_file, sub_categories = False):
		# Read in data.
		text_train, targets_train = self.data_handler.read_data(train_data_file)
		text_test, targets_test = self.data_handler.read_data(test_data_file)
		
		# Set up and fit model.
		param_grid = [
			{'tfidf__norm': ['l2'],  'tfidf__tokenizer': [self.tokenizer]}
		]
		model = Pipeline([
			('tfidf', TfidfVectorizer()),
			('clf', self.classifiers[model_name])
		])
		gs_model = GridSearchCV(model, param_grid)
		gs_model.fit(text_train, targets_train)

		# Evaluate model.
		prediction = gs_model.predict(text_test)
		if sub_categories:
			prediction = [self.lift_subcategories(x) for x in prediction]

		print 'macro f1:', f1_score(targets_test,prediction, average='macro')
		print 'accuracy:', accuracy_score(targets_test, prediction)
		ls = ['animals','faith','family','fashion','food','lgbtq','meetup','military','misc','personal',
		'pop_culture','qna','relationships','school','sports','tatoos','work']
		conf_mat = confusion_matrix(targets_test, prediction, labels=ls)
		print 'confusion matrix: \r\n', conf_mat


if __name__ == '__main__':
	exp = Experiments()
	exp.run_LinSVM()
	# exp.run_baseline()
	# exp.word_vector_subcategory()
	# exp.run_CNN()
	# exp.run_TreeSVM()
	# exp.word_vector_expand(True)
	
	




