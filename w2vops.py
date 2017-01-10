# Handles data modification involving word vectors. 

import gensim
import numpy as np
import re
import warnings
from sklearn.cluster import KMeans

class W2VOps():
	def __init__(self, pretrained_word_vecs):
		self.model = gensim.models.Word2Vec.load_word2vec_format(pretrained_word_vecs, binary=True)

	# Performs the word-vector-expansion process on a given sentence
	def wv_expand_sentence(self,sentence):
		num_expansion_words = 100
		expanded_sentence = []
		words = sentence.split(' ')
		for word in words:
			if word != ' ':
				# Small parsing issue solution
				word = word.replace('\r\n', ' ')
				word = word.split(' ')
				for w in word:
					# Add related words if w in model
					if w in self.model:
						expanded_sentence.append(w)
						related_words = [x[0] for x in self.model.most_similar(w,topn=num_expansion_words)]
						map(lambda x: expanded_sentence.append(x), related_words)
		return ' '.join(expanded_sentence)

	# Given text and targets, will write expanded text to the given filename. 
	def wv_expand_text(self, filename, text, targets):
		# Expand using word vector similarity
		write_file = open(filename,'w')
		write_file.write('category,text\r\n')
		for idx, line in enumerate(text) :
			full_line = ''.join([targets[idx],',',self.wv_expand_sentence(line).encode('utf8'),'\r\n'])
			write_file.write(full_line)

	# Constructs the average word vector for a given sentence.
	def avg_vector(self, words) :
		vecs = []
		for word in words:
			if word in self.model:
				vecs.append(self.model[word])
			else :
				vecs.append(np.random.uniform(low=-0.25, high=0.25, size=(300,)))
		v_matrix = np.array(vecs)
		v_avg = np.average(v_matrix, axis=0)
		if v_avg.ndim == 0:
			v_avg = np.random.uniform(low=-0.25, high=0.25, size=(300,))
		return v_avg

	# Takes in matrix of average word vectors and clusters them with K-Means
	def cluster_catchalls(self, vecs, num_clusters):
		vec_array = np.asarray(vecs)
		km = KMeans(n_clusters = num_clusters)
		km.fit(vec_array)
		return km.labels_.tolist()

	# Finds subcategories for the 'misc' and 'personal' categorires in training data and writes results 
	# 	to a new file.
	def subcategorize_catchall(self, text_train, targets_train, filename):
		num_misc_clusters = 3
		num_personal_clusters = 1
		misc_lines, personal_lines, misc_vecs, personal_vecs = [], [], [], []

		# Modify training data based on word-vector clusters
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			full_filename = ''.join(['data/', filename, '.csv'])
			modified_data_file = open(full_filename, 'w')
			modified_data_file.write('category,text\r\n')

			# Compute avg word vectors for each sentence.
			for idx, line in enumerate(text_train):
				if targets_train[idx] == 'misc':
					avg_vec = self.avg_vector(line)
					misc_vecs.append(avg_vec)
					misc_lines.append(line)
					
				elif targets_train[idx] == 'personal':
					avg_vec = self.avg_vector(line)
					personal_vecs.append(avg_vec)
					personal_lines.append(line)

				else:
					modified_data_file.write(''.join([targets_train[idx], ',', line.encode('utf8'), '\r\n']))

			# Find clusters
			misc_clusters = self.cluster_catchalls(misc_vecs, num_misc_clusters)
			personal_clusters = self.cluster_catchalls(personal_vecs, num_personal_clusters)

			# Write new data with cluster info in the categories (e.g. 'misc_i')
			for idx, cluster in enumerate(misc_clusters):
				subcategory = ''.join(['misc_', str(cluster)])
				modified_data_file.write(''.join([subcategory,',', misc_lines[idx].encode('utf8'), '\r\n']))
			for idx, cluster in enumerate(personal_clusters):
				subcategory = ''.join(['personal_', str(cluster)])
				modified_data_file.write(''.join([subcategory,',', personal_lines[idx].encode('utf8'), '\r\n']))





