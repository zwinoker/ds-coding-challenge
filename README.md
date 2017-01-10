1. OVERVIEW

This is a readme for code I wrote for a data scientist coding challenge. See the paper at http://zackwinoker.com/whisper

Before using the code, please refer to the white paper for details on the methods used. 

Experiments are done through a class 'Experiments' in experiments.py. This class has functions for running all of the models I employed. For example, to run the baseline model, use experiments.run_baseline(). 

Data i/o is handled with the DataHandler class and word vector operations are handled with the w2vops class. Note that word vector text expansion is time-intensive, so reduced datasets have been provided for the sake of demonstrating functionality. These are 'train-mini.csv' and 'test-mini.csv'. A tokenizer I used is provided in sltokenizer.py


2. RUNNING THE MODELS

Note that to use the CNN model or any of the word vector models, you will need a local copy of some pretrained word vectors. The one I used is available at https://code.google.com/archive/p/word2vec/. To use, change experiments.wv_database to the path to your local copy. 

To run the baseline model, use experiments.run_baseline().

To run the linear SVM model, use experiments.run_LinSVM().

To run the word vector expansion model, use experiments.word_vector_expand(expand). expand is a boolean that determines whether or not to run the expansion process. If you have already expanded the text on a previous run, set expand to false. That is, run experiments.word_vector_expand(False).

To run the model with catchall subcategories, use experiments.word_vector_subcategory().

To run the tree model, use experiments.run_TreeSVM().

Instructions for running the CNN model are found by running experiments.run_CNN(). Note, this usually took a few hours to run on my laptop.

