# Tokenizer for text processing.

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

class StemmaLemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.rt = RegexpTokenizer(r'\w+')
    def __call__(self, doc):
        lems = [self.wnl.lemmatize(t) for t in self.rt.tokenize(doc)]
        return [self.ps.stem(t) for t in lems]
    def tokenize(self, doc):
    	lems = [self.wnl.lemmatize(t) for t in self.rt.tokenize(doc)]
        return [self.ps.stem(t) for t in lems]

