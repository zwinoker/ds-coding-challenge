# Reads, cleans, and otherwise modifies data. 

from csv import DictReader

class DataHandler():
	def __init__(self):
		pass

	def read_data(self, name):
	    text, targets = [], []
	    with open('data/{}.csv'.format(name)) as f:
	        for item in DictReader(f):
	            text.append(item['text'].decode('utf8'))
	            targets.append(item['category'])
	    return text, targets

	def scrub(self,line) :
		scrubbed_data = re.sub(r'([^\s\w]|_)+','', line.strip('\r\n').replace(',', ' '))
		return scrubbed_data