#from nltk.corpus import wordnet
from PyDictionary import PyDictionary
from data_utils import load_dialog_task, load_candidates
 

class nltkHelper(object):
	def __init__(self, data_dir, task_id):
		self.data_dir = data_dir
		self.task_id = task_id

		candidates, self.candid2indx = load_candidates(
			self.data_dir, self.task_id)
		self.n_cand = len(candidates)
		print("Candidate Size", self.n_cand)
		self.indx2candid = dict(
		    (self.candid2indx[key], key) for key in self.candid2indx)
		# task data
		self.trainData, self.testData, self.valData = load_dialog_task(
		    self.data_dir, self.task_id, self.candid2indx, False)
		self.data = self.valData
		self.banned_words = ["i", "the"]
		self.pyD = PyDictionary()

	def find_synonyms(self, word):
		return self.pyD.synonym(word)

	# def recursive_find(query, syn, index):
	# 	# find a way to implement the recursive function
	# 	for i, w in enumerate(query):
	# 		continue if (i == index)
	# 		syns = find_synonyms(w)

	# 		for syn in syns:
	# 			print(query)
	# 			query[i] = syn
	# 			recursive_find(query, syn, i+1)

	def generate_queries(self, file):
		self.data.sort(key=lambda x:len(x[0]),reverse=True)
		join_string = " "
		for i, (story, query, answer) in enumerate(self.data):
			for j, w in enumerate(query):
				temp_query = list(query)
				syns = self.find_synonyms(w)
				if w in self.banned_words:
					continue
				if syns == None:
					continue
				# Generate syns for each word and then write to file the changed query and answer every syn
				# Should be a recursion to make use of every synonym commbinations of every word
				for syn in syns:
					temp_query[j] = syn
					#print(syn)
					#print("1 " + join_string.join(temp_query) + "\t" + self.indx2candid[answer] + "\n\n")
					file.write("1 " + join_string.join(temp_query) + "\t" + self.indx2candid[answer] + "\n\n")

	def generate_answers(self, file):
		self.data.sort(key=lambda x:len(x[0]),reverse=True)
		join_string = " "
		for i, (story, query, answer) in enumerate(self.data):
			for j, w in enumerate(answer):
				temp_query = list(query)
				syns = self.find_synonyms(w)
				if w in self.banned_words:
					continue
				if syns == None:
					continue
				# Need to have a framework for generating answers
				# for syn in syns:
				# 	temp_query[j] = syn
				# 	#print(syn)
				# 	#print("1 " + join_string.join(temp_query) + "\t" + self.indx2candid[answer] + "\n\n")
				# 	file.write("1 " + join_string.join(temp_query) + "\t" + self.indx2candid[answer] + "\n\n")


if __name__ == '__main__':
	task_id = 7
	data_dir = "data/1-1-QA-without-context/"

	gen_data = nltkHelper(data_dir, task_id)
	f = open(data_dir + "gen_data_" + str(task_id) + "_test.txt", "w")
	gen_data.generate_queries(f)
	f.close()