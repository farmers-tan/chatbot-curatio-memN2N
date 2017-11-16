import unittest
from data_utils import *

class TestDataUtilsMethods(unittest.TestCase):
	def test_get_dialogs(self):
		print get_dialogs("data/1-1-QA-without-context/dialog-babi-task1-API-calls-trn.txt")[0]
	def test_load_dialog_task(self):
		train_data, test_data, val_data=load_dialog_task("data/1-1-QA-without-context",1)
		print len(train_data)
		print len(test_data)
		print len(val_data)
	def test_load_candidates(self):
		candidates,d=load_candidates("data/1-1-QA-without-context",1)
		print(d['api_call italian rome six cheap'])

if __name__ == '__main__':
    unittest.main()