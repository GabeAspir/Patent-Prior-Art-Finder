from unittest import TestCase

from scipy.constants import pt

from _DevPatentPriorArtFinder import _DevPatentPriorArtFinder as dppaf
import AlternateMethods as alt
import pandas as pd
import re
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class Test(TestCase):

    def test_init(self):
        paf = dppaf()
        myCsv = open("../Patents/Day2-GoldSet.csv", "r")
        csvP = "../Patents/Day2-GoldSet.csv"
        test_setup = alt.setup(csvP)
        #print(test_setup)
        self.assertEqual(self.setup(myCsv),paf.init(csvP,'PublicationNumber'))

    def test__tokenize(self):
        paf = dppaf()
        test_table = {'Abstract': ["Doors are veRy similar to tables", "Doors are not very similar to cows",
                                    "Wood will cost 1.50 in Wallmart", "Wood won't cost 500 in Wallmart",
                                    "Once upon a time the foobat was done writing examples"],
                      'PublicationNumber': ["tables", "cows", "onefifty", "fivehumdred", "foobat"]
                      }
        test_tokenized= [["doors","similar","tables"],
                                    ["doors","similar","cows"],
                                    ["wood","cost","_NUM_","wallmart"],
                                    ["wood","cost","_NUM_","wallmart"],
                                    ["upon","time","foobat","done","writing","examples"]]
        frame = pd.DataFrame(test_table, columns= ['Abstract', 'PublicationNumber'])
        print(frame)
        paf._tokenize(frame)
        print(frame)
        self.assertEqual(frame['Tokens'].tolist(),test_tokenized)

    def test__tokenize_text(self):
        paf = dppaf()
        self.assertEqual(paf._tokenizeText("Doors are veRy similar to tables"),["doors","similar","tables"])

    def test__create_corpus(self):
        paf = dppaf()
        test_table = {'Abstract': ["Doors are veRy similar to tables", "Doors are not very similar to cows",
                                    "Wood will cost 1.50 in Wallmart", "Wood won't cost 500 in Wallmart",
                                    "Once upon a time the foobat was done writing examples"],
                      'Publication_number': ["tables", "cows", "onefifty", "fivehumdred", "foobat"]
                      }
        frame = pd.DataFrame(test_table, columns=['Abstract', 'Publication_number'])
        test_tokenized = [["doors", "similar", "tables"],
                                     ["doors", "similar", "cows"],
                                     ["wood", "cost", "_NUM_", "wallmart"],
                                     ["wood", "cost", "_NUM_", "wallmart"],
                                     ["upon", "time", "foobat", "done", "writing", "examples"]]
        frame["Tokens"]= test_tokenized

        corpus = ["doors","similar","tables","cows","done","wood","cost","_NUM_","wallmart","upon","time","foobat","writing","examples"]
        self.assertEqual(set(paf._createCorpus(frame)),set(corpus))

    # def test__create_new_corpus(self):
    #     self.fail()
    #
    def test__bag_of_wordize(self):
        paf = dppaf()
        bag= [[1,1,1,0,0,0,0,0,0,0,0,0,0],
              [1,1,0,1,0,0,0,0,0,0,0,0,0],
              [0,       0         ,0     ,0     ,1     ,1      ,1     ,1         ,0     ,0     ,0       ,0          ,0],
              [0,       0         ,0     ,0     ,1     ,1      ,1     ,1         ,0     ,0     ,0       ,0          ,0],
              [0,       0         ,0     ,0     ,0     ,0      ,0     ,0         ,1     ,1     ,1       ,1          ,1]]
        #     ["doors","simialr","tables","cows","wood","cost","_NUM_","wallmart","upon","time","foobat","writing","examples"]

    # def test__tfidfize(self):
    #     self.fail()
    #
    # def test__vectorize_tf_idf(self):
    #     self.fail()
    #
    # def test__appearences(self):
    #     self.fail()
    #
    # def test_jaccard_table(self):
    #     self.fail()
    #
    # def test_jaccard_similarity(self):
    #     self.fail()
    #
    # def test_cosine_similarity(self):
    #     self.fail()
    #
    # def test_cosine_table(self):
    #     self.fail()
    #
    # def test_compare_new_patent(self):
    #     self.fail()


