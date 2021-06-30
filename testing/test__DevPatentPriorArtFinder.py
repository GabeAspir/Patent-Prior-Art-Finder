from unittest import TestCase

from scipy.constants import pt

from _DevPatentPriorArtFinder import _DevPatentPriorArtFinder as dppaf
import pandas as pd
import re
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class Test(TestCase):
    paf = dppaf()
    myCsv = open("../Patents/Day2-GoldSet.csv", "r")
    csvP= "../Patents/Day2-GoldSet.csv"

    def test_init(self):
        paf = dppaf()
        csvP = "../Patents/Day2-GoldSet.csv"
        self.assertEqual(self.setup(),paf.init(csvP,'Publication_Number'))
    test_table = []

    def test__tokenize(self):
        paf = dppaf()
        test_table = {'Abstract': ["Doors are veRy similar to tables", "Doors are not very similar to cows",
                                    "Wood will cost 1.50 in Wallmart", "Wood won't cost 500 in Wallmart",
                                    "Once upon a time the foobat was done writing examples"],
                      'Publication_number': ["tables", "cows", "onefifty", "fivehumdred", "foobat"]
                      }
        test_tokenized= [["Doors","similar","tables"],
                                    ["doors","similar","cows"],
                                    ["wood","cost","_NUM_","wallmart"],
                                    ["wood","cost","_NUM_","wallmart"],
                                    ["upon","time","foobat","done","writing","examples"]]
        frame = pd.DataFrame(test_table, columns= ['Abstract', 'Publication_number'])
        paf._tokenize(frame)
        print(frame)
        self.assertEqual(frame['Tokens'].tolist(),test_tokenized) # <<< Add input, need to change output

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

    # Independent versions of the methods for testing purposes:
    def setup(aCSV, docCol='Abstract', indexCol='Publication_Number'):
        # I added '' to the stopwords to avoid the case where a short first word turns into an empty list item
        stop_words = {'', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                      "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                      'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                      'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
                      'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                      'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                      'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                      'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                      'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                      'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                      'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                      "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                      'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                      "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

        def tokenize(string):
            out = string.lower()
            out = re.sub(r'\b\w{1,2}\b', '', out)  # remove anything not a word of length 2+
            out = re.sub(r"[0-9]+", "_NUM_", out)  # substitute _NUM_ for any block of consecutive number chars
            words = re.split('\W+', out)  # Might need to change to pandas split at some point
            # Note capital W is "Not word"= [a-zA-Z0-9_]
            words = list(filter(lambda s: s not in stop_words, words))  # why list not set?  ¯\_(ツ)_/¯
            return words

        def tokenizer(dataframe):
            dataframe['Tokens'] = dataframe['Publication_Number']  # Create column to be replaced with tokenized text
            for index, row in pt.iterrows():
                dataframe['Tokens'][index] = tokenize(dataframe['Abstract'][index])

        def getCorpus(dataframe):
            corpus = set()
            for r in dataframe['Tokens']:
                corpus.update(r)
            return corpus

        def bow(series, corpus=None):  # Takes a tokenized series
            if corpus is None:
                corpus = getCorpus(series)
            counts = []
            for r in series:
                count = {}
                for w in corpus:
                    count[w] = r.count(w)
                counts.append(count)
            return counts

        def getTf(dataframe):
            documents = dataframe['Abstract']
            tfidf = TfidfVectorizer().fit_transform(documents)
            return tfidf

        df = pd.read_csv(aCSV)
        df['Tokens'] = tokenizer(df[docCol])
        df['BagOfWords'] = bow(df['Tokenized'])
        df['TF-IDF'] = getTf(df)