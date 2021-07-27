from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim import models
from gensim.corpora import Dictionary
from timeit import default_timer as timer
import gzip
import json

class _DevFilesPatentPriorArtFinder:
    def __init__(self, dirPath, publicationNumberColumnString='Publication_Number', comparisonColumnString='Abstract', cit_col= "Citations"):
        startTime = timer()
        self.corpus = []
        self.number_of_patents_with_word = {}
        self.plain_dataframe = None
        self.dataframe = None
        self.word_count_matrix = None
        self.model_words = None
        self.model_citations = None
        self.tfidf_gensim = models.TfidfModel()
        self.dictionary = None
        self.dirPath= dirPath
        if dirPath is None:
            raise IOError('The passed file path was empty')
        self.id_col = publicationNumberColumnString
        self.txt_col = comparisonColumnString
        self.cit_col = cit_col

        # Create the folders for metadata files, and will pass should an error thown when the directory exists from a previous object
        try:
            os.mkdir(dirPath+"\meta")
            os.mkdir(dirPath+"\w2v")
            os.mkdir(dirPath+"\other")
        except:
            print("Didn't make directories")
            pass
        print("Initialization complete T="+str(timer()))

    def train(self):
        # Iterates over the files in the directory twice.
        # Once to save the tokenization column to the file, and adds the file to the model's training
        # 2nd time to append the w2v encodings generated from the fully trained model to the files.
        print("Training has begun")
        self.dictionary= Dictionary()
        first=True
        for entry in os.scandir(self.dirPath):
            if entry.is_file():    # To avoid entering the directories
                print("tokenizing "+ str((entry)))
                self._makeModel(entry,first)
        print("Tokenization Completed T="+str(timer()))

        for entry in os.scandir(self.dirPath):
            if entry.is_file():
                print("getting embedding of "+str(entry)+" T="+str(timer()))
                self._makeEmbeddings(entry)
        print("Embeddings completed"+str(timer()))
        self.dictionary.save_as_text(self.dirPath + "\other\\dict.txt")

    def is_gz_file(self, filepath):
        with open(filepath, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'

    # Private methods for train to call
    def _makeModel(self,file, first):
        try:
            dataframe= pd.io.json.read_json(file,compression="gzip")
        except:
            #print('here before dataframe')
            dataframe= pd.io.json.read_json(file,compression="gzip",lines=True)
            #dataframe= pd.DataFrame.from_records(file)
            #print(dataframe)
            #print('here after dataframe')
        dataframe['Tokens'] = dataframe[self.txt_col].apply(self._tokenizeText)
        dataframe['TokenizedCitations'] = dataframe['Citations'].apply(self._tokenizeCitation)
        words = 0
        for index,doc in dataframe.iterrows():
            words += len(doc["Tokens"])
        print(str(file)+" has "+ str(words) +" word tokens")
        self.dictionary.add_documents(dataframe['Tokens'])

        print("Writing "+str(file))
        dataframe.to_json(self.get(file,"meta"), orient='records', indent=4)

        if first:
            first= False
            model_words = Word2Vec(dataframe['Tokens'])
            self.model_words = model_words
            model_citations = Word2Vec(dataframe['TokenizedCitations'], min_count=1)
            self.model_citations = model_citations
        else:
            self.model_words.build_vocab(dataframe["Tokens"], update=True)
            self.model_words.train(dataframe["Tokens"], total_examples=self.model_citations.corpus_count,
                                   epochs=self.model_words.epochs)
            self.model_words.build_vocab(dataframe["TokenizedCitations"], update=True)
            self.model_citations.train(dataframe['TokenizedCitations'],
                                       total_examples=self.model_citations.corpus_count,
                                       epochs=self.model_citations.epochs)

    def _tokenizeCitation(self, string):
        no_commas = string.replace(',',' ')
        tokenized = word_tokenize(no_commas)
        finished = []
        for token in tokenized:
            str = self._removeSuffix(token)
            finished.append(str)
        return list(set(finished))

    def _removeSuffix(self, string):
        tokens = string.split('-')
        # Should always have at least a prefix and the patent, this will take away the suffix or keep it the same
        return str(tokens[0] + '-' + tokens[1])

    # Will add column to dataframe called 'Tokens'
    def _tokenizeText(self, string):
        #prepares the string for tokenization, to lowercase, then removes punctutation, then changes numbers to _NUM_
        string = string.lower()
        string = re.sub(r"\d+\.?\d*", " _NUM_ ", string)
        string = re.sub(r'[^\w\s]', '',string)
        stop_words = set(stopwords.words("english"))
        tokenized = word_tokenize(string)
        return [word for word in tokenized if not word.lower() in stop_words]


    def _makeEmbeddings(self, file):
        try:
            dataframe= pd.io.json.read_json(self.get(file,"meta"), orient = 'records', lines=True)
        except:
            dataframe= pd.io.json.read_json(self.get(file,"meta"), orient = 'records')

        corpus = [self.dictionary.doc2bow(line) for line in dataframe['Tokens']]
        self.tfidf_gensim = models.TfidfModel(corpus)
        dataframe["TF-IDF"] = [self.tfidf_gensim[corpus[x]] for x in range(0, len(corpus))]
        dataframe.to_json(self.get(file,"meta"), orient = 'records', indent=4)
        vecs =[]
        vecs_tf = []
        for (tokenList, citationList, tfidfList) in zip(dataframe['Tokens'], dataframe['TokenizedCitations'], dataframe["TF-IDF"]):
            sum_words = np.empty(50)
            tfidf_sum = np.empty(50)
            sum_citations = np.empty(50)
            tfidfDict = dict(tfidfList)
            for word in tokenList:
                index = self.dictionary.token2id.get(word)
                tfidfValue = tfidfDict.get(index)
                try:
                    tfidf_sum += [val * tfidfValue for val in self.model_words.wv[word]]
                    sum_words += (self.model_words.wv[word])
                except:
                    pass
            for citation in citationList:
                try:
                    sum_citations += self.model_citations.wv[citation]
                except:
                    pass
            sum = np.concatenate((sum_words,sum_citations))
            vecs.append(sum)
            sum = np.concatenate((tfidf_sum,sum_citations))
            vecs_tf.append(sum)
        vec_frame =  pd.DataFrame(dataframe[self.id_col])
        vec_frame['Word2Vec'] = vecs
        vec_frame['w2v Modified']= vecs_tf
        vec_frame.to_json(self.get(file,"w2v"), orient = 'records', indent=4)


    @staticmethod
    def get(entry, folder):
        head, tail = os.path.split(entry.path)
        return head + "\\"+folder+"\\" + tail

    def cosineSimilarity(self, patent1, patent2):
        """
        Computes the cosine similarity of 2 patents. Used in CompareNewPatent below.
        Creates a 2d array to match the input requirements of scikit's cosine func,
         but only returns the 1 cell representing the 2 inputted patents
        :param patent1: Vector representing the text of 1 patent's text
        :param patent2: Vector representing the text of a 2nd patent's text
        :return: The cosine similarity index of those 2 patents
        """
        if patent1 is None or patent2 is None:
            raise IOError("One of or both of the Patents are empty")
        # elif type(patent1) is not list:
        #     raise IOError("Patent input must be a list, not "+str(type(patent1)))
        elif len(patent1) != len(patent2):
            raise IOError("Bag of Words must be the same length")
        v1 = np.array(patent1).reshape(1, -1)
        v2 = np.array(patent2).reshape(1, -1)
        return cosine_similarity(v1, v2)[0][0]


    # Comparing new patent based on TF-IDF/Cosine Similarity
    # dataframe must have TF-IDF column
    def compareNewPatent(self, newPatentSeries, dirPath, threshold):
        newPatentSeries['Tokens'] = self._tokenizeText(string=newPatentSeries['Abstract'])
        newPatentSeries['TokenizedCitations']= self._tokenizeCitation(string=newPatentSeries['Citations'])
        sum_words = np.empty(50)
        sum_tfidf= np.empty(50)
        sum_citations = np.empty(50)
        if(self.tfidf_gensim is None):
            self.tfidf_gensim = Dictionary.load_from_text(dirPath+"\other\dict.txt")

        for word in newPatentSeries['Tokens']:
            try:
                sum_tfidf += [val * tfidfValue for val in self.model_words.wv[word]]
                sum_words += self.model_words.wv[word]
            except:
                pass
        for citation in newPatentSeries['TokenizedCitations']:
            try:
                sum_citations += self.model_citations.wv[citation]
            except:
                pass
        sum = np.concatenate((sum_words,sum_citations))
        newPatentSeries['Word2Vec']= sum

        matches = []
        for file in os.scandir(dirPath+"\w2v"):
            if file.is_file(): # To avoid entering the emb directory
                print("reading "+str(file))
                try:
                    dataframe = pd.io.json.read_json(file, orient='records', lines=True)
                except:
                    dataframe = pd.io.json.read_json(file, orient='records')
                for index,doc in dataframe.iterrows():
                    #print(doc['Word2Vec'])
                    #print(newPatentSeries['Word2Vec'])
                    if self.cosineSimilarity(newPatentSeries['Word2Vec'], doc['Word2Vec']) >= threshold:
                        matches.append(doc)
        return matches