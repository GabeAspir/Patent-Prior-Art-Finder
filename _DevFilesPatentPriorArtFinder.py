import scipy.spatial.distance
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
import nltk.downloader
nltk.download('stopwords')


class _DevFilesPatentPriorArtFinder:
    def __init__(self, dirPath, publicationNumberColumnString='Publication_Number', comparisonColumnString='Abstract', cit_col= "Citations"):
        start_time = timer()
        self.model_words = None
        self.model_citations = None
        self.dictionary = Dictionary()
        self.dirPath= dirPath
        if dirPath is None:
            raise IOError('The passed file path was empty')
        self.id_col = publicationNumberColumnString
        self.txt_col = comparisonColumnString
        self.cit_col = cit_col
        self.old = True
        self.first = True
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
        for entry in os.scandir(self.dirPath):
            if entry.is_file():    # To avoid entering the directories
                print("tokenizing "+ str((entry)))
                self._makeModel(entry,self.first)
        print("Tokenization Completed T="+str(timer()))
        self.tfidf_model= models.TfidfModel(dictionary=self.dictionary)
        for entry in os.scandir(self.dirPath):
            if entry.is_file():
                print("getting embedding of "+str(entry)+" T="+str(timer()))
                self._makeEmbeddings(entry)
        print("Embeddings completed"+str(timer()))
        self.model_words.save(self.dirPath + "\other\\model_words.model")
        self.model_citations.save(self.dirPath + "\other\\model_citations.model")
        self.dictionary.save_as_text(self.dirPath + "\other\\dict.txt")
        self.old=False
    def is_gz_file(self, filepath):
        with open(filepath, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'
    # Private methods for train to call
    def _makeModel(self,file, first):
        try:
            dataframe= pd.io.json.read_json(file,compression="gzip")
        except:
            dataframe= pd.io.json.read_json(file,compression="gzip",lines=True)
        #dataframe= pd.DataFrame(index=dataframe[self.id_col])
        dataframe['Tokens'] = dataframe[self.txt_col].apply(self._tokenizeText)
        dataframe['TokenizedCitations'] = dataframe['Citations'].apply(self._tokenizeCitation)
        # words = 0
        # for index,doc in dataframe.iterrows():
        #     words += len(doc["Tokens"])
        # print(str(file)+" has "+ str(words) +" word tokens")
        self.dictionary.add_documents(dataframe['Tokens'])
        print("Writing "+str(file))
        dataframe.to_json(self.get(file,"meta"), orient='records', indent=4)
        if first:
            self.first= False
            # TODO: This local model_words seems redundant
            model_words = Word2Vec(dataframe['Tokens'],vector_size=50)
            self.model_words = model_words
            model_citations = Word2Vec(dataframe['TokenizedCitations'], min_count=1)
            self.model_citations = model_citations
        else:
            self.model_words.build_vocab(dataframe["Tokens"], update=True)
            self.model_words.train(dataframe["Tokens"], total_examples=self.model_words.corpus_count, epochs=self.model_words.epochs)
            self.model_citations.build_vocab(dataframe["TokenizedCitations"], update=True)
            self.model_citations.train(dataframe['TokenizedCitations'],total_examples=self.model_citations.corpus_count, epochs=self.model_citations.epochs)
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
        # Replaced with 1 time generation in train() between the loops
        #self.tfidf_gensim = models.TfidfModel(corpus)
        dataframe["TF-IDF"] = [self.tfidf_model[corpus[x]] for x in range(0, len(corpus))]
        dataframe.to_json(self.get(file,"meta"), orient = 'records', indent=4)
        vecs =[]
        # print("make embeddings")
        # print(self.model_words.wv.most_similar('computer', topn=10))
        # print(self.model_words.wv["computer"])
        for (tokenList, citationList, tfidfList) in zip(dataframe['Tokens'], dataframe['TokenizedCitations'], dataframe["TF-IDF"]):
            sum_words = np.zeros(50)
            sum_citations = np.zeros(50)
            sum_tfidf = np.empty(50)
            tfidfDict = dict(tfidfList)
            # Create a sum of the words in a given document to create a doc vector
            # Maintain 2 such vectors: 1 plain, and another where each word vector is multiplied by the word's tfidf weight
            for word in tokenList:
                index = self.dictionary.token2id.get(word)
                tfidfValue = tfidfDict.get(index)
                try:
                    sum_words= np.add(sum_words ,self.model_words.wv[word])
                    sum_tfidf= np.add(sum_tfidf,np.multiply(tfidfValue,self.model_words[word]))
                except: # In case the model does not have a given word, ignore it.
                    pass
            for citation in citationList:
                try:
                    sum_citations = np.add(sum_citations,self.model_citations.wv[citation])
                except:
                    pass
            sum = np.concatenate((sum_words,sum_citations))
            vecs.append(sum)
        vec_frame =  pd.DataFrame(dataframe[self.id_col])
        vec_frame['Word2Vec'] = vecs
        vec_frame.to_json(self.get(file,"w2v"), orient = 'records', indent=4)
    @staticmethod
    def get(entry, folder): # To easily access the meta-data files in parallel folders
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
        sum_words = np.zeros(50)
        sum_citations = np.zeros(50)
        sum_tfidf = np.empty(50)
        if self.old:
            dictFile = dirPath + "\other\dict.txt"
            self.dictionary = Dictionary.load_from_text(dictFile)
            #self.tfidf_gensim = models.TfidfModel(dictionary=self.dictionary)
            self.tfidf_model = models.TfidfModel(dictionary=self.dictionary)
            self.model_words= models.Word2Vec.load(dirPath + "\other\\model_words.model")
            self.model_citations= models.Word2Vec.load(dirPath + "\other\\model_citations.model")
            # print("comp new patent")
            # print(self.model_words.wv.most_similar('computer', topn=10))
        # print('before')
        # print(self.model_words.wv["invention"])
        # print(sum_words)
        print(self.model_citations.wv["US-2008111420"])
        tfidf_vector = self.tfidf_model[self.dictionary.doc2bow(newPatentSeries['Tokens'])]
        tfidfDict = dict(tfidf_vector)
        for word in newPatentSeries['Tokens']:
            index = self.dictionary.token2id.get(word)
            tfidfValue = tfidfDict.get(index)
            try:
                sum_words= np.add(sum_words, self.model_words.wv[word])
                #sum_tfidf += [val * tfidfValue for val in self.model_words.wv[word]]
            except:
                pass
        for citation in newPatentSeries['TokenizedCitations']:
            try:
                sum_citations= np.add(sum_citations,self.model_citations.wv[citation])
            except:
                pass
        sum = np.concatenate((sum_words,sum_citations))
        newPatentSeries['Word2Vec']= sum
        print("New pat w2v")
        print(newPatentSeries["Tokens"])
        print(sum)
        matches = []
        for file in os.scandir(dirPath+"\w2v"):
            if file.is_file(): # To avoid entering the emb directory
                print("reading "+str(file))
                try:
                    dataframe = pd.io.json.read_json(file, orient='records', lines=True)
                except:
                    dataframe = pd.io.json.read_json(file, orient='records')
                for index,doc in dataframe.iterrows():
                    # print(doc['Word2Vec'])
                    # print(newPatentSeries['Word2Vec'])
                    try:
                        similarity = 1 - scipy.spatial.distance.cosine(newPatentSeries['Word2Vec'], doc['Word2Vec'])
                    except:
                        print("Vec for doc "+doc+" @index "+str(index))
                        print(doc['Word2Vec'])
                    if similarity >= threshold:
                        matches.append(doc)
                        #matches.append((similarity, doc))
        print(str(len(matches))+" Matches found")
        return matches
        #return sorted(matches, key=lambda similarity: similarity[0], reverse=True)