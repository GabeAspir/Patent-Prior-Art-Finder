from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.text import TextCollection
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
nltk.download('punkt')
nltk.download('stopwords')

class _DevNLTKPatentPriorArtFinder:

    def __init__(self, dirPath, publicationNumberColumnString='PublicationNumber', comparisonColumnString='Abstract', cit_col= "Citations"):
        self.corpus = []
        self.number_of_patents_with_word = {}
        self.plain_dataframe = None
        self.dataframe = None
        self.word_count_matrix = None
        self.model_words = None
        self.model_citations = None

        if dirPath is None:
            raise IOError('The passed file path was empty')

        self.id_col = publicationNumberColumnString
        self.txt_col = comparisonColumnString
        self.cit_col = cit_col

        # Iterates over the files in the directory twice.
        # Once to save the tokenization column to the file, and adds the file to the model's training
        # 2nd time to append the w2v encodings generated from the fully trained model to the files.
        for entry in os.scandir(dirPath):
            self._addtoModel(entry)
        for entry in os.scandir(dirPath):
            self._getEmbeding(entry)

    # Private methods for init to call
    def _addtoModel(self,file):
        dataframe= pd.io.json.read_json(file)
        dataframe['Tokens'] = dataframe[self.txt_col].apply(self._tokenizeText)
        dataframe['TokenizedCitations'] = dataframe['Citations'].apply(self._tokenizeCitation)
        dataframe.to_json(file)
        self.model_words.train(dataframe["Tokens"])
        self.model_citations.train(dataframe['TokenizedCitations'])


    def _tokenizeCitation(self, string):
        no_commas = string.replace(',',' ')
        tokenized = word_tokenize(no_commas)
        finished = []
        for token in tokenized:
            str = self._takeAwaySuffix(token)
            finished.append(str)
        return list(set(finished))

    def _takeAwaySuffix(self, string):
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

    def _getEmbeding(self,file):
        data= pd.io.json.read_json(file)
        text_tokens = data['Tokens']
        citation_tokens = data['TokenizedCitations']
        vecs =[]

        for (tokenList, citationList) in zip(text_tokens, citation_tokens):
            sum_words = np.empty(50)
            sum_citations = np.empty(50)
            for word in tokenList:
                try:
                    sum_words += self.model_words.wv[word]
                except:
                    pass
            for citation in citationList:
                try:
                    sum_citations += self.model_citations.wv[citation]
                except:
                    pass
            sum = np.concatenate(sum_words,sum_citations)
            vecs.append(sum)
        data['Word2Vec'] = vecs
        data.to_json(file)


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
        elif type(patent1) is not list:
            raise IOError("Patent input must be a list")
        elif len(patent1) != len(patent2):
            raise IOError("Bag of Words must be the same length")

        v1 = np.array(patent1).reshape(1, -1)
        v2 = np.array(patent2).reshape(1, -1)
        return cosine_similarity(v1, v2)[0][0]


    # Comparing new patent based on TF-IDF/Cosine Similarity
    # dataframe must have TF-IDF column

    def compareNewPatent(self, newComparisonText, dataframe):
        """
        Takes a new patent and the old dataframe, updates the dataframe with the new patent and any new words it adds, returns a tfidf comparison table

        :param newComparisonText: The new patent to be added to the comparison
        :param dataframe: The old dataframe (with metadata created by init)
        :return: A new pandas dataframe with similarity metrics using cosine similarity based on the tfidf vectors
        """
        if newComparisonText is None:
            raise IOError("The new String is Empty")
        elif not isinstance(newComparisonText, str):
            raise IOError("The New Compariosn Text is not a String")
        elif type(dataframe) is not pd.core.frame.DataFrame:
            raise IOError("The passed object was not a dataframe")
        elif 'BagOfWords' not in dataframe.columns:
            raise IOError('The passed dataframe must have a column named TF-IDF.'
                          ' Make sure this is the dataframe returned from init')


        new_tokens = self._tokenizeText(newComparisonText)
        new_tokens_string = [" ".join(new_tokens)]
        tfidf_vectorizer_vectors = tfidf_vectorizer.transform(new_tokens_string) #tfidf_vectorizer is a global variavble from above
        new_vector = tfidf_vectorizer_vectors.toarray().tolist()[0]

        tuples = []
        # # The following 3 lines are Ephraim's simplified implementation to replace the for loop that follows. The sorting and onwards is not replaced here.
        # new_pat_vec = [new_vector for row in dataframe['BagOfWords']] # Create a 2d list where each line is the new_vector data, length matching our dataframe
        # new_comparison = cosine_similarity(dataframe['BagOfWords'].tolist(), new_pat_vec)
        # tuples = [[name,sim] for name,sim in zip(dataframe[id_col],new_comparison[0])]

        for pn, vec in zip(dataframe[self.id_col],dataframe['TF-IDF']):  # iterates through both of these columns at same time
            similarity = self.cosineSimilarity(new_vector, vec)  # compares new TF-IDF vector to the ones in dataframe
            tuples.append([pn, similarity])  # adds to the tuples, contains the patent number and similarity

        tuples = sorted(tuples, key=lambda similarity: similarity[1],reverse=True)  # sort the tuples based off of similarity
        df = pd.DataFrame(tuples,columns=[self.id_col,'Similarity'])  # turns the sorted tuple into a pandas dataframe

        return df


    def matches(self, compFrame, docFrame, threshold=.6):
        print("Matches: ")
        print("____________________")
        for r in range(0, len(compFrame)):
            for n in range(0, r + 1):
                entry = compFrame.iloc[n][r]
                if type(entry) is not str and entry < .99 and entry >= threshold:
                    print("val: " + str(compFrame.iloc[n][r]))
                    print(compFrame.columns[r])
                    print(compFrame.columns[n])
                    print("col: " + str(n))
                    print("row: " + str(r))
                    print(docFrame[self.txt_col][n])
                    print(docFrame[self.txt_col][r])
                    print()
        print("____________________")
