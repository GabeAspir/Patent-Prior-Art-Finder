from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import math
from os import path

class _DevPatentPriorArtFinder:

    def __init__(self):
        self.corpus = []
        self.number_of_patents_with_word = {}
        self.id_col = ""
        self.txt_col = ""

    # Gabe
    def init(self, csvPath, publicationNumberColumnString ='PublicationNumber', comparisonColumnString='Abstract'):
        """
        Prepares csv patents data for similarity functions by creating a pandas dataframe that can be passed to the desired function.

        - Renames publicationNumberColumnString, comparisonColumnString to 'PublicationNumber' and 'Abstract', respectivly
        - Adds a tokenized representation of the 'Abstract' columns, removing stop words, punctuation, and making all chars lowercase. Numbers are replaced with _NUM_
        - Adds a bag of words vector representing the wordcount (dense) of each document for every word in the corpus
        - Adds a vector representing the tfidf values for each document

        :param csv: Takes a csv file meant to represent a collection of patents
        :param publicationNumberColumnString: Optional, set by default to 'PublicationNumber'. The name of the column that contains the number of the patents, serves as an ID for that patent in output.
        :param comparisonColumnString: Optional, set by default to 'Abstract'. The name of the column that contains the patents' text, this is the data that is actually processed.
        :return: returns a pandas dataframe that adds relevant metadata to the patents. This metadata is later used for the similarity methods.
        """
        if csvPath is None:
            raise IOError('The passed file path was empty')
        #elif path.exists(csvPath) is False:
            #raise IOError('The passed file object does not exist')

        # Column Headers for dataframe:
        # PublicationNumber #Abstract
        # Dataframe will be created
        self.id_col = publicationNumberColumnString
        self.txt_col = comparisonColumnString
        dataframe = pd.read_csv(csvPath)

        # Testing that the necessary columns exist
        if self.txt_col not in dataframe.columns:
            raise IOError('the passed csv must have a column named Abstract or pass a column name as a parameter')
        if self.id_col not in dataframe.columns:
            raise IOError('the passed csv must have a column named PublicationNumber'
                          ' or pass a column name as a parameter')


        self._tokenize(dataframe)
        corpus = self._createCorpus(dataframe)
        self._bagOfWordize(dataframe, corpus)
        self._TFIDFize(dataframe, corpus)

        return dataframe

    # Private methods for init to call
    # Gabe
    def _tokenize(self,dataframe):
        dataframe['Tokens'] = dataframe[self.txt_col].apply(self._tokenizeText)

    # Will add column to dataframe called 'Tokens'
    # Gabe
    def _tokenizeText(self,string):

        def filterOut(word):
            remove_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                           'ourselves', 'you', "you're", "you've", "you'll", "you'd",
                           'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                           'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
                           "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                           'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                           "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                           'were', 'be', 'been', 'being', 'have', 'has', 'had',
                           'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                           'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                           'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                           'between', 'into', 'through', 'during', 'before', 'after', 'above',
                           'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                           'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                           'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                           'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                           'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                           'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
                           're', 've', 'y', 'ain', 'aren', "aren't",
                           'couldn', "couldn't", 'didn', "didn't", 'doesn',
                           "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                           'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                           "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                           'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                           'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

            if word in remove_list:
                return False
            else:
                return True

        lowercasedString = string.lower()
        # To split based on white space and random characters
        stringArray = re.split('\W+', lowercasedString)
        # Will substitute numbers for _NUM_
        stringArray = [re.sub(r"/^\d*\.?\d*$/", "_NUM_", s) for s in stringArray]
        # Will filter out 1 letter words like "I" and "a"
        stringArray = list(filter(lambda s: len(s) > 1, stringArray))
        stringArray = list(filter(filterOut, stringArray))
        # Will return a List/Array
        return stringArray


    # Zach
    #Need to know how many documents contain each word later on for  TF-IDF
    def _set_word_count(self, tokens):
        tokens_set = set(tokens)
        for word in tokens_set:
            if word in self.number_of_patents_with_word:
                self.number_of_patents_with_word[word]+=1
            else:
                self.number_of_patents_with_word[word]=1



    # Gabe
    def _createCorpus(self,dataframe):
        corpus = []

        for i in dataframe.index:
            tokens = dataframe['Tokens'][i]
            self._set_word_count(tokens)


            # Only adds the new words by converting the lists into sets (no doubles)
            # Then finding the new words by subtracting one set (a) from another set (b)
            # then adds back in the new words that were in (b) and not in (a), back into a
            token_set = set(tokens)
            corpus_set = set(corpus)
            new_tokens = token_set - corpus_set
            corpus = corpus + list(new_tokens)

        self.corpus = corpus
        return corpus




    # Ephraim
    # Will add column called 'BagOfWords' to dataframe
    def _bagOfWordize(self,dataframe, corpus):
        counts = []
        for row in dataframe['Tokens']:
            count = []  # Initialize count as an empty list
            for word in corpus:
                count.append(row.count(word))  # get the wordcount in each list of words, and record the count
            counts.append(
                count)  # Each list of wordCount vectors represents one document, and the counts variable is the list of all our docs' counts
        dataframe['BagOfWords'] = counts

    # Zach
    def _TFIDFize(self,dataframe, corpus):
        # adding column called 'TF-IDF'
        dataframe.insert(len(dataframe.columns), 'TF-IDF', '')

        # for each set of tokens, creates a vector of tf-idf values and adds it to the new column
        for i in dataframe.index:
            tokens = dataframe['Tokens'][i]
            vector = self._vectorize_tf_idf(dataframe, tokens, corpus)
            dataframe['TF-IDF'][i] = vector

    def _vectorize_tf_idf(self, data, tokens, corpus):
        v = []
        for word in corpus:
            # tf: number of times word appears in tokens for this abstract over the amounf of (tokenized) words in the patent
            tf = tokens.count(word) / len(tokens)
            appearences = self.number_of_patents_with_word.get(word)

             # idf: the log of the amount of documents divided by the number of patents with the word

            if tf !=0:
                idf = math.log(float(len(data)) / appearences)

            else: 
                idf = 0

            v.append(tf * idf)
        return v



    # def _appearences(self,data, word):
    #     # gets the number of times a word appears in the tokens of all the data
    #     number = 0
    #     for tokens in data['Tokens']:
    #         if word in tokens:
    #             number += 1

    #     return number

    # For the User
    # Must Initialize first

    # Zach
    def jaccardTable(self, dataframe):
        """ Takes a pandas dataframe with the metadata produced by init and returns a new table showing the jaccard index of each pair of patents"""

        if dataframe is None:
            raise IOError("The dataframe was empty")
        elif type(dataframe) is not pd.core.frame.DataFrame:
            raise IOError("The passed object was not a dataframe")
        elif 'BagOfWords' not in dataframe.columns:
            raise IOError('The passed dataframe must have a column named BagOfWords.'
                          ' Make sure this is the dataframe returned from init')
        for row in dataframe['BagOfWords']:
            if isinstance(row, list) is False:
                raise IOError('The contents of BagOfWords column were not all lists.')
            elif all(isinstance(entry,(int,float))for entry in row) is False:
                raise IOError('The contents of BagOfWords column were not all lists of numbers.')

        table = pd.DataFrame(dataframe[self.id_col])  # creating a new table for jaccard index data
        for bow, n in zip(dataframe['BagOfWords'], dataframe[
            self.id_col]):  # iterating through both data and name at same time to allow us to add the name to the row
            comps = []  # series that represents this bag of word's jaccard index with each bow's, will become a pandas series/column at the end
            for b in dataframe[
                'BagOfWords']:  # iterating over every other doc's bow vector (this actually results double for each pair)
                comps.append(self.jaccardSimilarity(bow,
                                                    b))  # applying jaccard similarity function (below) to the 2 BOWs, then adding it to the list
            table[n] = comps  # adding this new column, n is the publication number from above

        return table

    # accepts vector bag of words
    def jaccardSimilarity(self, patent1, patent2):
        """
        Returns the Jaccard Similarity index of 2 patents

        :param patent1: Vector representing the text of 1 patent's text
        :param patent2: Vector representing the text of a 2nd patent's text
        :return: The jaccard similarity index of those 2 patents
        """

        if patent1 is None or patent2 is None:
            raise IOError("One of or both of the Patents are empty")
        elif type(patent1) is not list:
            raise IOError("Patnet input must be a list")
        elif len(patent1) != len(patent2):
            raise IOError("Bag of Words must be the same length")

        count = 0

        # counting the number of total words combined between both of them
        for x in range(len(patent1)):
            if patent1[x] != 0 or patent2[x] != 0:  # not equaling 0 means that it occurs at least once
                count += 1
        numerator = 0

        # Counting the number of words in both
        for x in range(len(patent1)):
            if patent1[x] != 0 and patent2[x] != 0:
                numerator += 1

        return (numerator / count)

    # Zach
    def cosineSimilarity(self,patent1, patent2):
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
            raise IOError("Patnet input must be a list")
        elif len(patent1) != len(patent2):
            raise IOError("Bag of Words must be the same length")


        v1 = np.array(patent1).reshape(1, -1)
        v2 = np.array(patent2).reshape(1, -1)
        return cosine_similarity(v1, v2)[0][0]

    # Ephraim
    def cosineTable(self,dataframe):
        """
         Takes a dataframe and returns a table with the cosine simialrity metrics of each pair
        , uses scikit-learn's cosine function.

        :param dataframe: dataframe from init
        :return: pandas table with cosine similarity (SciKit implementation) index from BOW vectors
        """
        if dataframe is None:
            raise IOError("The dataframe was empty")
        elif type(dataframe) is not pd.core.frame.DataFrame:
            raise IOError("The passed object was not a dataframe")
        elif 'BagOfWords' not in dataframe.columns:
            raise IOError('The passed dataframe must have a column named BagOfWords.'
                          ' Make sure this is the dataframe returned from init')
        for row in dataframe['BagOfWords']:
            if isinstance(row,list) is False:
                raise IOError('The contents of BagOfWords column were not all lists.')
            elif all(isinstance(entry,(int,float))for entry in row) is False:
                raise IOError('The contents of BagOfWords column were not all lists of numbers.')

        newTable = pd.DataFrame(cosine_similarity(dataframe['BagOfWords'].tolist()))
        newTable.columns = dataframe[self.id_col]
        newTable.index = dataframe[self.id_col]

        return newTable

    # Zach
    # Comparing new patent based on TF-IDF/Cosine Similarity
    # dataframe must have TF-IDF column

    def compareNewPatent(self,newComparisonText, dataframe):
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

        for row in dataframe['BagOfWords']:
            if not all(isinstance(row,list)):
                raise IOError('The contents of BagOfWords column were not all lists.')
            elif all(isinstance(entry,(int,float))for entry in row) is False:
                raise IOError('The contents of BagOfWords column were not all lists of numbers.')


        new_tokens = self._tokenizeText(newComparisonText)
        new_corpus = self._createCorpus(dataframe)  # has to create new corpus
        new_vector = self._vectorize_tf_idf(dataframe, new_tokens,
                                       new_corpus)  # gets a vector with the tf-idf values of new text

        tuples = []

        # # The following 3 lines are Ephraim's simplified implementation to replace the for loop that follows. The sorting and onwards is not replaced here.
        # new_pat_vec = [new_vector for row in dataframe['BagOfWords']] # Create a 2d list where each line is the new_vector data, length matching our dataframe
        # new_comparison = cosine_similarity(dataframe['BagOfWords'].tolist(), new_pat_vec)
        # tuples = [[name,sim] for name,sim in zip(dataframe[id_col],new_comparison[0])]

        for pn, vec in zip(dataframe[self.id_col],
                           dataframe['TF-IDF']):  # iterates through both of these columns at same time
            similarity = self.cosineSimilarity(new_vector, vec)  # compares new TF-IDF vector to the ones in dataframe
            tuples.append([pn, similarity])  # adds to the tuples, contains the patent number and similarity

        tuples = sorted(tuples, key=lambda similarity: similarity[1],
                        reverse=True)  # sort the tuples based off of similarity
        df = pd.DataFrame(tuples,
                          columns=[self.id_col,
                                   'Similarity'])  # turns the sorted tuple into a pandas dataframe

        return df
