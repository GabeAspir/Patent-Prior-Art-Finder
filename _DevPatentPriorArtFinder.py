from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#Gabe
def init(csv, publicationNumberColumnString, comparisonColumnString):
	#Column Headers for dataframe:
		#PublicationNumber #Abstract
	#Dataframe will be created
	_tokenize(dataframe, comparisonColumnString)
	corpus = _createCorpus(dataframe)
	_bagOfWordize(dataframe, corpus)
	_TF_IDFize(dataframe, corpus)

	return newPandasDataFrame


#Private methods for init to call
#Gabe
def _tokenize(dataframe, comparisonColumnString):

	#Will add column to dataframe called 'Tokens'
#Gabe
def _tokenizeText(String):

#Gabe
def _createCorpus(dataframe):

	return corpus

#Zach
def _createNewCorpus(dataframe, newTokens):
	corpus = []

	for i in dataframe.index: 
		tokens = dataframe['Tokenized'][i]

		#Only adds the new words by converting the lists into sets (no doubles)
		#Then finding the new words by subtracting one set (a) from another set (b)
		#then adds back in the new words that were in (b) and not in (a), back into a
		token_set= set(tokens)
		corpus_set = set(corpus)
		new_tokens = token_set - corpus_set
		corpus = corpus + list(new_tokens)

	#doing the same thing with the new tokens
	corpus_set = set(corpus)
	token_set = set(newTokens) #set of the newTokens from the parameter
	new_tokens = token_set - corpus_set

	corpus = corpus + list(new_tokens)

	return corpus



#Ephraim
def _bagOfWordize(dataframe, Corpus): 

	#Will add column called 'BagOfWords'

#Zach
def _TFIDFize(dataframe, corpus):
	#adding column called 'TF-IDF'
	dataframe.insert(len(dataframe.columns), 'TF-IDF', '')

	#for each set of tokens, creates a vector of tf-idf values and adds it to the new column
	for i in data.index:
		tokens = dataframe['Tokens'][i]
		vector = vectorize_tf_idf(dataframe,tokens,corpus)
		dataframe['TF-IDF'][i] =vector

def _vectorize_tf_idf(data,tokens,corpus):
	v = []
	for word in corpus:
		#tf: number of times word appears in tokens for this abstract over the amounf of (tokenized) words in the patent
		tf = tokens.count(word)/len(tokens) 
		number_of_patents_with_word =appearences(data, word) 

		#idf: the log of the amount of documents divided by the number of patents with the word
		idf = math.log(float(len(data))/ number_of_patents_with_word) 
		v.append(tf*idf)
	return v

def _appearences(data, word):
	#gets the number of times a word appears in the tokens of all the data
	number = 0
	for tokens in data['Tokens']:
		if word in tokens:
			number +=1

	return number
	



#For the User
#Must Initialize first

#Zach
def jaccardTable(dataframe):


	return newTable


def jaccardSimilarity(patent1,patent2):
	return num

#Ephraim
def cosineSimilarity(patent1, patent2):

	return  num


def cosineTable(dataframe):

	return newTable

#Zach
#Comparing new patent based on TF-IDF/Cosine Similarity
#dataframe must have TF-IDF column
def compareNewPatent(newComparisonText, dataframe):
	#UpdateCorpus
	return newRanking #with two columns


