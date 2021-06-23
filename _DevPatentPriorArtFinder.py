from sklearn.metrics.pairwise import cosine_similarity
import pandas

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

# Ephraim
# Will add column called 'BagOfWords' to dataframe
def _bagOfWordize(dataframe, corpus):
	counts = []
	for row in dataframe['Tokens']:
		count = [] # Initialize count as an empty list
		for word in corpus:
			count.append(row.count(word)) #get the wordcount in each list of words, and record the count
		counts.append(count) # Each list of wordCount vectors represents one document, and the counts variable is the list of all our docs' counts
	dataframe['BagOfWords'] = counts

#Zach
def _TFIDFize(dataframe, Corpus):

	#column called 'TF-IDF'



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
	newTable = pandas.DataFrame(cosine_similarity(dataframe['BagOfWords']))
	newTable.columns = dataframe['Publication_Number']
	newTable.index = dataframe['Publication_Number']
	return newTable

#Zach
#Comparing new patent based on TF-IDF/Cosine Similarity
#dataframe must have TF-IDF column
def compareNewPatent(newComparisonText, dataframe):
	#UpdateCorpus
	return newRanking #with two columns


