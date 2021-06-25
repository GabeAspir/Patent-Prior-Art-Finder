from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import math

#Gabe
def init(csv, publicationNumberColumnString, comparisonColumnString):
	#Column Headers for dataframe:
		#PublicationNumber #Abstract
	#Dataframe will be created

	dataframe = pd.read_csv(csv)
	dataframe = dataframe.rename(columns={'PublicationNumber': publicationNumberColumnString})
	dataframe = dataframe.rename(columns={'Abstract': comparisonColumnString})

	_tokenize(dataframe)
	corpus = _createCorpus(dataframe)
	_bagOfWordize(dataframe, corpus)
	_TFIDFize(dataframe, corpus)

	return dataframe


#Private methods for init to call
#Gabe
def _tokenize(dataframe):

	dataframe['Tokens'] = dataframe['Abstract'].apply(_tokenizeText)
	#Will add column to dataframe called 'Tokens'
#Gabe
def _tokenizeText(string):

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

	lowercasedString = string.lower()
    #To split based on white space and random charactars
	stringArray = re.split('\W+', lowercasedString)
    #Will substitute numbers for _NUM_
	stringArray[:]= [re.sub(r"[0-9]+","_NUM_",s) for s in stringArray]
    #Will filter out 1 letter words like "I" and "a"
	stringArray = list(filter (lambda s: len (s) > 1, stringArray))

	for word in stringArray:
		if word in remove_list:stringArray.remove(word)


    #Will return a List/Array
	return stringArray




#Gabe
def _createCorpus(dataframe):

	corpus = []

	for i in dataframe.index:
		tokens = dataframe['Tokens'][i]

		#Only adds the new words by converting the lists into sets (no doubles)
		#Then finding the new words by subtracting one set (a) from another set (b)
		#then adds back in the new words that were in (b) and not in (a), back into a
		token_set= set(tokens)
		corpus_set = set(corpus)
		new_tokens = token_set - corpus_set
		corpus = corpus + list(new_tokens)

	return corpus

#Zach
def _createNewCorpus(dataframe, newTokens):
	corpus = []

	for i in dataframe.index:
		tokens = dataframe['Tokens'][i]

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
def _TFIDFize(dataframe, corpus):
	#adding column called 'TF-IDF'
	dataframe.insert(len(dataframe.columns), 'TF-IDF', '')

	#for each set of tokens, creates a vector of tf-idf values and adds it to the new column
	for i in dataframe.index:
		tokens = dataframe['Tokens'][i]
		vector = _vectorize_tf_idf(dataframe,tokens,corpus)
		dataframe['TF-IDF'][i] =vector

def _vectorize_tf_idf(data,tokens,corpus):
	v = []
	for word in corpus:
		#tf: number of times word appears in tokens for this abstract over the amounf of (tokenized) words in the patent
		tf = tokens.count(word)/len(tokens)
		number_of_patents_with_word =_appearences(data, word)

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

# Zach
def jaccardTable(dataframe):
	table = pd.DataFrame(dataframe['PublicationNumber'])  # creating a new table
	for bow, n in zip(dataframe['Bag of Words'], dataframe['PublicationNumber']):  # iterating through both at same time
		number = n  # getting the publication number so can use it as header later on
		comps = []  # series that represents this bag of word's cosine comp with all bow's
		for b in dataframe['Bag of Words']:  # getting the other bag of words
			comps.append(jaccardSimilarity(bow, b))  # applying jaccard similarity to the 2 BOWs
		table[n] = comps  # adding this new column, n is the publication number from above

	return table


# accepts vector bag of words
def jaccardSimilarity(patent1, patent2):
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
def cosineSimilarity(patent1, patent2):
	v1 = np.array(patent1).reshape(1, -1)
	v2 = np.array(patent2).reshape(1, -1)
	return cosine_similarity(v1, v2)[0][0]



def cosineTable(dataframe):
	newTable = pd.DataFrame(cosine_similarity(dataframe['BagOfWords']))
	newTable.columns = dataframe['Publication_Number']
	newTable.index = dataframe['Publication_Number']

	return newTable

#Zach
#Comparing new patent based on TF-IDF/Cosine Similarity
#dataframe must have TF-IDF column
def compareNewPatent(newComparisonText, dataframe):
	text = _tokenizeText(newComparisonText)
	new_tokens = _tokenizeText(newComparisonText)
	new_corpus = _createCorpus(dataframe, new_tokens) # has to create new corpus
	new_vector = _vectorize_tf_idf(dataframe, new_tokens, new_corpus) #gets a vector with the tf-idf values of new text

	tuples = []

	for pn,vec in zip(dataframe['PublicationNumber'], dataframe['TF-IDF']): #iterates through both of these columns at same time
		similarity = cosineSimilarity(new_vector,vec) #compares new TF-IDF vector to the ones in dataframe
		tuples.append([pn,similarity]) #adds to the tuples, contains the patent number and similarity
	tuples = sorted(tuples, key=lambda similarity: similarity[1], reverse = True)  #sort the tuples based off of similarity
	df = pd.DataFrame(tuples, columns = ['Publication Number', 'Similarity']) #turns the sorted tuple into a pandas dataframe


	return df





