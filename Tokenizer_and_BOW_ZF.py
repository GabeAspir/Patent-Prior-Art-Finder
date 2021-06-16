import pandas as pd
import re
import numpy as np


def token_and_bag(data, column):
 	tokenized_dataframe =tokenize(data, column)
 	vocab = create_vocab(data,column)
 	final_data =bow(tokenized_dataframe, column, vocab)
 	return final_data


#Input a data frame and specify a column that want to tokenize
#It will add the tokenized data in a new column on the right
def tokenize(data, column):



	#insert another column for tokenized words
	data.insert(len(data.columns), 'Tokenized', '')
	

	#iterate through each abstract
	for i in data.index:
		ab = data[column][i] #column is the argument passed above

		print(ab)
		print()
	
		ab =ab.lower() #to lower case
		ab = re.sub('\d+', ' _num_ ', ab) #removes number
		ab = re.sub(r'[^\w\s]', ' ', ab) #removes punctuation
		ab = re.sub(r'\b\w{1,2}\b', '', ab) #removes single and two charachter words


		#remove stopwords
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


		for word in remove_list:
			ab = ab.replace(" "+ word + " ", ' ')



		tokens = ab.split()
		print(tokens)
		print()


		data['Tokenized'][i] = tokens
	
	return data


#takes the data fram and specified column and creates the vocab for bag of words
def create_vocab(data, column):
	vocab = []

	for i in data.index:
		tokens = data['Tokenized'][i]

		token_set= set(tokens)
		vocab_set = set(vocab)
		new_tokens = token_set - vocab_set
		vocab = vocab + list(new_tokens)

	print("here is the vocab")
	print(vocab)


	return vocab






#takes the dataframe, the column, and the vocab as arguments
#will add the bag of words for the specified column in a new column at the end
def bow(data,column, vocab):
	
	data.insert(len(data.columns), 'Bag of Words', '')

	for i in data.index:
		tokens = data['Tokenized'][i]
		vector = vectorize(tokens,vocab)
		data['Bag of Words'][i] =vector

	return data





def vectorize(tokens,vocab):
	vector = []
	for word in vocab:
		vector.append(tokens.count(word))

	return vector





	
patents = pd.read_csv(r'C:\Users\zacha\OneDrive\Documents\Computer Science\YU CS 2021\TenPatents.csv')
new_data =token_and_bag(patents, 'Abstract')
#new_data.to_csv('TokenizedCSV.csv')
