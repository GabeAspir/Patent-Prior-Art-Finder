import pandas as pd
import re



#Input a data fram and specify a column that want to tokenize
#It will add the tokenized data in a new column on the right
def tokenize(data, column):

	#insert another column for tokenized words
	patents.insert(len(data.columns), 'Tokenized', '')


	#iterate through each abstract
	for i in data.index:
		ab = data[column][i] #column is the argument listed above

		print(ab)
	
		ab =ab.lower() #to lower case
		ab = re.sub('\d+', ' _num_ ', ab) #removes number
		ab = re.sub(r'[^\w\s]', ' ', ab) #removes punctuation
		ab = re.sub(r'\b\w{1,2}\b', '', ab) #removes single and two charachter words

		remove_list = ['the', 'and', 'for', 'with', 'via'] #more words to remove
		for word in remove_list:
			ab = ab.replace(word, ' ')



		tokens = ab.split()
		print(tokens)


		data['Tokenized'][i] = tokens



		
	
	return data	

	
patents = pd.read_csv(r'C:\Users\zacha\OneDrive\Documents\Computer Science\YU CS 2021\TenPatents.csv')
new_data =tokenize(patents, 'Abstract')
#new_data.to_csv('TokenizedCSV.csv')

