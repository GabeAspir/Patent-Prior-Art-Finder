from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import gzip
import json

def main():
    path= r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\testSet3"
    zpath = r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\sampleZipSet"
    bigpath= r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\Data Science Fixed Abstracts"
    for file in os.scandir(zpath):
        if file.is_file() is True:
            print("Raw file "+str(file))
            print(pd.read_json(file, compression="gzip"))
            print("-------------------")
            print("Unzipped "+str(gzip.GzipFile(file, 'r')))
            #print(pd.read_json(zpath))
            print("=======================")
if __name__ == "__main__":
    main()