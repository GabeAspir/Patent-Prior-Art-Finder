from _DevFilesPatentPriorArtFinder import _DevFilesPatentPriorArtFinder as paf
import pandas as pd

def main():

    myPaf = paf(r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\testSet", publicationNumberColumnString='publication_number', comparisonColumnString="abstract_en", cit_col= "Citations")



if __name__ == "__main__":
    main()