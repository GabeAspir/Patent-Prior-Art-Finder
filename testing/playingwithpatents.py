from _DevFilesPatentPriorArtFinder import _DevFilesPatentPriorArtFinder as paf
import pandas as pd

def main():
    path= r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\testSet2"
    myPaf = paf(r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\testSet2", publicationNumberColumnString='Publication_Number', comparisonColumnString="Abstract", cit_col= "Citations")
    newPat= pd.io.json.read_json(r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\similarPat.json", orient='records')
    myPaf.compareNewPatent(newPatentSeries=newPat.iloc[0], dirPath=path, threshold=.5)


if __name__ == "__main__":
    main()