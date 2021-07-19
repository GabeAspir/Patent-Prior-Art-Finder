from _DevFilesPatentPriorArtFinder import _DevFilesPatentPriorArtFinder as paf
import pandas as pd
import os

def main():
    path= r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\testSet3"
    myPaf = paf(path, publicationNumberColumnString='publication_number', comparisonColumnString="abstract_en", cit_col= "Citations")
    newPat= pd.io.json.read_json(r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\similarPat.json", orient='records')
    # myPaf.train()
    out= (myPaf.compareNewPatent(newPatentSeries=newPat.iloc[0], dirPath=path, threshold=.5))
    print("///////////////////////////////////////////////////////////////////////////////////////////")
    print(out)
    # print( path )
    # # for entry in os.scandir(path):
    # #     sep(entry)
def sep(entry):
    head, tail = os.path.split(entry.path)
    print( head + "\emb\\" + tail)
if __name__ == "__main__":
    main()