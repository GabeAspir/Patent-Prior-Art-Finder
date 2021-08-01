from _DevFilesPatentPriorArtFinder import _DevFilesPatentPriorArtFinder as paf
import pandas as pd
import os

def main():
    #path= r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\testSet3"
    zpath = r'C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\sampleZipSet'
    metapath = r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\MetaDataDrive"
    patpath = r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\Identical4280631.json"
    twopath= r"C:\Users\mocka\PycharmProjects\Patent-Prior-Art-Finder\Patent Queries\Data Science Fixed Abstracts"
    #zpath = r'C:\Users\zacha\OneDrive\Documents\Computer Science\Patent-Prior-Art-Finder\Patent Queries\sampleZipSet'
    myPaf = paf(zpath, publicationNumberColumnString='publication_number', comparisonColumnString="abstract_en", cit_col= "Citations")
    newPat= pd.io.json.read_json(patpath, orient='records')
    #myPaf.train()
    out= (myPaf.compareNewPatent(newPatentSeries=newPat.iloc[0], dirPath=zpath, threshold=1))
    print(out)
    print("///////////////////////////////////////////////////////////////////////////////////////////")





def sep(entry):
    head, tail = os.path.split(entry.path)
    print( head + "\emb\\" + tail)
if __name__ == "__main__":
    main()