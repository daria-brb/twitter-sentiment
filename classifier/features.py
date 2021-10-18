import numpy as np
import os

class Extractor():

    def __init__(self, dir, punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"):
        self.punct = punct
        self.dir = dir

    def readDocument(self, filename, voc):
        """Read a document and return its BoW representation."""
        f = open(filename, encoding="utf8")
        text = f.read()
        f.close()
        table = str.maketrans(self.punct, " " * len(self.punct))
        text = text.translate(table)
        # Start with all zeros
        bow = np.zeros(len(voc))
        for w in text.split():
            # If the word is the vocabulary...
            if w in voc:
                # ...increment the proper counter.
                index = voc[w]
                bow[index] += 1
        return bow

    def saveBow(self, name, voc, positiveDirectory, negativeDirectory):
        if os.path.exists(self.dir + name):
            return
        # The script compute the BoW representation of all documents.
        documents = []
        labels = []
        for f in os.listdir(positiveDirectory):
            documents.append(self.readDocument(positiveDirectory + f, voc))
            labels.append(1)
        for f in os.listdir(negativeDirectory):
            documents.append(self.readDocument(negativeDirectory + f, voc))
            labels.append(0)
        # np.stack transforms the list of vectors into a 2D array.
        X = np.stack(documents)
        Y = np.array(labels)
        # The following line append the labels Y as additional column of the
        # array of features so that it can be passed to np.savetxt.
        data = np.concatenate([X, Y[:, None]], 1)
        np.savetxt(self.dir + name, data)