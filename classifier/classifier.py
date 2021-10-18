import numpy as np
import os
import csv

class Trainer():
    def __init__(self, wordSize = 20):
        super(Trainer, self).__init__()
        self.wordSize = wordSize
        self.w = 0
        self.b = 0
        self.punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

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

    def trainNb(self, X, Y):
        """Train a binary NB classifier."""
        # + 1 for the Laplacian smoothing
        pos_p = X[Y == 1, :].sum(0) + 1
        pos_p = pos_p / pos_p.sum()
        neg_p = X[Y == 0, :].sum(0) + 1
        neg_p = neg_p / neg_p.sum()
        w = np.log(pos_p) - np.log(neg_p)
        # Estimate P(0) and P(1) and compute b
        b = 0
        return w, b


    def inferenceNb(self, X, w, b):
        """Prediction of a binary NB classifier."""
        logits = X @ w + b
        return (logits > 0).astype(int)

    def train(self, dataDirectory, bowFile):
        filename = dataDirectory + "w.txt"
        if os.path.exists(filename):
            self.w = np.loadtxt(filename)
            return
        # The script loads the data and tra
        # in a classifier.
        data = np.loadtxt(bowFile)
        X = data[:, :-1]
        Y = data[:, -1]
        self.w, self.b = self.trainNb(X, Y)
        with open(filename, "w", encoding="utf-8") as f:
            for value in self.w:
                print(value, file=f)
        f.close()

    def check(self, voc, positiveDirectory, negativeDirectory):
        documents = []
        labels = []
        for f in os.listdir(positiveDirectory):
            documents.append(self.readDocument(positiveDirectory + f, voc))
            labels.append(1)
        for f in os.listdir(negativeDirectory):
            documents.append(self.readDocument(negativeDirectory + f, voc))
            labels.append(0)
        X = np.stack(documents)
        Y = np.array(labels)

        predictions = self.inferenceNb(X, self.w, self.b)
        accuracy = (predictions == Y).mean()

        # This part detects the most relevant words for the classifier.
        indices = self.w.argsort()
        positiveWords = []
        negativeWords = []
        words = list(voc.keys())
        for i in indices[:20]:
            negativeWords.append(words[i] + " " + str(self.w[i]))

        for i in indices[-20:]:
            positiveWords.append(words[i] + " " + str(self.w[i]))

        return accuracy, positiveWords, negativeWords

    def savePredict(self, voc, rawDirectory, storeDirectory):
        documents = []
        for f in os.listdir(rawDirectory):
            documents.append(self.readDocument(rawDirectory + f, voc))
        X = np.stack(documents)
        predictions = self.inferenceNb(X, self.w, self.b)

        filename = storeDirectory + "predictions.csv"    
        with open(filename, "w", encoding="utf-8") as csvfile: 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(["ID", "Sentiment", "Content"])
            i = 0
            for f in os.listdir(rawDirectory): 
                filename = os.path.splitext(f)
                tweet = open(rawDirectory + f, encoding="utf8")
                text = tweet.read()
                tweet.close()
                if predictions[i]:
                    csvwriter.writerow([filename[0], "positive", text])
                else:
                    csvwriter.writerow([filename[0], "negative", text])
                i += 1

