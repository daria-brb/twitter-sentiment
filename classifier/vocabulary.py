import collections
import os

class Vocabulary:
    def __init__(self, dir, name, size):
        self.dir = dir
        self.name = name
        self.size = size
        self.punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.voc = {}

    def loadFromFile(self, filename):
        """Load the vocabulary and returns it.

        The return value is a dictionary mapping words to numerical
        indices.

        """
        f = open(filename, encoding="utf8")
        n = 0
        voc = {}
        for w in f.read().split():
            voc[w] = n
            n += 1
        f.close()
        return voc

    def readWordListStop(self, filename, stopWordsFile):
        """Read the file and returns a list of words excluding stop words."""
        f = open(filename, encoding="utf8")
        text = f.read()
        f.close()
        f = open(stopWordsFile, encoding="utf8")
        stopList = f.read().split()
        f.close()
        words = []
        # The three following lines replace punctuation symbols with
        # spaces.
        table = str.maketrans(self.punct, " " * len(self.punct))
        text = text.translate(table)
        for w in text.split():
            w = w.lower()
            if len(w) > 2 and w not in stopList:
                words.append(w)
        return words

    def write(self, voc, filename, n):
        """Write the n most frequent words to a file."""
        i = 0
        with open(filename, "w", encoding="utf-8") as f:
            for word, count in sorted(voc.most_common(n)):
                print(word, file=f)
                self.voc[word] = i
                i = i + 1
        f.close()

    def createFromStopList(self, positiveDirectory, negativeDirectory, stopWordsFile):
        if os.path.exists(self.dir + self.name):
            existedVoc = self.loadFromFile(self.dir + self.name)
            if len(existedVoc) == self.size:
                self.voc = existedVoc
                return

        words = collections.Counter()
        for f in os.listdir(positiveDirectory):
            words.update(self.readWordListStop(positiveDirectory + f, stopWordsFile))
        for f in os.listdir(negativeDirectory):
            words.update(self.readWordListStop(negativeDirectory + f, stopWordsFile))
            
        self.write(words, self.dir + self.name, self.size)