import sys
import os
import io
import numpy as np
import matplotlib.pyplot as plt

from classifier.classifier import Trainer
from classifier.features import Extractor
from classifier.vocabulary import Vocabulary

from reader.manager.component import APIConfig, APIManager
from reader.transformer.component import text_only

dataDirectory = "data/"
vocabularyName ="vocabulary.txt"
trainPosDirectory = "data/train/pos/"
trainNegDirectory = "data/train/neg/"
vocabulary = Vocabulary(dataDirectory, vocabularyName, 1000)
vocabulary.createFromStopList(trainPosDirectory, trainNegDirectory, dataDirectory + "stopwords.txt")
print("Vocabulary created")

extractor = Extractor(dataDirectory)
trainBowName = "train.txt.gz"
extractor.saveBow(trainBowName, vocabulary.voc, trainPosDirectory, trainNegDirectory)
print()
print("Bow saved")

trainer = Trainer()
trainer.train(dataDirectory, dataDirectory + trainBowName)
print()
print("Train completed")

""" testPosDirectory = "data/Trump/pos/"
testNegDirectory = "data/Trump/neg/"
accuracy, positiveWords, negativeWords = trainer.check(vocabulary.voc, testPosDirectory, testNegDirectory)
print()
print("Accuracy:", accuracy * 100)
print("NEGATIVE WORDS")
for row in negativeWords:
    word, grade = row.split()
    print ("{:<20} {:<20}".format(word, grade))
print()
print("POSITIVE WORDS")
for row in positiveWords:
    word, grade = row.split()
    print ("{:<20} {:<20}".format(word, grade)) """

config = APIConfig()
tweeterManager = APIManager(config)

print("Insert query string:")
queryString = input()
print("Saving data...")
rawDirectory = 'data/' + queryString + '/raw/'
storeDirectory = 'data/' + queryString + '/'
if not os.path.exists(rawDirectory):
    os.makedirs(rawDirectory)

tweets = tweeterManager.query(queryString + ' lang:en', pages=10)
cnt = 0
for tweet in tweets:
    filename = rawDirectory + '/' + tweet.id + '.txt'
    if not os.path.exists(filename):        
        file = io.open(filename, "w", encoding="utf-8")
        file.write(text_only(tweet.text))
        file.close()
        cnt += 1
print()
print(cnt, " files saved")

trainer.savePredict(vocabulary.voc, rawDirectory, storeDirectory)
print()
print("Predictions made")