import random
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

# Creating a class VoteClassifier that inherits from NLTK's ClassifierI


class VoteClassifier(ClassifierI):

    # Constructor
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    # Iterate through the list of classifier objects and classify based on the features
    # The classification is being treated as a vote
    # After done we return the mode(votes) which returns the most popular vote
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    # A similar function to classify, except we return a confidence score, which is how
    # many of the classifiers "agree" with the overall decision
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = (choice_votes / len(votes)) * 100
        return conf


new_docs = open("pickle/documents.pickle", "rb")
documents = pickle.load(new_docs)
new_docs.close()

new_features = open("pickle/word_features.pickle", "rb")
word_features = pickle.load(new_features)
new_features.close()

# A function that determines whether the most popular words in the data appear in the document


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)  # Create a boolean list

    return features


new_feature_sets = open("pickle/feature_sets.pickle", "rb")
feature_sets = pickle.load(new_feature_sets)
new_feature_sets.close()

random.shuffle(feature_sets)

training_set = feature_sets[:3600]
testing_set = feature_sets[3600:4000]

open_file = open("pickle/original_naive_bayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickle/mnb.pickle", "rb")
mnb = pickle.load(open_file)
open_file.close()

open_file = open("pickle/bnb.pickle", "rb")
bnb = pickle.load(open_file)
open_file.close()

open_file = open("pickle/log_reg.pickle", "rb")
log_reg = pickle.load(open_file)
open_file.close()

open_file = open("pickle/sgd.pickle", "rb")
sgd = pickle.load(open_file)
open_file.close()

open_file = open("pickle/l_svc.pickle", "rb")
l_svc = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier, mnb, bnb, log_reg, sgd, l_svc)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


def classify_out(text):
    tmp, tmp1 = sentiment(text)
    if tmp1 >= 80:
        return tmp
    else:
        return "Hmm...I'm not sure"
