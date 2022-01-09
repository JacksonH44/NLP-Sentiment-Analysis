from nltk.classify.scikitlearn import SklearnClassifier
import nltk
import random
from nltk.classify import naivebayes
import pickle
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

# Read in Twitter reviews
short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

# Add in the Twitter reviews as tuples to documents
documents = []
all_words = []

# We only allow adjectives
allowed_word_types = ["J"]

for r in short_pos.split("\n"):
    documents.append((r, "pos"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)  # Tag only adjectives
    for w in pos:
        if w[1][0] in allowed_word_types:  # Take the first definition of the word
            all_words.append(w[0].lower())

for r in short_neg.split("\n"):
    documents.append((r, "neg"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)  # Tag only adjectives
    for w in pos:
        if w[1][0] in allowed_word_types:  # Take the first definition of the word
            all_words.append(w[0].lower())

random.shuffle(documents)

save_documents = open("pickle/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

# Create a frequency distribution map for the words in all_words, with keys being the words and values being the frequencies
all_words = nltk.FreqDist(all_words)

# Use the top 5000 words to train the model
word_features = list(all_words.keys())[:5000]

save_word_features = open("pickle/word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# A function that determines whether the most popular words in the data appear in the document


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)  # Create a boolean list

    return features


# Create a dictionary that contains the word mapped to a boolean contingent on membership in the document
feature_sets = [(find_features(rev), category)
                for (rev, category) in documents]

training_set = feature_sets[:10000]
testing_set = feature_sets[10000:]

# Randomize data
random.shuffle(feature_sets)

save_feature_sets = open("pickle/feature_sets.pickle", "wb")
pickle.dump(feature_sets, save_feature_sets)
save_feature_sets.close()

# # positive data example


# Naive Bayes algorithm
# posterior = prior occurrences x liklihood / evidence

# Base training model
classifier = nltk.NaiveBayesClassifier.train(training_set)


# Testing model
print("Original Naive Bayes algo accuracy:",
      (nltk.classify.accuracy(classifier, testing_set)) * 100)

# Save the classifier with pickle

save_classifier = open("pickle/original_naive_bayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# Multinomial Naive Bayes algorithm
mnb = SklearnClassifier(MultinomialNB())
mnb.train(training_set)

print("Multinomial Naive Bayes algo accuracy:",
      (nltk.classify.accuracy(mnb, testing_set)) * 100)

save_mnb = open("pickle/mnb.pickle", "wb")
pickle.dump(mnb, save_mnb)
save_mnb.close()

# Bernoulli
bnb = SklearnClassifier(BernoulliNB())
bnb.train(training_set)

print("Bernoulli Naive Bayes algo accuracy:",
      (nltk.classify.accuracy(bnb, testing_set)) * 100)

save_bnb = open("pickle/bnb.pickle", "wb")
pickle.dump(bnb, save_bnb)
save_bnb.close()

# Logistic Regression
log_reg = SklearnClassifier(LogisticRegression(max_iter=200))
log_reg.train(training_set)

print("Logistic Regression algo accuracy:",
      (nltk.classify.accuracy(log_reg, testing_set)) * 100)

save_log_reg = open("pickle/log_reg.pickle", "wb")
pickle.dump(log_reg, save_log_reg)
save_log_reg.close()

# Stochastic Gradient Descent
sgd = SklearnClassifier(SGDClassifier())
sgd.train(training_set)

print("Stochastic Gradient Descent algo accuracy:",
      (nltk.classify.accuracy(sgd, testing_set)) * 100)

save_sgd = open("pickle/sgd.pickle", "wb")
pickle.dump(sgd, save_sgd)
save_sgd.close()

# Linear Support Vector Machine
l_svc = SklearnClassifier(LinearSVC())
l_svc.train(training_set)

print("Linear Support Vector Machine algo accuracy:",
      (nltk.classify.accuracy(l_svc, testing_set)) * 100)

save_l_svc = open("pickle/l_svc.pickle", "wb")
pickle.dump(l_svc, save_l_svc)
save_l_svc.close()
