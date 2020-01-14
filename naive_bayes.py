# Amy Wu
# Naive Bayes Classifier and Evaluation on the v2.0 polarity dataset
import math
import os
import numpy as np
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class NaiveBayes():

    def __init__(self):
        self.class_dict = {0: 'neg', 1: 'pos'}
        self.feature_dict = {}
        self.prior = np.empty(len(self.class_dict))
        self.likelihood = None
        self.V = set()  # store set of features
        self.total_docs = 0     # store total docs of all classes

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''
    def train(self, train_set):
        self.bigdoc = defaultdict(list)     # dictionary storing all tokens of each class
        self.Doc_count = defaultdict(int)   # dictionary storing document count of each class

        # iterate over training documents
        i = 0
        for root, dirs, files in os.walk(train_set):
            if len(files) != 0:
                self.Doc_count[i] = len(files)      # store total number of documents per class
                self.total_docs += len(files)       # store total num of docs
                for x, name in enumerate(files):
                    with open(os.path.join(root, name)) as f:
                        # collect class counts and feature counts
                        tokens = f.read().split()       # tokenize each doc text
                        self.bigdoc[i].extend(tokens)   # store all tokens into respective class
                        features = nb.select_features(tokens)   # select features from tokens
                        self.V.update(features)     # store unique set of features
                i += 1

        # store unique features into feature_dict to translate numpy indices to corresponding tokens
        for x, word in enumerate(self.V):
            self.feature_dict[x] = word

        self.likelihood = np.zeros(shape=(len(self.class_dict), len(self.feature_dict)))
        # print(self.V)

        # loop through pos and neg class
        for x in range(0, len(self.class_dict)):
            self.prior[x] = math.log10(self.Doc_count[x]/self.total_docs)       # get log priors of each class
            count_list = Counter(self.bigdoc[x])        # create dict of word sizes of all words in each class
            for y, word in zip(range(0, len(self.feature_dict)), self.feature_dict.values()):
                # normalize counts to probabilities, and take logs
                self.likelihood[x][y] = math.log10((count_list[word] + 1)/(len(self.bigdoc[x]) + 1))

        # print(self.prior)
        # print(self.feature_dict)
        # print(self.likelihood)

    '''
    Selects features that are negative or positive words and stopwords from each document. Returns list of said features
    '''
    def select_features(self, token_sent):
        filter_words = set(stopwords.words('english'))

        # nltk Vader sentiment analyzer that gives a score for words on intensity of positivity or negativity
        analyzer = SentimentIntensityAnalyzer()

        feats = []

        for word in token_sent:
            if (analyzer.polarity_scores(word)['compound']) >= 0.1:     # word is positive
                feats.append(word)
            elif (analyzer.polarity_scores(word)['compound']) <= -0.1:      # word is negative
                feats.append(word)
            if word in filter_words:        # include stopwords
                feats.append(word)

        return feats


    '''
     Tests the classifier on a development or test set.
     Returns a dictionary of filenames mapped to their correct and predicted
     classes such that:
     results[filename]['correct'] = correct class
     results[filename]['predicted'] = predicted class
     '''
    def test(self, dev_set):
        results = defaultdict(lambda: defaultdict(str))
        # iterate over testing documents
        i = 0
        for root, dirs, files in os.walk(dev_set):
            print(root)
            if len(files) != 0:
                for name in files:
                    with open(os.path.join(root, name)) as f:
                        # create feature vectors for each document
                        words = Counter(f.read().split())       # create counter dict with counts of each doc token
                        feat_vect = np.zeros(len(self.feature_dict))
                        # loop through feat dict and check if doc tokens are inside it
                        for x, word in zip(range(0, len(self.feature_dict)), self.feature_dict.values()):
                            if word in words:
                                feat_vect[x] = words[word]  # get count of words in doc and add to vector
                        # duplicate row of feature vector to get same dimension as likelihood
                        feat_vect = np.tile(feat_vect, (len(self.class_dict), 1))
                        # muliply likelihood with feature vector
                        feat_vect = np.multiply(feat_vect, self.likelihood)
                        # sum all logs of feature vector x likelihood products
                        feat_vect = np.sum(feat_vect, axis=1)
                        # add prior to feature vector
                        feat_vect = feat_vect+self.prior

                        arg_max = self.class_dict[int(np.argmax(feat_vect))]    # get arg max class
                        results[name]['correct'] = self.class_dict[i]
                        # print("correct = ", results[name]['correct'])
                        results[name]['predicted'] = arg_max
                        # print("predicted = ", results[name]['predicted'])
                i += 1

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # confusion matrix
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))

        # fill in confusion matrix
        for key in results.keys():
            if results[key]['correct'] == 'pos':
                if results[key]['predicted'] == 'pos':  # true positive
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[1][0] += 1     # false negative
            elif results[key]['correct'] == 'neg':
                if results[key]['predicted'] == 'neg':  # true negative
                    confusion_matrix[1][1] += 1
                else:
                    confusion_matrix[0][1] += 1     # false positive


        # get total number of docs per class
        pos_size = confusion_matrix[0][0] + confusion_matrix[1][0]
        neg_size = confusion_matrix[0][1] + confusion_matrix[1][1]

        print("Confusion Matrix")
        print(confusion_matrix)

        # compute scores
        pos_precision = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
        print('pos_precision = ', pos_precision)
        neg_precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
        print('neg_precision = ', neg_precision)
        pos_recall = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
        print('pos_recall = ', pos_recall)
        neg_recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
        print('neg_recall = ', neg_recall)
        pos_f1 = 2*(pos_precision*pos_recall/(pos_precision+pos_recall))
        print('pos_f1 = ', pos_f1)
        neg_f1 = 2*(neg_precision*neg_recall/(neg_precision+neg_recall))
        print('neg_f1 = ', neg_f1)

        accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1])/(neg_size+pos_size)
        print('accuracy = ', accuracy)

if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train('movie_reviews/train')     # train on the training set
    results = nb.test('movie_reviews/dev')      # test on dev set
    nb.evaluate(results)
