import os
import pickle
from random import shuffle

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import sentiwordnet as swn
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree


class LabeledTextExtractor:

    def split_data(self, list_of_documents, splitindex):
        shuffle(list_of_documents)
        # create training set and labels
        training = list_of_documents[:splitindex]
        # training_labels = assign_labels(training)
        # create test set and labels
        test = list_of_documents[splitindex:]
        # test_labels = assign_labels(test)
        return [training, test]

    # returns if word has positive or negative score >= 0.5 using sentiwordnet
    def is_pos_or_neg(self, word):
        # gives list of synsets for word
        synsets = list(swn.senti_synsets(word))
        if len(synsets) == 0:
            return False
        first_synset = synsets[0]
        positive_score = first_synset.pos_score()
        negative_score = first_synset.neg_score()
        if positive_score >= 0.5 or negative_score >= 0.5:
            return True
        else:
            return False

    # returns set of all words in the reviews that have positive or negative score of >=0.5
    def return_pos_or_neg_words(self, review_words):
        pos_or_neg_words_all = set()
        for word in review_words:
            if self.is_pos_or_neg(word):
                pos_or_neg_words_all.add(word)
        return pos_or_neg_words_all

    # returns list; feed this into the document_features method for raw counts
    def return_word_list_pos_neg(self, pos_neg_all_word_set):
        return list(pos_neg_all_word_set)

    # returns word list after filtering through swn words with score of >= 0.5 for pos or neg
    # For sentiwordnet #3
    # use the return of this as the wordlist
    # USE THIS FUNCTION, the rest are helpers
    def filter_by_pos_and_neg(self, wordlist):
        pos_or_neg = self.return_pos_or_neg_words(wordlist)
        pos_or_neg_list = self.return_word_list_pos_neg(pos_or_neg)
        return pos_or_neg_list

    def all_words(self, documents):
        """
        Create a set of all the words in all the reviews
        :param documents: List of (wordlist, label) tuples
        :return: Set of all the words in all the review
        """
        all_words_in_reviews = set()
        for document, label in documents:
            for word in document:
                all_words_in_reviews.add(word)
        return all_words_in_reviews

    # Gets a string containing filenames in each line and a given sentiment for those files.
    # Creates a tuple with the contents of each file. The first element of each
    # tuple is the content of a file and the second is the corresponding sentiment. Appends each
    # tuple to a list
    def extract(self, contents, sentiment):
        for filename in contents:
            filename = os.path.join(r"all", filename) + ".sgm"
            if os.path.isfile(filename):
                f = open(filename, "r")
                contents = f.read()
                soup = BeautifulSoup(contents, features="lxml")
                post = soup.find('doc')
                self.tagged_texts.append((nltk.word_tokenize(post.findChild('text').text), sentiment))

    # Returns feature vector for document for all words as binary features
    def document_features_bc(self, document, listofwords):
        document_words = set(document)
        features_list = []
        for word in listofwords:
            features_list.append(word in document_words)
        return features_list

    # Returns count of how many times each count appears in document
    def document_features_rc(self, document, listofwords):
        features = nltk.Counter(document)
        features_list = []
        for word in listofwords:
            features_list.append(features[word])
        # and this is what we feed into the classifier
        return features_list

    # just the reviews
    # Create training feature vector
    # for model takes 'bc', 'rc'
    def create_training_feature_vectors(self, training_set, wordlist, model):
        training_features = []
        if model == 'bc':
            for document in training_set:
                training_features.append(self.document_features_bc(document[0], wordlist))
        elif model == 'rc':
            for document in training_set:
                training_features.append(self.document_features_rc(document[0], wordlist))
        return training_features

    # Creates set of test features given test set and list of words
    def create_test_feature_vectors(self, testset, wordlist, model):
        test_features = []
        if model == 'bc':
            for document in testset:
                test_features.append(self.document_features_bc(document[0], wordlist))
        elif model == 'rc':
            for document in testset:
                test_features.append(self.document_features_rc(document[0], wordlist))
        return test_features

    # returns ordered list of positive or negative label associated with each review
    # takes list of tuples, each containing a list of words and a label
    def assign_labels(self, documents):
        labels = []
        for doc, label in documents:
            labels.append(label)
        return labels

    # Create Multinomial Naive Bayes Classifier
    # For use in 2
    def return_multinomial_nb_classifier(self):
        model_multinomialnb = MultinomialNB()
        return model_multinomialnb

    # Create Tree Classifier
    def return_tree_Classifier(self):
        model_tree = tree.DecisionTreeClassifier()
        return model_tree

    def __init__(self):
        filename_pos = "joy_clean.txt"
        f = open(filename_pos, "r")
        contents_pos = f.read().splitlines( )

        filename_neg = "sad_clean.txt"
        f = open(filename_neg, "r")
        contents_neg = f.read().splitlines( )

        self.tagged_texts = []

        self.extract(contents_pos, "pos")
        self.extract(contents_neg, "neg")

        # training_and_test_sets = self.split_data(self.tagged_texts, 10)
        # training_set = training_and_test_sets[0]
        # test_set = training_and_test_sets[1]
        #
        # word_list_types = ["all", "swn"]
        # feature_types = ["rc", "bc"]
        # classifiers = ["bayes", "tree"]
        # for wordlist_type in word_list_types:
        #     for feature_type in feature_types:
        #         if wordlist_type == "all":
        #             word_list = self.all_words(training_set)
        #             print(word_list)
        #         elif wordlist_type == "swn":
        #             word_list = self.filter_by_pos_and_neg(self.all_words(training_set))
        #
        #         if feature_type == "rc":
        #             training_features = self.create_training_feature_vectors(training_set, word_list, 'rc')
        #             test_features = self.create_test_feature_vectors(test_set, word_list, 'rc')
        #         elif feature_type == "bc":
        #             training_features = self.create_training_feature_vectors(training_set, word_list, 'bc')
        #             test_features = self.create_test_feature_vectors(test_set, word_list, 'bc')
        #
        #         for classifier_name in classifiers:
        #             pickle_name = feature_type + "_" + wordlist_type + "_" + classifier_name
        #             with open(pickle_name, "rb") as f:
        #                 (classifier, word_list) = pickle.load(f)
        #             # print(classifier)
        #             # if classifier_name == 'bayes':
        #             #     if feature_type == "rc":
        #             #         model_naive_bayes = self.return_multinomial_nb_classifier()
        #             #     elif feature_type == 'bc':
        #             #         model_naive_bayes = self.return_bernoulli_nb_classifier()
        #             #     # Print accuracy
        #             print(test_features[9].__len__())
        #             print(self.assign_labels(test_set).__len__())
        #             print(classifier.score(test_features, self.assign_labels(test_set)))
        #
        #             # elif classifier_name == 'tree':
        #             #     # model_tree = self.return_tree_Classifier()
        #             #     print(classifier.score(test_features, self.assign_labels(test_set)))

# LabeledTextExtractor()