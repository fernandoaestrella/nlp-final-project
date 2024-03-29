# Erica Granor
# LING 131a
# Assignment 5
# Due 10/10/19
from collections import Counter

import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import sentiwordnet as swn
import random
import pickle

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from newsapi import NewsApiClient
import spacy

# Returns a list containing all the movie reviews
# Format is a list of tuples, each containing a list of words in the review, and the label
# the return of this is what you pass in for the documents argument in the methods below the line
import FacebookPosts
from Article_Summarization import ArticleSummarization
from MarkovChain import MarkovChain
#from Part_Of_Speech import PosTagger

from SentimentExtractor import SentimentExtractor


class ModleAndTrainer:
    def __init__(self):
        self.list_of_people =[]
        self.score = 0
        self.verbs=[]
        self.nouns=[]
    def return_all_reviews(self):
        reviews = [(list(movie_reviews.words(fileid)), category)
                   for category in movie_reviews.categories()
                   for fileid in movie_reviews.fileids(category)]
        return reviews

    # returns ordered list of positive or negative label associated with each review
    # takes list of tuples, each containing a list of words and a label
    def assign_labels(self, documents):
        labels = []
        for doc, label in documents:
            labels.append(label)
        return labels

    # Returns frequency distribution of all words in the movie reviews
    # for raw counts
    def return_fd_all_words_in_reviews(self, documents):
        dist = nltk.FreqDist()
        for words, _ in documents:
            dist.update(words)
        return dist

    # returns the words in the freq dist as a list
    # e.g. use all_words from above
    def fd_to_list(self, word_set):
        word_features = list(word_set)
        return word_features

    # 2000 most common words in the reviews
    # the return of this is what to pass in for "listofwords" in the methods below
    def return_top_2000_words(self, word_freqdist):
        word_features_top_2000 = []
        for word in word_freqdist.most_common(2000):
            word_features_top_2000.append(word[0])
        return word_features_top_2000

    # ----------------

    # 1 RAW COUNTS
    # for the naive bayes model use return_multinomial_nb_classifier()

    # Returns count of how many times each count appears in document
    def document_features_rc(self, document, listofwords):
        features = Counter(document)
        features_list = []
        for word in listofwords:
            features_list.append(features[word])
        # and this is what we feed into the classifier
        return features_list

    # 2. BINARY COUNTS
    # For the Naive Bayes, use return_binary_nb_classifier()

    # Returns feature vector for document for all words as binary features
    def document_features_bc(self, document, listofwords):
        document_words = set(document)
        features_list = []
        for word in listofwords:
            features_list.append(word in document_words)
        return features_list

    # 3 SENTIWORDNET

    # Does not include part of speech
    # Create a set of all the words in all the reviews
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

    # 4. MPQA SUBJECTIVITY LEXICON

    # ------------

    # Train classifier, run predictions
    # to mix up the positive and negative reviews
    def shuffle(self, documents):
        documents_shuffled = random.shuffle(documents)
        return documents_shuffled

    # Split data- create training set and labels and test set and labels
    # takes set of documents (list containing tuples of the text and positive or negative label),
    # and the index at which you want to split
    # returns tuple with training set, training labels, test set, and test labels
    # TODO: This should shuffle and split, but not assign labels
    def split_data(self, list_of_documents, splitindex):
        random.shuffle(list_of_documents)
        # create training set and labels
        training = list_of_documents[:splitindex]
        # training_labels = assign_labels(training)
        # create test set and labels
        test = list_of_documents[splitindex:]
        # test_labels = assign_labels(test)
        return [training, test]

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

    # Create Binary Naive Bayes Classifier
    # For use in 1 & 3
    def return_bernoulli_nb_classifier(self):
        model_bernoulli = BernoulliNB()
        return model_bernoulli

    # Create Multinomial Naive Bayes Classifier
    # For use in 2
    def return_multinomial_nb_classifier(self):
        model_multinomialnb = MultinomialNB()
        return model_multinomialnb

    # Create Tree Classifier
    def return_tree_Classifier(self):
        model_tree = tree.DecisionTreeClassifier()
        return model_tree

    # trains a model given a model, training set of list of documents, list of labels that correspond
    def train(self, model, trainingset, labelslist):
        model.fit(trainingset, labelslist)

    # This is how you predict given a model and set of features
    def predict(self, model, feature_set):
        print(model.predict(feature_set))

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

    # returns accuracy given a document and corresponding list of labels
    def return_accuracy(self, model, features, labels):
        model.score(features, labels)

    def main(self, content_url_description_list):
        # Here we would train all the classifiers and save them via pickle.
        # In lieu of argparse, this is an attempt at taking user input.
        random.seed(1)

        feature = input(
            "Choose a model: \n1 - all words raw counts\n2 - all words binary\n3 - SentiWordNet words raw counts\n4 - SentiWordNet words binary\nType a number: ")
        if feature not in ('1', '2', '3', '4'):
            print("Invalid input")
            exit(1)

        classifier = input("Choose 'bayes' or 'tree': ")
        if classifier not in ('bayes', 'tree'):
            print("Invalid input")
            exit(1)

        mode = input("Choose 'train' or 'run': ")
        if mode not in ('train', 'run'):
            print("Invalid input")
            exit(1)

        if mode == 'run':
            pass
            # filename = input("Enter a filename: ")

        if feature == '1':
            feature_type = "rc"
            wordlist_type = "all"
        elif feature == '2':
            feature_type = "bc"
            wordlist_type = "all"
        elif feature == '3':
            feature_type = "rc"
            wordlist_type = "swn"
        elif feature == '4':
            feature_type = "bc"
            wordlist_type = "swn"
        else:
            wordlist_type = None
            feature_type = None
            exit(1)

        pickle_name = feature_type + "_" + wordlist_type + "_" + classifier

        if mode == 'train':
            # Creates list of all reviews
            all_reviews = self.return_all_reviews()
            training_and_test_sets = self.split_data(all_reviews, 1000)
            training_set = training_and_test_sets[0]
            test_set = training_and_test_sets[1]

            if wordlist_type == "all":
                word_list = self.all_words(training_set)
            elif wordlist_type == "swn":
                word_list = self.filter_by_pos_and_neg(self.all_words(training_set))

            if feature_type == "rc":
                training_features = self.create_training_feature_vectors(training_set, word_list, 'rc')
                test_features = self.create_test_feature_vectors(test_set, word_list, 'rc')
            elif feature_type == "bc":
                training_features = self.create_training_feature_vectors(training_set, word_list, 'bc')
                test_features = self.create_test_feature_vectors(test_set, word_list, 'bc')
            else:
                exit(1)

        if mode == 'run':
            with open(pickle_name, "rb") as f:
                (classifier, word_list) = pickle.load(f)
                # need to check here which model to run
            # with open(filename, "r") as f:
            #     document = f.read().split()
            document = content_url_description_list  # fix condition where this is None

            # Load English tokenizer, tagger, parser, NER and word vectors
            nlp = spacy.load("en_core_web_sm")

            # Process whole documents
            # text = ("When Sebastian Thrun started working on self-driving cars at "
            #         "Google in 2007, few people outside of the company took him "
            #         "seriously. “I can tell you very senior CEOs of major American "
            #         "car companies would shake my hand and turn away because I wasn’t "
            #         "worth talking to,” said Thrun, in an interview with Recode earlier "
            #         "this week.")
            doc = nlp(document)

            # Analyze syntax
            self.nouns= [chunk.text for chunk in doc.noun_chunks]
            self.verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

            # Find named entities, phrases and concepts
            for entity in doc.ents:
                if entity.label_ is "PERSON":
                    self.list_of_people.append(entity.text)

                # here might need to convert the document into the correct format
            if feature_type == 'rc':
                prediction = classifier.predict([self.document_features_rc(document, word_list)])
                print(prediction)
                # turn the document into a list of strings
                # do document_feature() on the correct type
                # do document_feature () on the list of strings and the word list
            elif feature_type == 'bc':
                prediction = classifier.predict([self.document_features_bc(document, word_list)])
                print(prediction)
            print(classifier.predict_proba([self.document_features_bc(document, word_list)]))
            se = SentimentExtractor(document, prediction[0])
            matching_sentiment_wordlist = se.return_sentiment_words()
            print(matching_sentiment_wordlist)
            # prints list as set
            print(set(matching_sentiment_wordlist))

        if mode == 'train':
            if classifier == 'bayes':
                if feature == '1' or feature == '3':
                    model_naive_bayes = self.return_multinomial_nb_classifier()
                elif feature == '2' or feature == '4':
                    model_naive_bayes = self.return_bernoulli_nb_classifier()
                self.train(model_naive_bayes, training_features, self.assign_labels(training_set))
                with open(pickle_name, "wb") as f:
                    pickle.dump((model_naive_bayes, word_list), f)
                # Print accuracy
                print(model_naive_bayes.score(test_features, self.assign_labels(test_set)))

            elif classifier == 'tree':
                model_tree = self.return_tree_Classifier()
                self.train(model_tree, training_features, self.assign_labels(training_set))
                # Do these all need to have different file names?
                with open(pickle_name, "wb") as f:
                    pickle.dump((model_tree, word_list), f)
                print(model_tree.score(test_features, self.assign_labels(test_set)))


# First train the model, so enter 2, bayes, train (this is the most accurate). It'll take about a minute
# Then do 2, bayes, run, and then enter the file name of a text file in the same directory
# The article passed in should be a text file
# The positive or negative prediction should be printed out
# At this time, only one document at a time is predicted upon. We could easily write a method that batches them, or
# modify the current methods.
# Let's see if this works on news articles.
if __name__ == '__main__':
    # content_url_description_list is a list of articles and each element is in a tuple goes like content url description
    # content_and_url[(content,url,description)]

    content_url_description_list = FacebookPosts.create_news()
    print(content_url_description_list)
    article_summarization = ArticleSummarization(content_url_description_list[0][1],content_url_description_list[0][0])
    mc = MarkovChain(article_summarization.orginal_text,article_summarization)
    mc.create_dictonary()
    t = ModleAndTrainer()
    t.main(article_summarization.orginal_text)
    if t.list_of_people:
        mc.randomText(nltk.FreqDist(t.list_of_people).most_common(1))
    else:
        mc.randomText(None)


    """summary = article_summarization.generate_summerzization()
    trainer = ModleAndTrainer()
    trainer.main(article_summarization.orginal_text)
    list_of_people = trainer.list_of_people
    pos = PosTagger(summary,list_of_people,trainer.verbs)"""
