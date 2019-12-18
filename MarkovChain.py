import random
from collections import defaultdict

import nltk


class MarkovChain:
    def __init__(self,corpus):
        self.corpus = corpus
        self.word_dic = defaultdict(list)

    def create_dictonary(self):
        words = self.corpus.split(' ')
        for current_word, next_word in zip(words[0:-1],words[1:]):
            self.word_dic[current_word].append(next_word)
        self.word_dic = dict(self.word_dic)

    ## word - put in a name that you want the sentence to be about
    def randomText(self,word):
        sen=word[0][0]
        word1 = word[0][0]
        while "." not in sen:
            word2 = random.choice(self.word_dic[word1])
            word1 = word2
            sen+=" "+word2

        print(sen)


