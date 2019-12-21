import random
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import re


class MarkovChain:
    def __init__(self, corpus, article_summary, words_to_use):
        self.corpus = corpus
        self.article_summary = article_summary
        self.words_to_use = words_to_use
        self.word_dic = defaultdict(list)
        self.create_dictonary()

    def create_dictonary(self):
        stopWords = set(stopwords.words('english'))
        words = self.corpus.split(' ')
        for current_word, next_word in zip(words[0:-1], words[1:]):
            self.word_dic[current_word].append(next_word)
        self.word_dic = dict(self.word_dic)

    def randomText(self, word):
        if word:
            if " " in word[0][0]:
                word = word[0][0].split(" ")[0]
            else:
                word = word[0][0]
        summary = re.split('[?.!]', self.article_summary)
        biggest = 0
        index = 0
        count = 0;
        for m in summary:
            in_to_use = 0
            for x in m.split(" "):
                if x in self.words_to_use:
                    in_to_use += 1
            if in_to_use > biggest:
                biggest = in_to_use
                index = count
            count += 1

        print("Can you believe "+word+" said \"" + summary[index].strip()+"\"")
