
from collections import defaultdict
from nltk.corpus import stopwords
import re


class PostGenerator:
    def __init__(self, corpus, article_summary, words_to_use):
        self.corpus = corpus
        self.article_summary = article_summary
        self.words_to_use = words_to_use

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

        print("Facebook post generated: Can you believe "+word+" said \"" + summary[index].strip()+"\"")
