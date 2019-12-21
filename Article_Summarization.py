import heapq
import urllib
from webbrowser import Mozilla

import nltk
import bs4 as bs
from requests import Request
from urllib.request import Request, urlopen
from FacebookPosts import create_news


class ArticleSummarization:
    # the parameter is the article url please pass that in
    def __init__(self,url,backup):
        self.backup = backup
        self.orginal_text = self.read_article(url)
        # print(self.orginal_text+" hi")
        self.sentence_scores = {}
        self.sentence_tokens = nltk.sent_tokenize(self.orginal_text)
        self.word_frequencies = {}

        self.article_content = nltk.re.sub('[^a-zA-Z]', ' ', self.orginal_text)
        self.article_content  = nltk.re.sub(r'\s+', ' ', self.article_content)

    def read_article(self,url):
        req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
        article = urlopen(req).read()
        parsed_article = bs.BeautifulSoup(article, 'lxml')
        paragraphs = parsed_article.find_all('p')
        article_text = ""
        for p in paragraphs:
            article_text += p.text

        if article_text:
            return article_text
        else:
            return self.backup

    def frequency_dis(self):
        stopwords = nltk.corpus.stopwords.words('english')
        word_frequencies = {}
        for word in nltk.word_tokenize(self.article_content):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        self.word_frequencies = word_frequencies

    def frequency_caculate(self):
        # print(self.orginal_text+"hi")
        maximum_frequncy = max(self.word_frequencies.values())
        for word in self.word_frequencies.keys():
            self.word_frequencies[word] = (self.word_frequencies[word] / maximum_frequncy)

    def sentence_freq(self):
        for sent in self.sentence_tokens:
            for word in nltk.word_tokenize(sent.lower()):
                if word in self.word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in self.sentence_scores.keys():
                            self.sentence_scores[sent] = self.word_frequencies[word]
                        else:
                            self.sentence_scores[sent] += self.word_frequencies[word]
    # this will generate a new summary of the article with three sentence if you want a shorter or longer summary just switch the 3 to w/e #
    # u want
    def generate_summerzization(self):
        self.frequency_dis()
        self.frequency_caculate()
        self.sentence_freq()
        summary_sentences = heapq.nlargest(3, self.sentence_scores, key=self.sentence_scores.get)
        summary = '\n'.join(summary_sentences)
        # print(summary)
        return summary
    # this will get you the actual content of the page
    def get_article_web_content(self):
        return self.orginal_text

if __name__ == '__main__':
    content_url_description_list = create_news()
    article = ArticleSummarization(content_url_description_list[0][1]);
    article.generate_summerzization()