import heapq

import nltk

from FacebookPosts import create_news


class ArticleSummarization:
    def __init__(self,article_content):
        self.sentence_scores = {}
        self.sentence_tokens = nltk.sent_tokenize(article_content)
        self.word_frequencies = {}
        self.orginal_text = article_content
        self.article_content = nltk.re.sub('[^a-zA-Z]', ' ', article_content)
        self.article_content  = nltk.re.sub(r'\s+', ' ', self.article_content)

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

    def generate_summerzization(self):
        self.frequency_dis()
        self.frequency_caculate()
        self.sentence_freq()
        summary_sentences = heapq.nlargest(2, self.sentence_scores, key=self.sentence_scores.get)
        summary = ' '.join(summary_sentences)
        print(summary)
        return summary



if __name__ == '__main__':
        content_url_description_list = create_news()
        n = ArticleSummarization(content_url_description_list[0][0])
        n.generate_summerzization()