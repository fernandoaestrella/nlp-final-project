import nltk
import pickle
import spacy
import FacebookPosts
from Article_Summarization import ArticleSummarization
from MarkovChain import MarkovChain
from SentimentExtractor import SentimentExtractor


class ModleAndTrainer:
    def __init__(self):
        self.list_of_people =[]
        self.score = 0
        self.verbs = []
        self.nouns = []

    # Returns feature vector for document for all words as binary features
    def document_features_bc(self, document, listofwords):
        document_words = set(document)
        features_list = []
        for word in listofwords:
            features_list.append(word in document_words)
        return features_list

    def main(self, content_url_description_list):
        with open("bc_all_bayes", "rb") as f:
            (classifier, word_list) = pickle.load(f)
        document = content_url_description_list

        # Load English tokenizer, tagger, parser, NER and word vectors
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(document)

        # Find named entities, phrases and concepts
        for entity in doc.ents:
            if entity.label_ is "PERSON":
                self.list_of_people.append(entity.text)

        return classifier.predict([self.document_features_bc(document, word_list)]) # this is what you need to inpu somewhere else


# First train the model, so enter 2, bayes, train (this is the most accurate). It'll take about a minute
# Then do 2, bayes, run, and then enter the file name of a text file in the same directory
# The article passed in should be a text file
# The positive or negative prediction should be printed out
# At this time, only one document at a time is predicted upon. We could easily write a method that batches them, or
# modify the current methods.
# Let's see if this works on news articles.
if __name__ == '__main__':
    fp = FacebookPosts.Facebook_post_generator()
    article_content = fp.create_news()
    train = ModleAndTrainer()
    sentiment = train.main(article_content.orginal_text)
    print(sentiment)
    extract_sentiment = SentimentExtractor(article_content.orginal_text,sentiment)
    mkc = MarkovChain(article_content.orginal_text,article_content.generate_summerzization(),extract_sentiment.return_sentiment_words())
    mkc.randomText(nltk.FreqDist(train.list_of_people).most_common(1))



