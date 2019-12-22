import nltk
import pickle
import spacy
import FacebookPosts
from CommentGenerator import CommentGenerator
from PostGenerator import PostGenerator
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
        with open("rc_all_bayes", "rb") as f:
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


if __name__ == '__main__':
    """ part below written by Vitor Mouzinho"""
    fp = FacebookPosts.Facebook_post_generator()
    article_content = fp.create_news()
    train = ModleAndTrainer()
    sentiment = train.main(article_content.orginal_text)
    extract_sentiment = SentimentExtractor(article_content.orginal_text,sentiment)
    summary_of_article = article_content.generate_summerzization()
    pg = PostGenerator(article_content.orginal_text,summary_of_article,extract_sentiment.return_sentiment_words())
    print("\nArticle summary: \n"+summary_of_article+"\n")
    pg.randomText(nltk.FreqDist(train.list_of_people).most_common(1))
    comments = CommentGenerator(article_content.orginal_text)
    print("\nComment for post: \n"+comments.generate_comment(sentiment))






