import nltk
import pickle
import spacy
import FacebookPosts
from Article_Summarization import ArticleSummarization
from MarkovChain import MarkovChain


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

        print(classifier.predict([self.document_features_bc(document, word_list)])) # this is what you need to inpu somewhere else


if __name__ == '__main__':
    # content_url_description_list is a list of articles and each element is ina tuple goes like
    # content url description
    content_url_description_list = FacebookPosts.create_news()
    article_summarization = ArticleSummarization(content_url_description_list[0][1],
                                                 content_url_description_list[0][0])
    mc = MarkovChain(article_summarization.orginal_text, article_summarization)
    mc.create_dictonary()
    t = ModleAndTrainer()
    t.main(article_summarization.orginal_text)
    generated_comment = ""
    if t.list_of_people:
        generated_comment = mc.randomText(nltk.FreqDist(t.list_of_people).most_common(1))
    else:
        generated_comment = mc.randomText(None)
    print("SUMMARY OF THE ARTICLE\n" + article_summarization.generate_summerzization() + "\n")
    print("GENERATED COMMENT\n" + generated_comment)
