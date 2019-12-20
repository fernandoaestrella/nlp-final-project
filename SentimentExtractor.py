from nltk.corpus import sentiwordnet as swn
class SentimentExtractor:
    def __init__(self, document, sentiment):
        self.document = document
        self.sentiment = sentiment

    def return_sentiment_words(self):
        matching_sentiment_words = []
        words_in_doc = self.document.split()
        if self.sentiment == 'pos':
            for word in words_in_doc:
                if is_pos(word):
                    matching_sentiment_words.append(word)
        elif self.sentiment == 'neg':
            for word in words_in_doc:
                if is_neg(word):
                    matching_sentiment_words.append(word)
        return matching_sentiment_words


# returns True if word is positive (pos if pos score >=0.5 in SentiWordNet)
def is_pos(word):
    # gives list of synsets for word
    synsets = list(swn.senti_synsets(word))
    # checks to make sure synsets exist
    if len(synsets) == 0:
        return False
    first_synset = synsets[0]
    positive_score = first_synset.pos_score()
    if positive_score > 0:
        return True
    else:
        return False


# Returns true if word is negative
def is_neg(word):
    synsets = list(swn.senti_synsets(word))
    # checks to make sure synsets exist
    if len(synsets) == 0:
        return False
    first_synset = synsets[0]
    negative_score = first_synset.neg_score()
    if negative_score > 0:
        return True
    else:
        return False