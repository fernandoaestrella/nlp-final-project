from nltk.corpus import sentiwordnet as swn

def extract_sentiment(text, sentiment):
    print(list(swn.senti_synsets("long"))[0].pos_score())


text = "things"
extract_sentiment(text, "pos")