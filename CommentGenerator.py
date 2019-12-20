# Utilizes sentiment and entities to generate a short comment response to a given news article
import spacy


# Takes document as a string
class CommentGenerator:
    def __init__(self, document):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(document)
        # Analyze syntax
        self.list_of_orgs = []
        for entity in doc.ents:
            if entity.label_ is "ORG":
                self.list_of_orgs.append(entity.text)

        # Find named entities, phrases and concepts
        self.list_of_people = []
        for entity in doc.ents:
            if entity.label_ is "PERSON":
                self.list_of_people.append(entity.text)

        self.list_of_GPE = []
        for entity in doc.ents:
            if entity.label_ is "GPE":
                self.list_of_GPE.append(entity.text)

    # Takes self
    # Returns comment generated
    # takes sentiment as "pos" or "neg" after it is returned from the sentiment analysis
    def generate_comment(self, sentiment):
        if len(self.list_of_people) > 0:
            subject = self.list_of_people[0]
        elif len(self.list_of_orgs) > 0:
            subject = self.list_of_orgs[0]
        elif len(self.list_of_GPE) > 0:
            subject = self.list_of_GPE[0]
        else:
            subject = 'none'

        if subject != 'none':
            if sentiment == 'pos':
                comment = "Check out this great article about " + subject + "!"
            elif sentiment == 'neg':
                comment = "This is a terrible article about " + subject + "!"
        else:
            if sentiment == 'pos':
                comment = "Check out this great article."
            elif sentiment == 'neg':
                comment = "This is a terrible article."
            else:
                comment = "Check out this article."
        return comment


if __name__ == '__main__':
    c = CommentGenerator("This one's for the birds.")
    print(c.generate_comment("pos"))
    c = CommentGenerator("This is an article about Apple.")
    print(c.generate_comment("pos"))
    c = CommentGenerator("This is an article about Doug from Apple.")
    print(c.generate_comment("pos"))
    c = CommentGenerator("This is an article about the U.K. and Doug from Apple.")
    print(c.generate_comment("pos"))
    c = CommentGenerator("This is an article about the U.K.")
    print(c.generate_comment("neg"))
