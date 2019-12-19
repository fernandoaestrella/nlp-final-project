import os

from bs4 import BeautifulSoup

filename_pos = "joy_clean.txt"
f = open(filename_pos, "r")
contents_pos = f.read().splitlines( )

filename_neg = "sad_clean.txt"
f = open(filename_neg, "r")
contents_neg = f.read().splitlines( )

# filename="all\APW_ENG_20030424.0698.sgm"

missed_docs = 0
tagged_texts = []

for filename in contents_pos:
    filename = os.path.join(r"all", filename) + ".sgm"
    if os.path.isfile(filename):
        f = open(filename, "r")
        contents = f.read()
        soup = BeautifulSoup(contents, features="lxml")
        post = soup.find('doc')
        tagged_texts.append((post.findChild('text').text, "pos"))
    else:
        print(filename)
        missed_docs += 1

for filename in contents_neg:
    filename = os.path.join(r"all", filename) + ".sgm"
    if os.path.isfile(filename):
        f = open(filename, "r")
        contents = f.read()
        soup = BeautifulSoup(contents, features="lxml")
        post = soup.find('doc')
        tagged_texts.append((post.findChild('text').text, "neg"))
    else:
        print(filename)
        missed_docs += 1

print(missed_docs)
print(tagged_texts)
print(tagged_texts.__len__())