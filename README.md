## Facebook Post Generator

December 2019

LING131A- Natural Language Processing in Python Final Project

Authors: Fernando Estrella, Erica Granor, Vitor Mouzinho

The aim of this project is to generate short summaries of news articles for the purpose of posting to Facebook. This project produces a text summary of the article, a customized comment about the post (using sentiment analysis), and a computer-generated comment in response to the post.

The NLP methods deployed are:
- Sentiment Analysis
- Naive Bayes Classifier/Machine Learning
- Entity Extraction/Characterization

## To use:

1. Install NLTK, SpaCy, Pickle, NewsApi, and BeautifulSoup
2. Open the `all_together.py` file
3. Run the main method
4. When asked for a `'topic'`, input any keyword (>= 3 characters) you would like to see the summary of current articles for.

## The output will be:

`SUMMARY OF THE ARTICLE`- a 3-sentence objective summary of the first article retrieved with the given keyword. Also includes the URL of the news article.

`DESCRIPTIVE COMMENT`- provides a one sentence, customized comment that the poster of the article might write about the article.

`SAMPLE RESPONSE COMMENT`- provides a comment that a respondent might post on Facebook in response to the article

## Files in this Repository:

`ArticleSummariztion.py` – Methods for summarizing the article

`all_together.py` -- Returns all elements- summary, first and second comments

`bc_all_bayes` – The initial model used to identify the sentiment of the article

`CommentGenerator.py` – Generates the "Sample Response Comment"

`Extract_text.py` – Reference code for creating the model for determining sentiment

`FacebookPosts.py` – Retrieves article data from the NewsApi client

`README.md`

NLP Final Report.pdf – Final write-up

`PostGenerator.py` -- Generates post summary from article

Proposal.rtf- copy of project proposal

`rc_all_bayes` -- actual model used for sentiment analysis

`SentimentExtractor.py` – Retrieves words in the article that match the sentiment of the article
