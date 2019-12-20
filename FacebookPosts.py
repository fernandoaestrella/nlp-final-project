from newsapi import NewsApiClient

from Article_Summarization import ArticleSummarization


class Facebook_post_generator:

    def __init__(self):
        pass

    def create_news(self):
        run = True;
        while run:
            topic = input("type a an article topic:")
            if len(topic) > 2:
                newsapi = NewsApiClient(api_key='e43b9508c4954478a8e5a1ea0f6f000c')
                top_headlines = newsapi.get_top_headlines(q=topic, language='en')
                content_and_url = []
                for c_a in top_headlines.get("articles"):
                    c_a.get
                    content_and_url.append((c_a.get("content"), c_a.get("url"), c_a.get("description")))
                if len(content_and_url) < 1:
                    print("choose a different topic, topic in top headlines is not available. ")
                else:
                    return self.choose_article(content_and_url)

    def choose_article(self, content_and_url):
        article_content = []
        for article in content_and_url:
            if article[1]:
                m = ArticleSummarization(article[1], article[2])
                if len(m.orginal_text) > 200:
                    article_content.append(m)
                    break
        if article_content.__sizeof__() > 0:
            longest_article = article_content[0]
        else:
            return content_and_url[0][2]

        for content in article_content:
            if len(content.orginal_text) > len(longest_article.orginal_text):
                longest_article = content
        return longest_article



