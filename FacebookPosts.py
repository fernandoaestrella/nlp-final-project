from newsapi import NewsApiClient


def create_news():
    run = True;
    while run:
        topic = input("type a an article topic:")
        if len(topic) > 2:
            newsapi = NewsApiClient(api_key='e43b9508c4954478a8e5a1ea0f6f000c')
            top_headlines = newsapi.get_top_headlines(q=topic, language='en')
            content_and_url =[]
            for m in top_headlines.get("articles"):
                if len(top_headlines.get("articles")) > 0:
                    content_and_url.append(m.get("content"))
                    content_and_url.append(m.get("url"))
                    return content_and_url
        print("choose a different topic, topic in top headlines is not available. ")

# https://github.com/fernandoaestrella/nlp-final-project/
content_and_url = create_news()
