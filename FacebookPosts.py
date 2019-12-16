def create_news():
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
            content_and_url = []
            for c_a in top_headlines.get("articles"):
                content_and_url.append((c_a.get("content"), c_a.get("url"), c_a.get("description")))
            if len(content_and_url) < 1:
                print("choose a different topic, topic in top headlines is not available. ")
            else:
                return content_and_url

# <<<<<<< HEAD
# https://github.com/fernandoaestrella/nlp-final-project/
content_and_url = create_news()
print(content_and_url)
# =======

# content_url_description_list is a list of articles and each element is in a tuple goes like content url description
# content_and_url[(content,url,description)]
# content_url_description_list = create_news()
# >>>>>>> 2f3df3b250f151b99e0d61a4f5857106cc631156