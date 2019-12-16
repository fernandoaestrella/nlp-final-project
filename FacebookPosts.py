from newsapi import NewsApiClient


def create_news():
    run = True;
    while run:
        topic = input("type a an article topic:")
        if len(topic) > 2:
            newsapi = NewsApiClient(api_key='e43b9508c4954478a8e5a1ea0f6f000c')
            top_headlines = newsapi.get_top_headlines(q=topic, language='en')
            content_and_url = []
            for c_a in top_headlines.get("articles"):
                content_and_url.append((c_a.get("content"), c_a.get("url"), c_a.get("description")))
            if len(content_and_url) < 1:
                print("choose a different topic, topic in top headlines is not available. ")
            else:
                return content_and_url


# content_url_description_list is a list of articles and each element is in a tuple goes like content url description
# content_and_url[(content,url,description)]
content_url_description_list = create_news()
print(content_url_description_list)