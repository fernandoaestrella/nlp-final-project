#!/usr/bin/env python
import newsapi as newsapi
from newsapi import NewsApiClient

#nyt_api = dqHDPIgNpuKxkiNoHbBEkAnKmYV42rLV
#google_api = e43b9508c4954478a8e5a1ea0f6f000c



all_articles = newsapi.get_everything(q='bitcoin',
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2017-12-01',
                                      to='2017-12-12',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)

# work with api
# retrieve information
# specify articles
# what it's about/who it's about