---
title: "scraping quora is fun"
layout: post
---

recently for a project, i had to collect social media data. the problem with that is, most of the popular sites just hate giving out their data. what they hate even more is being scraped. so i had to find a site that was popular, had a lot of data, and was easy to scrape. and i ended up with quora, which is pretty much the perfect site for this. it has a lot of data, and it is pretty easy to scrape. so i wrote a scraper for it, which you can find [here](https://github.com/viciousAegis/QuoraScraper)

### what it does
this scraper does a few things:
- given a list of search terms, it can scrape posts related to those search terms from quora. the scraped information includes: the post text, post author, list of users who upvoted the post and list of users who commented on the post. you can pass the number of posts scraped for each search term as a command line argument. default is 50.
- (experimental) given a list of users, scrape their followers, following and their bio
- (experimental) given a list of search terms, scrape the answers to those search terms. the scraped information includes: the answer text, answer author, etc.

### setting up
the scraper is primarily written using selenium, and is very simple to setup. i use chromedriver to run the scraper on google chrome, which is fine for most cases, and unless you know what you are doing, you should not change that. if you clone the above repo you will find chromedriver already installed. you might need to use a different version of chromedriver depending on your chrome version, you can check that [here](https://chromedriver.chromium.org/downloads). the other needed requirements are listed in the requirements.txt file.

### how to use
write your search terms in the `search.terms` file, one term per line. like this:
```
aaloo
baingan
ghobi
```
then run the scraper using `python3 run.py`. the scraped data will be saved in the `data` folder. the data is saved in json format, and the file name is the search term. so for the above example, the data will be saved in `data/aaloo.csv`, `data/baingan.csv` and `data/ghobi.csv`. the data is saved in the following format:
```
{
    "post_text": "post text",
    "post_author": "post author",
    "upvotes": ["user1", "user2", "user3"],
    "comments": ["user1", "user2", "user3"]
}
```
please run `python3 run.py --help` for more information on how to use the scraper.

the user scraper and the answer scrapers are a bit finicky, and i am still working on them. so if you can understand the source code, feel free to use them, otherwise the post scraper by itself should be enough for most use cases.

### caveats
since the scraper runs on selenium and includes interacting with certain web elements to scrape the data, you might have to change the code a bit depending on your browser/os. hopefully it works out of the box for you, but if it doesn't, i hope you know how selenium works :)

### conclusion
i hope this scraper helps you in your projects. if you have any questions, feel free to ask me on twitter (linked below). if you find any bugs, please open an issue on the github repo. if you want to contribute, please open a pull request. thanks for reading!