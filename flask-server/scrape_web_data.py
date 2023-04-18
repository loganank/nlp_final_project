import requests
import time
import random
import pandas as pd

# Create data, only needs run once


# Web-Scraper for Reddit Data

# Adapted from https://github.com/ayaanzhaque/SDCNL/blob/main/web-scraper.py

# creating user agent
headers = {"User-agent" : "randomuser"} # set user agent to reddit account username
url_1 = "https://www.reddit.com/r/depression.json"

res = requests.get(url_1, headers=headers)
res.status_code

# scraper function
def reddit_scrape(url_string, number_of_scrapes, output_list):
    # scraped posts outputted as lists
    after = None
    for _ in range(number_of_scrapes):
        if _ == 0:
            print("SCRAPING {}\n--------------------------------------------------".format(url_string))
            print("<<<SCRAPING COMMENCED>>>")
            print("Downloading Batch {} of {}...".format(1, number_of_scrapes))
        elif (_ + 1) % 5 == 0:
            print("Downloading Batch {} of {}...".format((_ + 1), number_of_scrapes))

        if after == None:
            params = {}
        else:
            # THIS WILL TELL THE SCRAPER TO GET THE NEXT SET AFTER REDDIT'S after CODE
            params = {"after": after}
        res = requests.get(url_string, params=params, headers=headers)
        if res.status_code == 200:
            the_json = res.json()
            output_list.extend(the_json["data"]["children"])
            after = the_json["data"]["after"]
        else:
            print(res.status_code)
            break
        time.sleep(random.randint(1, 6))

    print("<<<SCRAPING COMPLETED>>>")
    print("Number of posts downloaded: {}".format(len(output_list)))
    print("Number of unique posts: {}".format(len(set([p["data"]["name"] for p in output_list]))))


# remove any repeat posts
def create_unique_list(original_scrape_list, new_list_name):
    data_name_list = []
    for i in range(len(original_scrape_list)):
        if original_scrape_list[i]["data"]["name"] not in data_name_list:
            new_list_name.append(original_scrape_list[i]["data"])
            data_name_list.append(original_scrape_list[i]["data"]["name"])
    # CHECKING IF THE NEW LIST IS OF SAME LENGTH AS UNIQUE POSTS
    print("LIST NOW CONTAINS {} UNIQUE SCRAPED POSTS".format(len(new_list_name)))


# scraping suicide_watch data
suicide_data = []
reddit_scrape("https://www.reddit.com/r/SuicideWatch.json", 50, suicide_data)

suicide_data_unique = []
create_unique_list(suicide_data, suicide_data_unique)

# add suicide_watch to dataframe
suicide_watch = pd.DataFrame(suicide_data_unique)
suicide_watch["label"] = 0
suicide_watch.head()

# scraping depression data
depression_data = []
reddit_scrape("https://www.reddit.com/r/depression.json", 50, depression_data)

depression_data_unique = []
create_unique_list(depression_data, depression_data_unique)

# add depression to dataframe
depression = pd.DataFrame(depression_data_unique)
depression["label"] = 1
depression.head()

subreddits = ["AskReddit", "news", "politics", "worldnews", "technology", "science", "programming"]
subreddits_data_unique = []
for subreddit in subreddits:
    # scraping subreddit data
    subreddit_data = []
    reddit_scrape(f"https://www.reddit.com/r/{subreddit}.json", 50, subreddit_data)

    subreddit_data_unique = []
    create_unique_list(subreddit_data, subreddit_data_unique)
    subreddits_data_unique.extend(subreddit_data_unique)

# add subreddit to dataframe
subreddits = pd.DataFrame(subreddits_data_unique)
subreddits["label"] = 2

# saving data
suicide_watch.to_csv('suicide_watch.csv', index=False)
depression.to_csv('depression.csv', index=False)
subreddits.to_csv('subreddits.csv', index=False)

# creating combined CSV
suicide_watch = pd.read_csv('suicide_watch.csv')
depression = pd.read_csv('depression.csv')
subreddits = pd.read_csv('subreddits.csv')

sui_columns = suicide_watch[["title", "selftext", "author", "num_comments", "label", "url"]]
dep_columns = depression[["title", "selftext", "author", "num_comments", "label", "url"]]
sub_columns = subreddits[["title", "selftext", "author", "num_comments", "label", "url"]]

combined_data = pd.concat([sui_columns, dep_columns, sub_columns], axis=0, ignore_index=True)
combined_data["selftext"].fillna("emptypost", inplace=True)
combined_data.head()
combined_data.isnull().sum()

# saving combined CSV
combined_data.to_csv('suicide_vs_depression_vs_generic.csv', index=False)