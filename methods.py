# import statements required
import time
import requests
from datetime import datetime
import pickle
import pandas as pd
import json
import matplotlib.pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk

# download the vader module required
nltk.download('vader_lexicon')

# determine sentiment for each row
def determine_sentiment_polarity(row):

    # declare SentimentIntesityAnalyzer object
    sid = SentimentIntensityAnalyzer()

    # set the polarity score of each row
    cs = sid.polarity_scores(row)

    # return the compound score
    return(cs["compound"])


# method to extract the day from the row of data
def extract_day(row):

    # set the datetime for each row
    date = datetime.fromtimestamp(row)

    # return an integer of the day
    return date.day


# class to define a reddit submission
class redditSubmission:

    # initialisation method
    def __init__(self):

        # set the body to an empty string
        self.body = ""

        # set the created date to an empty
        self.created_utc = ""


# get data from the website
def pull_shift_pull(start_stamp, end_stamp):

    # specify the subreddit
    subreddit = "ireland"

    # url injection
    # take 100 submissions
    url = "https://api.pushshift.io/reddit/search/?limit=100&after={}&before={}&subreddit={}"

    # declare a list class
    list_class = []

    # while the starting date is less than the end date
    while start_stamp < end_stamp:

        # stop bombardment of reddit
        time.sleep(1)

        # update the url with the variables
        update_url = url.format(start_stamp, end_stamp, subreddit)

        # make a json request
        json = requests.get(update_url)

        # pass data to json object
        json_data = json.json()

        # if data is not in json data
        if "data" not in json_data:

            # break from the loop
            break

        # otherwise
        else:

            # set the json_data to the data from the api call
            json_data = json_data['data']

            # print the length of the data returned
            print(len(json_data))

            # if there is no data
            if len(json_data) == 0:

                # print no more data
                print("no more data to harvest")

                # break from the loop
                break

            # try
            try:

                # set the stamp to the last entry of the list
                start_stamp = json_data[-1]['created_utc']

            # handle exceptions
            except:

                # set the start stamp to the end stamp to end loop
                start_stamp = end_stamp

            # use a list to store the data
            list_class = process_json_data(json_data, list_class)

    # return the list class
    return list_class


# method to process json data
def process_json_data(data, list_class):

    # for all of the items in the data
    for item in data:

        # set a new reddit class object for the submission
        reddit_submission = redditSubmission()

        # set the body of the object to the body of the item
        reddit_submission.body = item['body']

        # set the time of the object to the time of the item
        reddit_submission.created_utc = item['created_utc']

        # append the object to the list class
        list_class.append(reddit_submission)

    # return the list class
    return list_class


# method to build the dataframes required
def build_dataframe(pickle_class):

    # create list to hold the dates
    dates = []

    # create a list to hold the body text
    text = []

    # for every object in the later class list
    for cls in pickle_class:

        # append the dates to a list
        dates.append(cls.created_utc)

        # append the body text to a list
        text.append(cls.body)

    # create a dataframe
    df = pd.DataFrame({"dates": dates, "body": text})

    # return the dataframe
    return df


# method to calculate the total summation of the polarity
def calculate_total_summation(df):

    # create columns
    df['days'] = df.dates.apply(extract_day)
    df['polar_sent'] = df.body.apply(determine_sentiment_polarity)

    # set a limit of the length of the later list
    limit = len(df.days.value_counts().tolist())

    # declare empty list for earlier
    total_summation = []

    # for every day in the range
    for i in range(1, limit + 1):

        # sum all of the instances of the polar sentiment
        one_total = df['polar_sent'][df['days'] == i].sum()

        # append the normalised data per day
        total_summation.append(one_total/df["days"].value_counts().tolist()[i-1])

    # return the list
    return total_summation


# method to plot the data from the dataframes
def plot_data(early_summation, later_summation):

    # declare the x axis
    x = list(range(1, 8))

    # plot the figure and axis
    fig, ax = plt.subplots()

    # label the earlier
    ax.plot(x, early_summation, color = "black", label = "earlier")

    # label the later
    ax.plot(x, later_summation, color = "red", label = "later")

    # add the legend to the plot
    leg = ax.legend()

    # show the plot
    plt.show()


# method to create the file
def create_file(filename, start_stamp, end_stamp):

    # declare the list class from the pull_shift_pull method
    list_class = pull_shift_pull(start_stamp, end_stamp)

    # open the file for writing
    picklefile = open(f"{filename}", "wb")

    # dump the list class and the file into
    pickle.dump(list_class, picklefile)
