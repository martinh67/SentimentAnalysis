# import statements required
import time
import requests
from datetime import datetime
import pickle
import pandas as pd
import json
from methods import *
import pprint as pp
from nltk import bigrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')

# start the timer
start = time.time()


# main method
def main():

    '''
    # uncomment this code to build the files
    early_file = create_file("2019_data.pkl", 1554073200, 1554678000)
    later_file = create_file("2021_data.pkl", 1617231600, 1617836400)

    '''

    # open the earlier dates pickle file for reading
    early_file = open("2019_data.pkl", "rb")
    early_class = pickle.load(early_file)

    # open the later dates class file for reading
    later_file = open("2021_data.pkl", "rb")
    later_class = pickle.load(later_file)

    # build the dataframes
    df_early = build_dataframe(early_class)
    df_later = build_dataframe(later_class)

    # calculate the summations
    early_summation = calculate_total_summation(df_early)
    later_summation = calculate_total_summation(df_later)

    # plot the data
    plot_data(early_summation, later_summation)

    '''
    # uncomment this code to see the alternative approaches

    # declare a sentiment_intensity_analyzer object
    sentiment_intensity_analyzer = SentimentIntensityAnalyzer()

    # print the dicitonary of terms
    pp.pprint(sentiment_intensity_analyzer.make_lex_dict())

    # a dataframe used to limit the submissions
    df_early_limit = df_early[:100]

    # iterate through the dataframe
    for index, row in df_early_limit.iterrows():

        # print a space
        print()

        # print the text
        print(df_early_limit.body[index])

        # show where the sentiment is negated
        pp.pprint(mark_negation(df_early_limit.body[index].replace(", ", " , ").replace(". ", " . ").split()))

        # print a space
        print()


    # calculate the sentiment
    df_early_limit['polar_sent'] = df_early_limit.body.apply(determine_sentiment_polarity)

    # empty column to hold sentiment type
    df_early_limit['sentiment_type'] = ''

    # classify the sentiment
    df_early_limit.loc[df_early_limit.polar_sent > 0,'sentiment_type']='POSITIVE'
    df_early_limit.loc[df_early_limit.polar_sent == 0,'sentiment_type']='NEUTRAL'
    df_early_limit.loc[df_early_limit.polar_sent < 0,'sentiment_type']='NEGATIVE'

    # print the dataframe
    print(df_early_limit.head())

    '''


# magic method to run the main function
if __name__ == "__main__":

    # run main
    main()


# print the time of the program
print("\n" + 40*"#")
print(time.time() - start)
print(40*"#" + "\n")
