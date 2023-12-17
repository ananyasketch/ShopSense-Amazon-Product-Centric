from bs4 import BeautifulSoup
import requests
import time
import datetime
import pandas as pd
import streamlit as st
import smtplib
import re
import pandas as pd
import seaborn as sns
import datetime as dt
import numpy as np
from dateutil import parser
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from IPython.display import clear_output
from tabulate import tabulate
import urllib.request
from PIL import Image 
from IPython.display import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)
sns.set(style="ticks",  # The 'ticks' style
        rc={"figure.figsize": (6, 9),  # width = 6, height = 9
            "figure.facecolor": "ivory",  # Figure colour
            "axes.facecolor": "ivory"})  # Axes colour
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def mapValue(element):
    
    # 'Positive', 'Neutral', 'Negative' unqiue values
    if element == 'Positive':
        return 1

    if element == 'Neutral':
        return 0

    else:
        return -1

def getVaderDf(df):
    global sia
    sia = SentimentIntensityAnalyzer()
    # Runn the polarity score on the entire dataset
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Vader Model"):
        text = row['text']
        index = list(df.index)[i]
        val = sia.polarity_scores(text)
        res[index] = {
            'vader_neg': float(val['neg']),
            'vader_pos': float(val['pos']),
            'vader_comp': float(val['compound'])
        }

    # Converting thed dictionary into a dataframe and using transpose to make columns as rows
    vaders = pd.DataFrame(res).T
    vaders.reset_index(inplace=True, drop=True)
    # Merfing the vaders dataframe with the original dataset

    return vaders
def polarity_scores_roberta(example, max_length=512):
    # Truncate or split the input text if it's too long
    if len(example) > max_length:
        example = example[:max_length]

    encoded_text = tokenizer(example, return_tensors='pt', truncation=True, max_length=max_length)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

def getRobetaDf(df):

    # Runn the polarity score on the entire dataset
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Roberta Model"):
        text = row['text']
        index = list(df.index)[i]
        val = polarity_scores_roberta(text)
        res[index] = val

    # Converting thed dictionary into a dataframe and using transpose to make columns as rows
    df = pd.DataFrame(res).T
    df.reset_index(inplace=True, drop=True)

    return df
sia = SentimentIntensityAnalyzer()
def getVaderSentiment(df):
    arr = []
    for x in range(len(df)):
        text = df["text"][x]  # Use the 'text' column from your dataset

        val = sia.polarity_scores(text)
        vader_compound_score = val['compound']

        if vader_compound_score >= 0.05:
            arr.append("Positive")
        elif vader_compound_score <= -0.05:
            arr.append("Negative")
        else:
            arr.append("Neutral")

    if 'vader_sentiment' not in df.columns:
        df.insert(loc=6, column='vader_sentiment', value=arr)
    else:
        df['vader_sentiment'] = arr

def getRobertaSentiment(df):
    arr = []
    for pos in range(len(df)):
        # Assuming 'text' is the column containing the review content
        text = df["text"].iloc[pos]

        # Calculate sentiment scores using your polarity_scores_roberta function or any other method you have
        scores = polarity_scores_roberta(text)

        # Use the correct column names from your dataset for sentiment scores
        roberta_sentiment_dict = {
            "Positive": scores["roberta_pos"],
            "Negative": scores["roberta_neg"],
            "Neutral": scores["roberta_neu"]
        }

        roberta_sentiment_value = max(roberta_sentiment_dict, key=roberta_sentiment_dict.get)
        arr.append(roberta_sentiment_value)

    if 'roberta_sentiment' not in df.columns:
        df.insert(loc=6, column='roberta_sentiment', value=arr)
    else:
        df['roberta_sentiment'] = arr

def getFinalSentiment(df):
    arr = []
    roberta_sentiment_vals = list(map(mapValue, list(df['roberta_sentiment'])))
    vader_sentiment_vals = list(map(mapValue, list(df['vader_sentiment'])))

    roberta_val = list(df['roberta_sentiment'])
    vader_val = list(df['vader_sentiment'])

    for x in range(len(df)):
        val = float((roberta_sentiment_vals[x] * 0.6) + (vader_sentiment_vals[x] * 0.4))

        if val >= 0.19:
            arr.append("Positive")

        elif val >= (-0.4):
            arr.append("Neutral")


        elif val >= (-1):
            arr.append("Negative")

    if 'sentiment' not in df.columns:
        df.insert(loc=6,
                  column='sentiment',
                  value=arr)

    else:
        df['sentiment'] = arr
def ratingsItemPlot(df):
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)

    rev_count = list(df['rating'].value_counts().sort_index())
    rev_tags = list(df['rating'].unique())
    rev_tags.sort()

    plt.bar(rev_tags, rev_count)

    plt.plot(rev_tags, rev_count, color='black', linewidth=7.0)

    plt.xlabel('Review Stars', fontsize=16, color="red")
    plt.ylabel('Review Count', fontsize=16, color="red")

    plt.suptitle("Reviews Count for the product", fontsize=30, color='b')

    plt.subplot(1, 2, 2)
    #myexplode = [0.15, 0.19, 0.22, 0.12, 0.15]

    plt.pie(x=rev_count,
            shadow=True, autopct='%.0f%%')

    plt.legend(rev_tags)
    # plt.rcParams.update({'font.size': 10})

    plt.tight_layout()

    # Use st.pyplot() to display the plot in Streamlit
    st.pyplot(plt)

# User defined function to count and show a woordle like frequency chart
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1  # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    st.image(wordcloud.to_array(), use_column_width=True)
    
# Graph that compares the review length and the ratings given
def sizeComPlot(df):
    st.subheader("Product Rating vs Review Length")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    
    sns.pointplot(data=df, x="rating", y="review_size", ax=ax1)
    ax1.set_xlabel("Rating", fontsize=16, color="red")
    ax1.set_ylabel("Review Length", fontsize=16, color="red")
    ax1.set_title("Point Plot", fontsize=16)
    
    sns.boxplot(data=df, x="rating", y="review_size", ax=ax2)
    ax2.set_xlabel("Rating", fontsize=16, color="red")
    ax2.set_ylabel("Review Length", fontsize=16, color="red")
    ax2.set_title("Box Plot", fontsize=16)
    
    st.pyplot(fig)


# User defined function to compare Vader and Roberta Model through dist plot
import seaborn as sns
import matplotlib.pyplot as plt

def compModelsGraph(df):
    st.subheader("In-Depth VADER and ROBERTA Model Analysis")
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    plt.suptitle("In-Depth VADER and ROBERTA Model Analysis", color="blue", fontsize=30)

    axs[0] = plt.subplot(1, 3, 1)
    sns.distplot(list(df['vader_compound']), kde_kws=dict(linewidth=7))
    sns.distplot(df['roberta_pos'], kde_kws=dict(linewidth=7))
    plt.xlabel("VADER COMPOUND SCORE", color='red', fontsize=16)
    plt.ylabel("DENSITY", color='red', fontsize=16)
    plt.legend(["VADER MODEL", "ROBERTA MODEL"])

    axs[1] = plt.subplot(1, 3, 2)
    sns.distplot(list(df['vader_neg']), kde_kws=dict(linewidth=7))
    sns.distplot(df['roberta_neg'], kde_kws=dict(linewidth=7))
    plt.xlabel("VADER NEGATIVE SCORE", color='red', fontsize=16)
    plt.ylabel("DENSITY", color='red', fontsize=16)
    plt.legend(["VADER MODEL", "ROBERTA MODEL"])

    axs[2] = plt.subplot(1, 3, 3)
    sns.distplot(list(df['vader_pos']), kde_kws=dict(linewidth=7))
    sns.distplot(df['roberta_neu'], kde_kws=dict(linewidth=7))
    plt.xlabel("VADER POSITIVE SCORE", color='red', fontsize=16)
    plt.ylabel("DENSITY", color='red', fontsize=16)
    plt.legend(["VADER MODEL", "ROBERTA MODEL"])

    st.pyplot(fig)


# Plotting Vader Scores vs Ratings
def plotVaderResults(df):
    st.subheader("VADER Model Analysis")
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    sns.barplot(data=df, x='rating', y='vader_comp', ax=axs[0], ci=None)
    axs[0].set_title("Compound Score (VADER)")

    sns.barplot(data=df, x='rating', y='vader_neg', ax=axs[1], ci=None)
    axs[1].set_title("Negative Score (VADER)")

    sns.barplot(data=df, x='rating', y='vader_pos', ax=axs[2], ci=None)
    axs[2].set_title("Positive Score (VADER)")

    plt.tight_layout()

    st.pyplot(fig)

# Plotting Roberta Model Scores vs Ratings
def plotRobertaResults(df):
    st.subheader("ROBERTA Model Analysis")
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    sns.barplot(data=df, x=df['rating'], y=df['roberta_neu'], ax=axs[0], ci=None)
    axs[0].set_title("Neutral Score (ROBERTA MODEL)")

    sns.barplot(data=df, x=df['rating'], y=df['roberta_neg'], ax=axs[1], ci=None)
    axs[1].set_title("Negative Score (ROBERTA MODEL)")

    sns.barplot(data=df, x=df['rating'], y=df['roberta_pos'], ax=axs[2], ci=None)
    axs[2].set_title("Positive Score (ROBERTA MODEL)")

    plt.tight_layout()

    st.pyplot(fig)
# Density Plot for final sentiment density (Date wise)
def sentimentDensityPlot(df):
    st.subheader("Weighted Sentiment Spread")
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df, x='date', hue="sentiment", palette="icefire", multiple="fill")
    plt.title("Weighted Sentiment Spread", fontsize=30, color='b')
    plt.xlabel("Review Date", fontsize=16, color="red")
    plt.ylabel("Density", fontsize=16, color="red")

    st.pyplot(plt)



# Bar Chart to compare the Sentiment values of ROBERTA and VADER model
def modelValsComp(df):
    rob_vad_vals = list(df["vader_sentiment"]) + list(df["roberta_sentiment"])
    which_vals = ["VADER" for x in range(len(df))] + ["ROBERTA" for x in range(len(df))]
    temp_df = pd.DataFrame({"Sentiment": rob_vad_vals, "Model": which_vals})

    st.subheader("ROBERTA VS VADER MODEL SENTIMENT COMPARISON")
    plt.figure(figsize=(12, 7))
    sns.histplot(data=temp_df, x="Sentiment", hue="Model", multiple="dodge", shrink=.8)
    plt.title("ROBERTA VS VADER MODEL SENTIMENT COMPARISON", fontsize=30, color='b')
    plt.xlabel("Sentiment", fontsize=16, color="red")
    plt.ylabel("Review Count", fontsize=16, color="red")

    st.pyplot(plt)

# Bar Chart and Pie Chart of weighted Sentiment Values
def sentimentPlot(df):
    pos_count = sum(df["sentiment"] == "Positive")
    neu_count = sum(df["sentiment"] == "Neutral")
    neg_count = sum(df["sentiment"] == "Negative")

    st.write("-> Positive Sentiment Reviews : ", pos_count)
    st.write("-> Neutral Sentiment Reviews : ", neu_count)
    st.write("-> Negative Sentiment Reviews : ", neg_count)

    st.subheader("Weighted 3:2 (ROBERTA MODEL : VADER MODEL) Comparison")
    plt.figure(figsize=(12, 7))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50

    plt.subplot(1, 2, 1)
    plt.bar(x=["Positive", "Neutral", "Negative"],
            height=[pos_count, neu_count, neg_count],
            color=["green", 'yellow', 'red'])

    plt.xlabel("Sentiment", fontsize=16, color="red")
    plt.ylabel("Count of reviews", fontsize=16, color="red")

    plt.subplot(1, 2, 2)
    plt.pie(x=[pos_count, neu_count, neg_count], labels=["Positive", "Neutral", "Negative"],
            colors=['green', 'yellow', 'red'], autopct='%.0f%%')

    plt.tight_layout()

    st.pyplot(plt)



def extract_location_and_date(text):
    parts = text.split(' on ')
    location_part = parts[0].replace("Reviewed in ", "").strip()
    
    if len(parts) > 1:
        date_part = parts[1].strip()
        # Parse the date string to a consistent date format (optional)
        try:
            parsed_date = parser.parse(date_part)
            date_part = parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            date_part = ""  # Handle invalid date format
    else:
        date_part = ""
    
    return pd.Series([location_part, date_part], index=['location', 'date'])



# Apply the function and preprocess the DataFrame
df=pd.read_csv(r'C:\Users\Ananya Gupta\Desktop\help\scarp_review\amazon_reviews.csv')
print(df.dtypes)
df['rating'] = df['rating'].astype(float)  # Convert 'rating' to float
print(df.dtypes)
df[['location', 'date']] = df['location_and_date'].apply(extract_location_and_date)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['text'] = df['text'].replace(np.nan, '')
df['review_size'] = df['text'].apply(len)
df = df.drop(columns=['location_and_date'])
getVaderSentiment(df)
getRobertaSentiment(df)
getFinalSentiment(df)

# Word cloud function
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(str(data))

    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(wordcloud)
    st.pyplot(plt)

# Streamlit app
st.title("Review Analysis App")

# Show word cloud
st.subheader("Word Cloud")
show_wordcloud(list(df['text']))

# Show sentiment comparison plot
st.subheader("Sentiment Comparison")
sentimentPlot(df)

# Show sentiment density plot
st.subheader("Sentiment Density")
sentimentDensityPlot(df)

# Show model values comparison plot
st.subheader("Model Sentiment Comparison")
modelValsComp(df)

# Show ratings item plot
st.subheader("Ratings Item Plot")
ratingsItemPlot(df)

# Show size comparison plot
st.subheader("Review Length vs Rating")
sizeComPlot(df)

