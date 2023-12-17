# Importing Required libraries and modules

from bs4 import BeautifulSoup
import requests
import time
import datetime
import streamlit as st
import smtplib
import re
import pandas as pd
import seaborn as sns
import datetime as dt
import numpy as np
# Finding user agent at: https://httpbin.org/get
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

# Change some of seaborn's style settings with `sns.set()`
sns.set(style="ticks",  # The 'ticks' style
        rc={"figure.figsize": (6, 9),  # width = 6, height = 9
            "figure.facecolor": "ivory",  # Figure colour
            "axes.facecolor": "ivory"})  # Axes colour

# Importing files created by me, and saved in. webscrapping.py
from ecom_prod_scraper import AmazonScraper
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


"""
User Defined Functions
"""


# User Defined Function To neatly print the products of the particular dataset

from urllib.parse import urlparse, parse_qs
def extract_asin(url):
    # Define a regular expression pattern to match ASINs in Amazon URLs
    pattern = r'/dp/([A-Z0-9]{10})'
    
    # Use re.search to find the ASIN in the URL
    match = re.search(pattern, url)
    
    if match:
        asin = match.group(1)
        return asin
    else:
        return None


def printTableProducts(df):
    filtered_df = df[df["ecommerce_website"] == "Amazon"]
    filtered_df["ASIN"] = filtered_df["product_url"].apply(extract_asin)
    st.write(filtered_df)



# User Defined Function To Show the selected products details in a neat manner
def showProductDeets(df, index):
    element = df.iloc[index]

    st.write((color.BOLD + color.BLUE + 'PRODUCT DETAILS' + color.END).center(100, "-"))
    st.write()
    st.write(color.BOLD + 'Product Name : ' + color.END, element[2])
    st.write(color.BOLD + 'Product Price : ' + color.END, element[3])
    st.write(color.BOLD + 'Average Product Rating : ' + color.END, element[4])
    st.write(color.BOLD + 'Total Number of Ratings : ' + color.END, element[5])
    st.write()

    # Display the image using Streamlit
    img_url = df.iloc[index][6]
    st.image(img_url, caption='Product Image', use_container_width=True)

    st.write("-" * 100)