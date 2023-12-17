# Importing required libraries for further analysis
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import random
import matplotlib.pyplot as plt
import warnings
import time
import plotly.express as px
import matplotlib.patheffects as pe
import regex as re

warnings.filterwarnings("ignore")

# Finding user agent at: https://httpbin.org/get

# amazon orange, flipkart yellow
colors = ["#FEBD69"]
customPalette = sns.set_palette(sns.color_palette(colors))

# Change some of seaborn's style settings with `sns.set()`
sns.set(style="ticks",  # The 'ticks' style
        rc={"figure.figsize": (6, 9),  # width = 6, height = 9
            "figure.facecolor": "ivory",  # Figure colour
            "axes.facecolor": "ivory"},
        palette=customPalette)  # Axes colour

# dark black-blue amazon combo, flipkart blue
colors_2 = ["#37475A"]
import streamlit as st

def countActualProductsPlot(df, df_correct):
    st.subheader("Count of Relevant and Irrelevant Products")
    amazon_prod_count = sum(df["ecommerce_website"] == "Amazon")
    actual_amazon_prod_count = sum(df_correct["ecommerce_website"] == "Amazon")

    st.write(f"Amazon has {amazon_prod_count} products listed on each page for the queried product.")
    st.write(f"Out of which, {amazon_prod_count - actual_amazon_prod_count} are of other brands.")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor('ivory')
    ax.bar(["Amazon"], amazon_prod_count - actual_amazon_prod_count, color='red', edgecolor=colors_2)
    ax.bar(["Amazon"], actual_amazon_prod_count, bottom=amazon_prod_count - actual_amazon_prod_count, color=colors, edgecolor=colors_2)

    ax.set_xlabel("E-Commerce Website", fontsize=16, color="red")
    ax.set_ylabel("Count of Products", fontsize=16, color="red")
    ax.legend(["Irrelevant Products", "Relevant Products Amazon"])

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.pie([actual_amazon_prod_count, amazon_prod_count - actual_amazon_prod_count], explode=[0.4, 0], shadow=True, colors=["g", "r"])
    ax2.legend(["Relevant Products", "Irrelevant Products"])
    ax2.set_title("Relevance of Amazon Products", fontsize=16, color="b")

    st.pyplot(fig)
    st.pyplot(fig2)

import streamlit as st

def countProductsPlot(df):
    amazon_prod_count = sum(df["ecommerce_website"] == "Amazon")

    st.subheader("Number of Products Listed Per Page")
    st.write(f"Amazon has {amazon_prod_count} products listed on each page for the queried product.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    fig.set_facecolor('ivory')

    sns.countplot(x='ecommerce_website', data=df, color=customPalette, linewidth=3, edgecolor=colors_2, ax=ax1)
    ax1.set_xlabel("Ecommerce Website", fontsize=16, color="red")
    ax1.set_ylabel("No. of Products per page", fontsize=16, color="red")

    ax2.pie([amazon_prod_count], labels=["Amazon"], colors=customPalette, autopct='%.0f%%')
    ax2.axis("equal")

    st.pyplot(fig)


# Function to find average of a clolumn for the particular e-commerce website
def subset_avg_gen(df, website, col):
    df_subset = df[df["ecommerce_website"] == website]
    average = df_subset[col].mean()
    return average


# User Defined Function : Average Price of the products displayed upon query, website wise
def priceCompLine(df):
    amazon_prices = df[df["ecommerce_website"] == "Amazon"]["product_price"]

    st.subheader("Comparing Prices (Line Plot)")

    plt.figure(figsize=(15, 10))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    overlapping = 0.1

    plt.plot(amazon_prices, lw=6, path_effects=[pe.SimpleLineShadow(shadow_color=colors_2[0]), pe.Normal()])
    plt.legend(["Amazon"], loc='upper left')
    plt.xlabel("Product No.", color="red")
    plt.ylabel("Price (in rs.)", color="red")
    plt.title("Comparing Prices (line-plot)", fontsize=30, c='b')

    st.pyplot(plt)



def CompPriceBox(df):
    amz_avg_price = subset_avg_gen(df, "Amazon", "product_price")

    st.subheader("Comparing Prices (Box Plot)")
    st.write(f"Average Product Price for queried product at Amazon is: {amz_avg_price}")

    plt.figure(figsize=(15, 10))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50

    fig, ax = plt.subplots()
    b = sns.boxplot(data=df,
                    x="ecommerce_website",
                    y="product_price",
                    width=0.4,
                    linewidth=2,
                    showfliers=False,
                    ax=ax)

    b = sns.stripplot(data=df,
                      x="ecommerce_website",
                      y="product_price",
                      linewidth=1,
                      alpha=0.4,
                      ax=ax)

    ax.set_ylabel("Price", fontsize=14, color="red")
    ax.set_xlabel("E-Commerce Website", fontsize=14, color="red")
    ax.set_title("Comparing Prices (box-plot)", fontsize=30, color='b')

    sns.despine(offset=5, trim=True)
    
    st.pyplot(fig)


# User Defined Function : Number of products available on Amazon and flipkart for the given querry
def priceNamePlotly(df):
    fig = px.line(df, x="product_name", y="product_price",
                  color="ecommerce_website", template="plotly_dark",
                  color_discrete_sequence=colors,
                  title="Products and their price")

    fig.update_xaxes(showticklabels=False)


    return fig.show()


# User Defined Function : Average Rating of the products displayed upon query, website wise
def CompRatingBox(df):
    amz_avg_ratings = subset_avg_gen(df, "Amazon", "product_rating")

    st.write("Average Product Rating for queried product at amazon is : ", amz_avg_ratings)

    plt.figure(figsize=(15, 10))

    # Box plot
    b = sns.boxplot(data=df,
                    x="ecommerce_website",
                    y="product_rating",
                    width=0.4,
                    linewidth=2,

                    showfliers=False)
    # Strip plot
    b = sns.stripplot(data=df,
                      x="ecommerce_website",
                      y="product_rating",
                      # Colours the dots
                      linewidth=1,
                      
                      alpha=0.4)

    b.set_ylabel("Ratings (out of 5)", fontsize=14, color="red")
    b.set_xlabel("E-Commerce Website", fontsize=14, color="red")
    b.set_title("Comparing Ratings (box-plot)", fontsize=30, c='b')

    sns.despine(offset=5, trim=True)
    b.get_figure()
    return b

def priceNamePlotly(df):
    st.subheader("Products and Their Prices (Plotly Line Chart)")

    fig = px.line(df, x="product_name", y="product_price",
                  color="ecommerce_website", template="plotly_dark",
                  color_discrete_sequence=colors,
                  title="Products and their price")

    fig.update_xaxes(showticklabels=False)
    
    st.plotly_chart(fig)

# User Defined Function : Analysing the relationship between Product's Price and its Rating
# Function that plots a scatter plot between price and ratings
def price_rat_corr(df):
    price_rating_corr = df.corr()['product_rating']["product_price"]

    st.subheader("Analyzing the Relationship Between Product's Price and Ratings")

    if abs(price_rating_corr) > 0.9:
        st.write("Product Ratings and Product Prices are very highly correlated.")
    elif abs(price_rating_corr) > 0.7:
        st.write("Product Ratings and Product Prices are highly correlated.")
    elif abs(price_rating_corr) > 0.5:
        st.write("Product Ratings and Product Prices are moderately correlated.")
    elif abs(price_rating_corr) > 0.3:
        st.write("Product Ratings and Product Prices have low correlation.")
    else:
        st.write("Product Ratings and Product Prices have very little to no correlation.")

    fig = px.scatter(df, x="product_rating", y="product_price",
                     color="ecommerce_website", template="plotly_dark",
                     color_discrete_sequence=colors,
                     trendline="ols",
                     title="Analysing the relationship between Product's Price and Product's Ratings")

    fig.update_xaxes(showticklabels=True)

    st.plotly_chart(fig)


"""
Main function for vsiualising all the results
"""


def visualiseQueryReults(df, brand_present_df):
    # amazon orange, flipkart yellow
    colors = ["#FEBD69"]
    customPalette = sns.set_palette(sns.color_palette(colors))

    # Change some of seaborn's style settings with `sns.set()`
    sns.set(style="ticks",  # The 'ticks' style
            rc={ "figure.facecolor": "ivory",  # Figure colour
                "axes.facecolor": "ivory"},
            palette=customPalette)  # Axes colour

    ax1 = countProductsPlot(df)
    if len(brand_present_df)==len(df):
        pass
    else:
        ax2 = countActualProductsPlot(df, brand_present_df)

    ax3 = priceCompLine(brand_present_df)
    ax4 = CompPriceBox(brand_present_df)
    ax5 = priceNamePlotly(brand_present_df)

    ax6 = CompRatingBox(brand_present_df)

    ax7 = price_rat_corr(brand_present_df)
