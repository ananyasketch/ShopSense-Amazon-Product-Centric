import streamlit as st
import pandas as pd
import subprocess
import re
st.title("Amazon Reviews Scraper")

# Text input field for Amazon URL
amazon_url = st.text_input("Enter Amazon product URL:")

# Text input field for ASIN code
asin_pattern = r'/product-reviews/([A-Z0-9]{10})'

# Use re.search to find the pattern in the URL
match = re.search(asin_pattern, amazon_url)

# Check if a match is found
if match:
    # Extract the ASIN code from the match
    asin_code = match.group(1)

# Button to start scraping
if st.button("Scrape Amazon Reviews"):
    if not amazon_url and not asin_code:
        st.write("Please enter an Amazon product URL")
    else:
        st.write("Scraping in progress... Please wait.")
        
        # Determine whether to use URL or ASIN code
        if amazon_url:
            input_argument = f"url={amazon_url}"
        # Run the Scrapy spider as a separate process
        try:
            subprocess.run([r"scrapy", "runspider", r"C:\Users\Ananya Gupta\Desktop\MAJOR\scarp_review\scrap_review\spiders\scrape_reviews.py", "-a", input_argument, "-o", "amazon_reviews.csv"])
            st.write("Scraping complete.")
            st.write("Data saved to 'amazon_reviews.csv'")
        except Exception as e:
            st.write(f"Error: {e}")

# Display the scraped data if available
if st.checkbox("Show Scraped Data"):
    try:
        
        df = pd.read_csv('amazon_reviews.csv')
        df.insert(0, 'ASIN', asin_code)
        st.dataframe(df)
    except FileNotFoundError:
        st.write("No data available. Please run the scraper first.")

# Button to run review.py file
if st.button("Run review.py"):
    try:
        # Start the Streamlit app defined in review.py
        subprocess.Popen(["streamlit", "run", r"C:\Users\Ananya Gupta\Desktop\MAJOR\scarp_review\review.py"])
        st.write("Review app is now running.")
    except Exception as e:
        st.write(f"Error: {e}")



