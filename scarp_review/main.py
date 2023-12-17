import os
import streamlit as st
import pickle
import time
import pandas as pd
import re
import streamlit as st
import subprocess
import os
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import subprocess
#from ecom_prod_scraper import AmazonScraper
load_dotenv()  # take environment variables from .env (especially openai api key)
def convert_to_pdf(url, output_file):
    # Check if the output file already exists, and delete it if it does
    command = [
        "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
        "--no-images",  # Ignore external images (optional, adjust as needed)
        url,
        output_file
    ]
    try:
        # Run the command
        subprocess.run(command, check=True)
        st.success("Conversion successful!")
    except subprocess.CalledProcessError as e:
        st.error(f"Error during conversion: {e}")
def product_bot_page():
    st.subheader("Product Bot 📦")
    # Get user input (URL)
    url = st.text_input("Enter URL for PDF conversion:")

    if url:
        pdf_name = "output1.pdf"  # Modify the output file name as needed
        if os.path.exists(pdf_name ):
            os.remove(pdf_name )
    # Check if the corresponding .pkl file exists, and delete it if it does
        pkl_file = pdf_name [:-4] + ".pkl"
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
        convert_to_pdf(url, pdf_name)
        pdf_reader = PdfReader(pdf_name)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf_name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
# Define page names
PAGES = {
    "Product Bot":"Product Bot",
    "Product Visualization": "Product Visualization",
    "Amazon Reviews Scraper": "Amazon Reviews Scraper",
}

load_dotenv() 
# Create a navigation sidebar
selection = st.sidebar.radio("Navigation", list(PAGES.keys()))
        
if selection == PAGES["Product Visualization"]:
    from webscrapping import AmazonScraper, execCode
    from ProjectVisualisations import visualiseQueryReults
    from ProductReviewAnalysis import printTableProducts, showProductDeets

    class color:
        BLUE = '\033[94m'
        BOLD = '\033[1m'
        END = '\033[0m'

    st.title("Product Analysis")

    qry = st.text_input("Enter Product Name : ")
    brand = st.text_input("Enter Brand (if any) :")

    if st.button("Submit"):
        try:
            st.text("PLEASE GRAB A BITE! WE WILL TAKE 2 MINS")
            df, brand_present_df = execCode(qry, brand)

            st.markdown(("AMAZON" ).center(100, "-"))
            st.markdown(("PRODUCT QUERIED : " + qry))
            st.markdown(("BRAND : " + brand ))

            # Plotting files using the visualiseQueryReults() method created in ProjectVisualisations
            visualiseQueryReults(df, brand_present_df)

            # Calling method from ProductReviewAnalysis.py to display products in the dataset in a neat manner
            printTableProducts(df)
        except ValueError as e:
            st.text("Oops! Sorry, could not fetch your request for this search. Please try another product.")
if selection == PAGES["Amazon Reviews Scraper"]:
    st.title("Amazon Reviews Scraper")

    amazon_url = st.text_input("Enter Amazon product URL:")

# Text input field for ASIN code
    asin_pattern = r'/product-reviews/([A-Z0-9]{10})'

# Use re.search to find the pattern in the URL
    match = re.search(asin_pattern, amazon_url)

# Check if a match is found
    if match:
    # Extract the ASIN code from the match
        asin_code = match.group(1)
    if st.button("Scrape Amazon Reviews"):
        
        if not amazon_url and not asin_code:
            st.write("Please enter an Amazon product URL")
        else:
            st.write("Scraping in progress... Please wait.")
            if os.path.exists(r'C:\Users\Ananya Gupta\Desktop\help\scarp_review\amazon_reviews.csv'):
                os.remove(r'C:\Users\Ananya Gupta\Desktop\help\scarp_review\amazon_reviews.csv')
            
            # Determine whether to use URL or ASIN code
            if amazon_url:
                input_argument = f"url={amazon_url}"
            # Run the Scrapy spider as a separate process
            try:
                subprocess.run([r"scrapy", "runspider", r"C:\Users\Ananya Gupta\Desktop\help\scarp_review\scrap_review\spiders\scrape_reviews.py", "-a", input_argument, "-o", "amazon_reviews.csv"])
                st.write("Scraping complete.")
                st.write("Data saved to 'amazon_reviews.csv'")
            except Exception as e:
                st.write(f"Error: {e}")


    # Button to start scraping
    # Display the scraped data if available
    if st.checkbox("Show Scraped Data"):
        try:
            df = pd.read_csv(r'C:\Users\Ananya Gupta\Desktop\help\scarp_review\amazon_reviews.csv')
            df.insert(0, 'ASIN', asin_code)
            if df.empty:
                st.write("No reviews found for the given ASIN.")
            else:
                st.dataframe(df)
        except FileNotFoundError:
            st.write("No data available. Please run the scraper first.")
        except pd.errors.EmptyDataError:
            st.write("No reviews found for the given ASIN.")
    
    # Button to run review.py file
    if st.button("Run review.py"):
        try:
            # Start the Streamlit app defined in review.py
            subprocess.Popen(["streamlit", "run", r"C:\Users\Ananya Gupta\Desktop\help\scarp_review\review.py"])
            st.write("Review app is now running.")
        except Exception as e:
            st.write(f"Error: {e}")
if selection == PAGES["Product Bot"]:
    product_bot_page()

