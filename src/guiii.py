import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

authors = []
dates = []
statements = []
sources = []
targets = []

# Function to read a CSV file
def read_csv_file(file):
    df = pd.read_csv(file)
    return df

st.title("Scraping and Fake news detection")
file = st.file_uploader("Upload CSV file", type=["csv"])
if file is not None:
    # Read the CSV file using the function
    df = read_csv_file(file)
    # Display the dataframe
    st.write(df)
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    tfidf = tfidf_vectorizer.fit_transform(df["statement"])
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf, df["Label"])

    # Create the Streamlit app
    st.write("Enter a news article and press submit to see if it is real or fake.")

    text = st.text_area("Enter text here", height=200)
    if st.button("Submit"):
        text = [text]
        text_tfidf = tfidf_vectorizer.transform(text)
        result = pac.predict(text_tfidf)
        if result[0] == 1:
            st.write("This news article is real.")
        else:
            st.write("This news article is fake.")


def scrape_website(page_number):
    page_num = str(page_number)
    URL = 'https://www.politifact.com/factchecks/list/?page=2&' + page_num
    webpage = requests.get(URL)
    soup = BeautifulSoup(webpage.text, 'html.parser')
    #Get the location of the information
    statement_footer = soup.find_all('span', attrs={'class':'article_date'}) #Location of the author
    statement_quote = soup.find_all('div', attrs={'class':'article_title'})  #Location of the statement
    statement_meta = soup.find_all('div', attrs={'class':'m-statement__meta'}) # #Location of the source
    target =  soup.find_all('div', attrs={'class':'m-statement__meter'})  #Location of the target (score card)

    #loop through the statement_footer
    for i in statement_footer:
        link1 =i.text.strip()
        name_and_date = link1.split()
        first_name = name_and_date[1]
        last_name = name_and_date[2]
        full_name = first_name +' '+ last_name
        month =  name_and_date[4]
        day = name_and_date[5]
        year = name_and_date[6]
        date = month+' '+day+' '+year
        dates.append(date)
        authors.append(full_name)

    #loop through the statement_quote 
    for i in statement_quote:
        link2 = i.find_all('a')
        statement_text = link2[0].text.strip()
        statements.append(statement_text)
    #loop through the meta
    for i in statement_meta:
        link3 = i.find_all('a')
        source_text = link3[0].text.strip()
        sources.append(source_text)
    #loop through the target 
    for i in target:
        link4 = i.find('div', attrs={'class':'c-image'}).find('img').get('alt')
        targets.append(link4)


#loop through 'n-1' webpage(s) to scrapecthe data
def run_scrape():
    n = 2
    for i in range(1, n):
        scrape_website(i)
    dff = pd.DataFrame({
    "Name": [authors],
    "statements": [statements],
    "source" : [sources],
    "date" : [dates],
    "target" : [targets]
    })
    # If a file was uploaded
    # Define the Streamlit app
    st.dataframe(dff, height=300)

if st.button("Run Scraping"):
    # Call the function when the button is clicked
    run_scrape()






    




