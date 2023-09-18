import streamlit as st
import pandas as pd
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
import mysql.connector
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
import string
from nltk.tokenize import word_tokenize
from collections import Counter
import time

# Function to read a CSV file

def read_csv_file(file):
    df = pd.read_csv(file)
    return df

# Set up the sidebar
st.sidebar.title("Sidebar")
sidebar_option = st.sidebar.selectbox("Select an option", ("Beranda", "Upload CSV", "Scrape Website"))

# Main content
if sidebar_option == "Beranda":
    st.title("Selamat datang di Fake News Scraping and Analysis App!")
    st.write("Aplikasi ini adalah alat untuk mendeteksi dan menganalisis berita palsu secara otomatis. Dalam era informasi digital saat ini, penyebaran berita palsu atau fake news telah menjadi ancaman serius bagi kebenaran dan integritas informasi. Dengan menggunakan teknologi canggih, kami telah menciptakan aplikasi yang bertujuan untuk membantu Anda dalam memerangi berita palsu dan memastikan bahwa Anda menerima informasi yang akurat.")
    st.write('Terdapat pilihan pada menu opsi sidebar.\n1. Upload Data CSV\n2. Scraping Data')
    st.write('Didalam kedua opsi terdapat fitur yang sama tetapi pada Menu Upload Data CSV ada fitur untuk menambahkan Data latih')
elif sidebar_option == "Upload CSV":
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        # Read the CSV file using the function
        file_name = file.name
        file_directory = 'D:/Learning Program/kuliah/Semester 5/Informatika Capstone Project/tugas per bab/projek/projek hampir ok'
        filepath = f'{file_directory}/{file_name}'
        dataset_name = os.path.splitext(file_name)[0]
        df = read_csv_file(file)
        st.dataframe(df)
        new_data = {} 
        with st.form(key='df', clear_on_submit=True):
            # x = 0
             # Reset the dictionary on form submission

            for column in df.columns:
                value = st.text_input(f"Input Data {column}: ")
                new_data[column] = [value]  # Assign input value directly, without creating a list
                # x += 1
            submit_button = st.form_submit_button(label = "Submit")
            if submit_button:
                new_data_df = pd.DataFrame(new_data)  # Create DataFrame from the updated dictionary
                df = df.append(new_data_df, ignore_index=True)
                df.to_csv(filepath, index=False)
                st.success("Data berhasil dimasukkan ke dalam dataset.")
                st.header('After Update')
                st.dataframe(df)    

        df.dropna(inplace=True, axis=0)

        # taking only the name of website from the URLs
        # pattern = 'https?://([\w.]+)/'
        # df['Website'] = df.URLs.str.extract(pattern)
        # df.drop('URLs', axis=1, inplace=True)

        # @st.cache_data
        def analisis():
            fake_news_count = df[df.Label == 0]['source'].value_counts()
            real_news_count = df[df.Label == 1]['source'].value_counts()
            st.subheader("Distribution of Fake vs Real News")
            fig, ax = plt.subplots()
            sns.countplot(x='Label', data=df, palette=['#ffaec0', '#b8b5ff'], saturation=1, ax=ax)
            sns.despine()
            plt.xticks([0, 1], ['Fake', 'Real'])
            plt.title('Fake Vs Real news')
            st.pyplot(fig)  

            # Count of Fake and Real news by source

            # Create DataFrames for visualization
            fdf = pd.DataFrame({'source': fake_news_count.index, 'Fake': fake_news_count.values})
            rdf = pd.DataFrame({'source': real_news_count.index, 'Real': real_news_count.values})
            rf_count = pd.merge(rdf, fdf, on='source', how='outer').fillna(0)
            rf_count['Real'] = rf_count['Real'].astype(int)
            rf_count['Fake'] = rf_count['Fake'].astype(int)
            # Display rf_count DataFrame
            st.subheader('Source-wise counts of Fake and Real News')
            st.dataframe(rf_count)

            # Display top sources posting Real News
            st.subheader("Top Sources Posting Real News")
            fig, ax = plt.subplots()
            sns.barplot(y=real_news_count[:5].index, x=real_news_count[:5].values,
                        palette=['#7868e6', '#b8b5ff', '#ffaec0', 'grey', '#a7c5eb'], ax=ax)
            sns.despine(bottom=True, left=True)
            plt.title('Top source posting Real News')
            st.pyplot(fig)

            # Display top sources posting Fake News
            st.subheader("Top Sources Posting Fake News")
            fig, ax = plt.subplots()
            sns.barplot(y=fake_news_count[:5].index, x=fake_news_count[:5].values,
                        palette=['#7868e6', '#b8b5ff', '#ffaec0', 'grey'], ax=ax)
            sns.despine(bottom=True, left=True)
            plt.title('Top source posting Fake News')
            st.pyplot(fig)   

        # Function for text cleaning
        # Text cleaning function
        def clean_text(text):
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'\d+', '', text)  # Remove numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
            return text
        

        df['statement'] = df['statement'].apply(clean_text)
        # Preprocessing
        # @st.cache_data
        def data_bersih():
            # Display the processed data
            # @st.cache_data
            st.dataframe(df)
            
            # Separate fake and real news
            fake_news = df[df['Label'] == 0]
            real_news = df[df['Label'] == 1]

            sw = set(STOPWORDS)

            # Word cloud for fake news
            fake_wc = WordCloud(width=1200, height=600, background_color='white', stopwords=sw, min_font_size=10).generate(' '.join(fake_news.statement))
            st.subheader("Word Cloud for Fake News")
            plt.figure(figsize=(12, 6), facecolor=None)
            plt.imshow(fake_wc)
            plt.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot()

            # Word cloud for real news
            real_wc = WordCloud(width=1200, height=600, background_color='white', stopwords=sw, min_font_size=10).generate(' '.join(real_news.statement))
            st.subheader("Word Cloud for Real News")
            plt.figure(figsize=(12, 6), facecolor=None)
            plt.imshow(real_wc)
            plt.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot()

        X = df['statement']
        y = df['Label']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        vocab_size = len(tokenizer.word_index) + 1

        # Convert text to sequences
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        # Pad sequences to a fixed length
        max_sequence_length = 1000
        X_train_seq = pad_sequences(X_train_seq, maxlen=max_sequence_length)
        X_test_seq = pad_sequences(X_test_seq, maxlen=max_sequence_length)

        # Define the LSTM model
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Function to save the model to the database
        def save_model_to_database(model, dataset_name):
            # Save model to pickle file
            model_file = f'{dataset_name}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

            # Connect to the database
            # Membuat objek koneksi dengan mengatur waktu tunggu baca
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="projek_pp",
                connect_timeout=600,
                autocommit=True,  # Pastikan autocommit diaktifkan
                buffered=True,  # Aktifkan mode buffered
            )

            # Create a table to store models if it doesn't exist
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS models
                            (dataset_name VARCHAR(255) PRIMARY KEY,
                            model LONGBLOB)''')
            # cursor = self.conn.cursor(buffered=True)
            # Read the model file as binary
            with open(model_file, 'rb') as f:
                model_binary = f.read()

            # Save the model to the database
            cursor.execute("INSERT INTO models (dataset_name, model) VALUES (%s, %s)", (dataset_name, model_binary))
            conn.commit()

            # Close the connection and remove the model file
            
            cursor.close()
            conn.close()
            os.remove(model_file)

        # Function to load the model from the database with the same name as the dataset
        def load_model_from_database(dataset_name):
            # Connect to the database
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="projek_pp",
                connect_timeout=600,
                autocommit=True,  # Ensure autocommit is enabled
                buffered=True,  # Enable buffered mode
            )

            # Check if the model exists in the database
            cursor = conn.cursor()
            cursor.execute("SELECT model FROM models WHERE dataset_name = %s", (dataset_name,))
            row = cursor.fetchone()

            if row:
                # If the model exists, load the model from the database
                saved_model = pickle.loads(row[0])
                return saved_model

            # If the model doesn't exist, return None
            return None
        @st.cache_data
        def train_and_predict():
            saved_model = load_model_from_database(dataset_name)
            if saved_model:
                # Jika model tersedia, gunakan model yang tersimpan
                st.write("Model telah tersedia di database.")
                st.write("Gunakan model untuk melakukan prediksi.")
                
                pred = (saved_model.predict(X_test_seq) > 0.5).astype("int32")
                pred2 = (saved_model.predict(X_train_seq) > 0.5).astype("int32")

                # Display the classification report and confusion matrix
                cm = confusion_matrix(y_test, pred)
                fig, ax = plt.subplots(figsize=(2, 2))
                sns.heatmap(cm, annot=True, fmt='', cbar=False, linewidths=2, xticklabels=['Fake','Real'], yticklabels=['Fake','Real'], ax=ax)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted', color='navy', fontsize=15)
                plt.ylabel('Actual', color='navy', fontsize=15)
                st.pyplot(fig)


                report = classification_report(y_test, pred, target_names = ['Fake','Real'])
                st.text("Classification Report:")
                st.text(report)

                accuracy = round(accuracy_score(y_test, pred), 2)
                st.text(f"Accuracy Data Test: {accuracy}")

                accuracy = round(accuracy_score(y_train, pred2), 2)
                st.text(f"Accuracy Data Train: {accuracy}")

                # Tampilkan opsi untuk memuat data dan melakukan prediksi
                # Implementasikan sesuai kebutuhan aplikasi Anda
            else:
                # Jika model tidak tersedia, lakukan pemodelan
                st.write("Model belum tersedia di database.")
                st.write("Sedang melakukan pemodalan harap tunggu......")
                # Train the model
                history  = model.fit(X_train_seq, y_train, batch_size=32, epochs=10, validation_data=(X_test_seq, y_test))
                save_model_to_database(model, dataset_name)
                st.write("Model telah disimpan di database.")
                # Predict on the test set
                def plot_loss_epochs(history):
                    epochs = np.arange(1,len(history.history['accuracy']) + 1,1)
                    train_acc = history.history['accuracy']
                    train_loss = history.history['loss']
                    val_acc = history.history['val_accuracy']
                    val_loss = history.history['val_loss']

                    fig , ax = plt.subplots(1,2, figsize=(7,3))
                    ax[0].plot(epochs , train_acc , '.-' , label = 'Train Accuracy')
                    ax[0].plot(epochs , val_acc , '.-' , label = 'Validation Accuracy')
                    ax[0].set_title('Train & Validation Accuracy')
                    ax[0].legend()
                    ax[0].set_xlabel("Epochs")
                    ax[0].set_ylabel("Accuracy")

                    ax[1].plot(epochs , train_loss , '.-' , label = 'Train Loss')
                    ax[1].plot(epochs , val_loss , '.-' , label = 'Validation Loss')
                    ax[1].set_title('Train & Validation Loss')
                    ax[1].legend()
                    ax[1].set_xlabel("Epochs")
                    ax[1].set_ylabel("Loss")
                    fig.tight_layout()
                    st.pyplot(fig)

                pred = (model.predict(X_test_seq) > 0.5).astype("int32")

                # Display the classification report and confusion matrix
                cm = confusion_matrix(y_test, pred)
                fig, ax = plt.subplots(figsize=(2, 2))
                sns.heatmap(cm, annot=True, fmt='', cbar=False, linewidths=2, xticklabels=['Fake','Real'], yticklabels=['Fake','Real'], ax=ax)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted', color='navy', fontsize=15)
                plt.ylabel('Actual', color='navy', fontsize=15)
                st.pyplot(fig)

                plot_loss_epochs(history)

                report = classification_report(y_test, pred, labels=[0, 1], target_names=['Fake', 'Real'])
                st.text("Classification Report:")
                st.text(report)

                accuracy = round(accuracy_score(y_test, pred), 2)
                st.text(f"Accuracy: {accuracy}")
        # Function to predict label
        def predict_label(text):
            saved_model = load_model_from_database(dataset_name)
            if saved_model:
                # Clean and tokenize the input news article
                st.write("Model telah tersedia di database.")
                # Clean the new articles
                cleaned_new_articles = [clean_text(article) for article in text]

                # Tokenize and pad the new articles
                new_articles_seq = tokenizer.texts_to_sequences(cleaned_new_articles)
                new_articles_seq = pad_sequences(new_articles_seq, maxlen=max_sequence_length)

                # Make predictions
                predictions = saved_model.predict(new_articles_seq)

                # Convert probabilities to labels
                labels = ["Real" if pred >= 0.5 else "Fake" for pred in predictions]
                return labels
            
            else:   
                st.write("Model belum tersedia di database.")
                st.write("Sedang melakukan pemodalan harap tunggu......")
                # model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5)  # Train the model with appropriate epochs
                model.fit(X_train_seq, y_train, batch_size=32, epochs=10, validation_data=(X_test_seq, y_test))
                save_model_to_database(model, dataset_name)
                st.write("Model telah disimpan di database.")
                cleaned_new_articles = [clean_text(article) for article in text]

                # Tokenize and pad the new articles
                new_articles_seq = tokenizer.texts_to_sequences(cleaned_new_articles)
                new_articles_seq = pad_sequences(new_articles_seq, maxlen=max_sequence_length)

                # Make predictions
                predictions = model.predict(new_articles_seq)

                # Convert probabilities to labels
                labels = ["Real" if pred >= 0.5 else "Fake" for pred in predictions]
                return labels

        st.sidebar.write("---")
        st.sidebar.write("Pilih Opsi:")
        if st.sidebar.button("Data Analisis"):
            # Drop rows with missing values
            # Display count plot of labels
            analisis()

        # Check if Cleaning Text button is clicked
        if st.sidebar.button("Visualisasi WordCloud "):
            # Preprocess the text
            data_bersih()

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Check if Run Model button is clicked
        if st.sidebar.button("Laporan Model"):
            # Function to train and predict the label
            st.title("Model Training and Prediction")
            train_and_predict()

        # Main section of the app
        
        # if st.button("Prediksi Berita"):
        #     st.dataframe(df)
        st.write('---')
        st.write('Deteksi Berita')
        new_articles_input = st.text_area("New Articles", "Enter new articles here (one article per line)")
        if st.button("Deteksi"):
            if new_articles_input:
                new_articles = new_articles_input.split("\n")
                labels = predict_label(new_articles)
                for article, label in zip(new_articles, labels):
                    st.info(f"Article: {article}")
                    st.success(f"Label: {label}")
                    st.write("---")
                    # st.success(f"The news article is classified as: {label}")
            else:
                st.warning("Please enter a news article.")

                

elif sidebar_option == "Scrape Website":
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('punkt')
    def scrape_website(page_number):
        page_num = str(page_number)
        URL = f'https://www.politifact.com/factchecks/list/?page={page_num}&'
        webpage = requests.get(URL)
        soup = BeautifulSoup(webpage.text, 'html.parser')

        statement_footer = soup.find_all('footer', attrs={'class': 'm-statement__footer'})
        statement_quote = soup.find_all('div', attrs={'class': 'm-statement__quote'})
        statement_meta = soup.find_all('div', attrs={'class': 'm-statement__meta'})
        target = soup.find_all('div', attrs={'class': 'm-statement__meter'})

        authors = []
        dates = []
        statements = []
        sources = []
        targets = []

        for i in statement_footer:
            link1 = i.text.strip()
            name_and_date = link1.split()
            first_name = name_and_date[1]
            last_name = name_and_date[2]
            full_name = first_name + ' ' + last_name
            month = name_and_date[4]
            day = name_and_date[5]
            year = name_and_date[6]
            date = month + ' ' + day + ' ' + year
            dates.append(date)
            authors.append(full_name)

        for i in statement_quote:
            link2 = i.find_all('a')
            statement_text = link2[0].text.strip()
            statements.append(statement_text)

        for i in statement_meta:
            link3 = i.find_all('a')
            source_text = link3[0].text.strip()
            sources.append(source_text)

        for i in target:
            link4 = i.find('div', attrs={'class': 'c-image'}).find('img').get('alt')
            targets.append(link4)

        return authors, dates, statements, sources, targets


    def datascrape(start_page, end_page):
        authors = []
        dates = []
        statements = []
        sources = []
        targets = []

        for page_number in range(start_page, end_page + 1):
            page_authors, page_dates, page_statements, page_sources, page_targets = scrape_website(page_number)
            authors.extend(page_authors)
            dates.extend(page_dates)
            statements.extend(page_statements)
            sources.extend(page_sources)
            targets.extend(page_targets)
            time.sleep(1)

        data = pd.DataFrame({
            'author': authors,
            'statement': statements,
            'source': sources,
            'date': dates,
            'target': targets
        })

        def get_binary_num_target(text):
            if text == 'true':
                return 1
            else:
                return 0

        data['Label'] = data['target'].apply(get_binary_num_target)
        return data

    def main():
        st.sidebar.write("---")
        st.sidebar.write("Pilih Opsi:")
        start_page = st.sidebar.number_input("Enter the starting page number:", value=1, min_value=1)
        end_page = st.sidebar.number_input("Enter the ending page number:", value=1)

        # def save_data(df):
        #     # Save the dataframe to a CSV file
        #     df.to_csv(f'data page {start_page}-{end_page}.csv', index=False)
        #     st.success("Data saved successfully!")
        def save_data_to_csv(data):
            df = pd.DataFrame(data)  # Convert the data to a DataFrame
            csv = df.to_csv(index=False)  # Convert DataFrame to CSV format

            # Provide the file download link to the user
            st.download_button(
                label='Download CSV',
                data=csv,
                file_name=f'data page {start_page}-{end_page}.csv',
                mime='text/csv'
            )
        if start_page < 1 or start_page > end_page:
            st.sidebar.error("Invalid page numbers!")
            return
        
        if st.sidebar.button("Scrape"):
            st.info("Scraping in progress...")
            df = datascrape(int(start_page), int(end_page))
            st.success("Scraping completed!")
            st.write(df)
            save_data_to_csv(df)
            # st.dataframe(data, height=500)

        # if st.button("Save DataFrame"):
        #     df = datascrape(int(start_page), int(end_page))
        #     save_data_to_csv(df)

        if st.sidebar.button("Data Analisis"):
            df = datascrape(int(start_page), int(end_page))
            # @st.cache_data
            def analisis():
                df.dropna(inplace=True, axis=0)
                fake_news_count = df[df.Label == 0]['source'].value_counts()
                real_news_count = df[df.Label == 1]['source'].value_counts()
                st.subheader("Distribution of Fake vs Real News")
                fig, ax = plt.subplots()
                sns.countplot(x='Label', data=df, palette=['#ffaec0', '#b8b5ff'], saturation=1, ax=ax)
                sns.despine()
                plt.xticks([0, 1], ['Fake', 'Real'])
                plt.title('Fake Vs Real news')
                st.pyplot(fig)  

                # Count of Fake and Real news by source

                # Create DataFrames for visualization
                fdf = pd.DataFrame({'source': fake_news_count.index, 'Fake': fake_news_count.values})
                rdf = pd.DataFrame({'source': real_news_count.index, 'Real': real_news_count.values})
                rf_count = pd.merge(rdf, fdf, on='source', how='outer').fillna(0)
                rf_count['Real'] = rf_count['Real'].astype(int)
                rf_count['Fake'] = rf_count['Fake'].astype(int)
                # Display rf_count DataFrame
                st.subheader('Source-wise counts of Fake and Real News')
                st.dataframe(rf_count)

                # Display top sources posting Real News
                st.subheader("Top Sources Posting Real News")
                fig, ax = plt.subplots()
                sns.barplot(y=real_news_count[:5].index, x=real_news_count[:5].values,
                            palette=['#7868e6', '#b8b5ff', '#ffaec0', 'grey', '#a7c5eb'], ax=ax)
                sns.despine(bottom=True, left=True)
                plt.title('Top source posting Real News')
                st.pyplot(fig)

                # Display top sources posting Fake News
                st.subheader("Top Sources Posting Fake News")
                fig, ax = plt.subplots()
                sns.barplot(y=fake_news_count[:5].index, x=fake_news_count[:5].values,
                            palette=['#7868e6', '#b8b5ff', '#ffaec0', 'grey'], ax=ax)
                sns.despine(bottom=True, left=True)
                plt.title('Top source posting Fake News')
                st.pyplot(fig)
            analisis()

        if st.sidebar.button("Visualisasi WordCloud"):
            df = datascrape(int(start_page), int(end_page))    
            # Function for text cleaning
            df.dropna(inplace=True, axis=0)

            df.dropna(inplace=True, axis=0)
            def clean_text(text):
                text = text.lower()  # Convert to lowercase
                text = re.sub(r'\d+', '', text)  # Remove numbers
                text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
                text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
                return text
            df['statement'] = df['statement'].apply(clean_text)
            st.dataframe(df)
            # @st.cache(persist=True)
            
            def data_bersih():

                fake_news = df[df['Label'] == 0]
                real_news = df[df['Label'] == 1]
                sw = set(STOPWORDS)

                # Word cloud for fake news
                fake_wc = WordCloud(width=1200, height=600, background_color='white', stopwords=sw, min_font_size=10).generate(' '.join(fake_news.statement))
                st.subheader("Word Cloud for Fake News")
                plt.figure(figsize=(12, 6), facecolor=None)
                plt.imshow(fake_wc)
                plt.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot()

                # Word cloud for real news
                real_wc = WordCloud(width=1200, height=600, background_color='white', stopwords=sw, min_font_size=10).generate(' '.join(real_news.statement))
                st.subheader("Word Cloud for Real News")
                plt.figure(figsize=(12, 6), facecolor=None)
                plt.imshow(real_wc)
                plt.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot()
            data_bersih()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Check if Run Model button is clicked
        if st.sidebar.button("Laporan Model"):
            df = datascrape(int(start_page), int(end_page))

            # Streamlit App
            st.title("Fake News Classifier")

            # Function to train and predict the label

            st.title("Model Training and Prediction")
            df.dropna(inplace=True, axis=0)
            def clean_text(text):
                text = text.lower()  # Convert to lowercase
                text = re.sub(r'\d+', '', text)  # Remove numbers
                text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
                text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
                return text
            df['statement'] = df['statement'].apply(clean_text)
            # Tokenization
            X = df['statement']
            y = df['Label']

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Tokenization
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(X_train)
            vocab_size = len(tokenizer.word_index) + 1

            # Convert text to sequences
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)

            # Pad sequences to a fixed length
            max_sequence_length = 1000
            X_train_seq = pad_sequences(X_train_seq, maxlen=max_sequence_length)
            X_test_seq = pad_sequences(X_test_seq, maxlen=max_sequence_length)

            # Define the LSTM model
            model = Sequential()
            model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
            model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(X_train_seq, y_train, batch_size=32, epochs=10, validation_data=(X_test_seq, y_test))
            def plot_loss_epochs(history):
                epochs = np.arange(1,len(history.history['accuracy']) + 1,1)
                train_acc = history.history['accuracy']
                train_loss = history.history['loss']
                val_acc = history.history['val_accuracy']
                val_loss = history.history['val_loss']

                fig , ax = plt.subplots(1,2, figsize=(7,3))
                ax[0].plot(epochs , train_acc , '.-' , label = 'Train Accuracy')
                ax[0].plot(epochs , val_acc , '.-' , label = 'Validation Accuracy')
                ax[0].set_title('Train & Validation Accuracy')
                ax[0].legend()
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("Accuracy")

                ax[1].plot(epochs , train_loss , '.-' , label = 'Train Loss')
                ax[1].plot(epochs , val_loss , '.-' , label = 'Validation Loss')
                ax[1].set_title('Train & Validation Loss')
                ax[1].legend()
                ax[1].set_xlabel("Epochs")
                ax[1].set_ylabel("Loss")
                fig.tight_layout()
                st.pyplot(fig)

            
            pred = (model.predict(X_test_seq) > 0.5).astype("int32")
            pred2 = (model.predict(X_train_seq) > 0.5).astype("int32")
            # Display the classification report and confusion matrix
            cm = confusion_matrix(y_test, pred)
            fig, ax = plt.subplots(figsize=(2, 2))
            sns.heatmap(cm, annot=True, fmt='', cbar=False, linewidths=2, xticklabels=['Fake','Real'], yticklabels=['Fake','Real'], ax=ax)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted', color='navy', fontsize=15)
            plt.ylabel('Actual', color='navy', fontsize=15)
            st.pyplot(fig)

            plot_loss_epochs(history)

            report = classification_report(y_test, pred, labels=[0, 1], target_names=['Fake', 'Real'])
            st.text("Classification Report:")
            st.text(report)

            accuracy = round(accuracy_score(y_test, pred), 2)
            st.text(f"Accuracy Data Train: {accuracy}")

            accuracy = round(accuracy_score(y_train, pred2), 2)
            st.text(f"Accuracy Data Train: {accuracy}")

        st.write('---')
        st.write('Deteksi Berita')
        new_articles_input = st.text_area("New Articles", "Enter new articles here (one article per line)")
        if st.button("Deteksi"):
            df = datascrape(int(start_page), int(end_page))  
            # Function for text cleaning
            # @st.cache_data
            def clean_text(text):
                text = text.lower()  # Convert to lowercase
                text = re.sub(r'\d+', '', text)  # Remove numbers
                text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
                text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
                return text
            # @st.cache_data
            def modeling():
                df['statement'] = df['statement'].apply(clean_text)
                # Tokenization
                X = df['statement']
                y = df['Label']

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Tokenization
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts(X_train)
                vocab_size = len(tokenizer.word_index) + 1

                # Convert text to sequences
                X_train_seq = tokenizer.texts_to_sequences(X_train)
                X_test_seq = tokenizer.texts_to_sequences(X_test)

                # Pad sequences to a fixed length
                max_sequence_length = 1000
                X_train_seq = pad_sequences(X_train_seq, maxlen=max_sequence_length)
                X_test_seq = pad_sequences(X_test_seq, maxlen=max_sequence_length)

                # Define the LSTM model
                model = Sequential()
                model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
                model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
                model.add(Dense(1, activation='sigmoid'))

                # Compile the model
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train_seq, y_train, batch_size=32, epochs=10, validation_data=(X_test_seq, y_test))
                return model, tokenizer, max_sequence_length
            # @st.cache_data
            def predict_label(text):
                model, tokenizer, max_sequence_length = modeling()
                # save_model_to_database(model, dataset_name)
                st.write("Model telah disimpan di database.")
                cleaned_new_articles = [clean_text(article) for article in text]

                # Tokenize and pad the new articles
                new_articles_seq = tokenizer.texts_to_sequences(cleaned_new_articles)
                new_articles_seq = pad_sequences(new_articles_seq, maxlen=max_sequence_length)

                # Make predictions
                predictions = model.predict(new_articles_seq)

                # Convert probabilities to labels
                labels = ["Real" if pred >= 0.5 else "Fake" for pred in predictions]
                return labels

            if new_articles_input:
                new_articles = new_articles_input.split("\n")
                labels = predict_label(new_articles)
                for article, label in zip(new_articles, labels):
                    st.write(f"Article: {article}")
                    st.success(f"Label: {label}")
                    st.write("---")
                    # st.success(f"The news article is classified as: {label}")
            else:
                st.warning("Please enter a news article.")

    if __name__ == '__main__':
        main()
