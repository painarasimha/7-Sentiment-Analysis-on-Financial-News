# Importing Library
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import joblib

# Preprocess text
def preprocess_text(text):
    
    # Load data
    text_data = pd.read_csv(text, encoding='ISO-8859-1')
    #print(text_data.head(3))

    lemmatized_article = []

    for i in range(len(text_data)):
        article = text_data['News_article'][i]
        article = article.lower()
        #print(article)

        # Tokenize text
        words = word_tokenize(article)
        sentences = sent_tokenize(article)
        #print(words)
        #print(sentences)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        root_word = [lemmatizer.lemmatize(word) for word in words]

        # Joining the words back to sentences
        lemmatized_text = ' '.join(root_word)
        lemmatized_article.append(lemmatized_text)

    text_data['Lemmatized_article'] = lemmatized_article
    #print(text_data.head(5))

    text_data.to_csv('data/processed_data.csv', index=False)

if __name__ == '__main__':
    preprocess_text('data/all-data.csv')