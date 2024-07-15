import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def vectorize_and_train(data):
    
    df = pd.read_csv(data, encoding='ISO-8859-1')
 
    # Extract the preprocessed text and labels
    text = df['Lemmatized_article']
    labels = df['Sentiment']

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Train the model using classification
    models = {
        'logistic_regression' : LogisticRegression(),
        'svm' : SVC(),
        'random_forest': RandomForestClassifier(),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predicting the model
        y_pred = model.predict(X_test)
        report = open(f'reports/{name}_report.txt','w')
        report.write(f"Classification Report for {name}:\n")
        report.write(classification_report(y_test, y_pred))
        report.write('='*60)


if __name__ == '__main__':
    vectorize_and_train('data/processed_data.csv')