from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from concurrent.futures import ThreadPoolExecutor
import string


app = Flask(__name__)

# Load your LDA model using pickle
with open('lda_model.pkl', 'rb') as file:
    lda_model = pickle.load(file)


with open('lda_result.pk', 'rb') as file:
    all_lda_topic_distributions = pickle.load(file)

with open('all_headlines.txt', 'r', encoding='utf-8') as file:
    all_headlines = [line.strip() for line in file]

def preprocess(headline):
    # Lowercase and strip
    headline = headline.lower().strip()

    # Tokenize the headline
    tokens = word_tokenize(headline)

    # Remove stopwords
    nltk.download('stopwords')
    snowball_stemmer = SnowballStemmer("english")
    stop_words_nltk = set(stopwords.words('english'))
    stop_words_custom = list(stop_words_nltk.union(set(string.punctuation)))
    Vectorizer = CountVectorizer(stop_words=stop_words_custom)
    tokens = [token for token in tokens if token not in stop_words_custom]

    # Part-of-speech tagging
    pos_tags = pos_tag(tokens)

    # Only include nouns and verbs
    filtered_tokens = [word for word, pos in pos_tags if pos.startswith('N') or pos.startswith('V')]

    # Join the filtered tokens back into a string
    preprocessed_headline = ' '.join(filtered_tokens)

    return preprocessed_headline

# Function to predict the topic
def predict_topic(new_headline):
    with open('filtered_vocabulary.pkl', 'rb') as file:
        filtered_vocabulary = pickle.load(file)
        
    vectorizer = CountVectorizer(vocabulary=filtered_vocabulary)
    lda_topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_word_indices = np.argsort(topic)[::-1][:10]
        top_words = [vectorizer.get_feature_names_out()[index] for index in top_word_indices]
        lda_topic_words.append(top_words)
    
    new_headline_dtm = vectorizer.transform([new_headline])
    common_features = np.intersect1d(filtered_vocabulary, vectorizer.get_feature_names_out())
    new_headline_dtm = new_headline_dtm[:, np.searchsorted(vectorizer.get_feature_names_out(), common_features)]
    lda_topic_distribution = lda_model.transform(new_headline_dtm)
    dominant_topic = np.argmax(lda_topic_distribution) + 1 
    top_words = lda_topic_words[dominant_topic - 1]
    return dominant_topic, top_words


def get_related_headlines(dominant_topic, all_headlines, all_lda_topic_distributions):
    related_indices = [i for i, dist in enumerate(all_lda_topic_distributions) if np.argmax(dist) + 1 == dominant_topic]
    related_headlines = [all_headlines[i] for i in related_indices]
    return related_headlines

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        headline = request.form['headline']
        preprocessed_headline = preprocess(headline)
        # print(preprocessed_headline)
        dominant_topic, top_words = predict_topic(preprocessed_headline)

        # Get related headlines
        related_headlines = get_related_headlines(dominant_topic, all_headlines, all_lda_topic_distributions)
        print(related_headlines)
        # for head in related_headlines:
        #     print(head)
        # print(headline)

        return render_template('index.html', headline=headline, topic=dominant_topic, related_headlines=related_headlines, top_words=top_words)

    return render_template('index.html', headline=None, topic=None, related_headlines=None, top_words=None)

if __name__ == '__main__':
    app.run(debug=True)
