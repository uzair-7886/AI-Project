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
from wordcloud import WordCloud  # Import WordCloud
import matplotlib.pyplot as plt  # Import matplotlib for visualization
from io import BytesIO
import base64





app = Flask(__name__)

# Load your LDA model using pickle
with open('lda_model.pkl', 'rb') as file:
    lda_model = pickle.load(file)


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
    

    # Transform the new headline using the existing vocabulary
    new_headline_dtm = vectorizer.transform([new_headline])

    # Extract the common features between the new headline DTM and the training DTM
    common_features = np.intersect1d(filtered_vocabulary, vectorizer.get_feature_names_out())

    # Reorder the features to match the training feature order
    new_headline_dtm = new_headline_dtm[:, np.searchsorted(vectorizer.get_feature_names_out(), common_features)]

    # Now, you can use this new_headline_dtm for LDA prediction


    # Obtain the topic distribution for the new headline
    lda_topic_distribution = lda_model.transform(new_headline_dtm)

    # Identify the dominant topic
    dominant_topic = np.argmax(lda_topic_distribution) + 1 
    top_words = lda_topic_words[dominant_topic - 1]
    return dominant_topic, top_words

def generate_wordcloud(top_words):
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(top_words))

    # Plot the WordCloud image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save the WordCloud image to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Encode the image to base64 for embedding in HTML
    encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    
    return f'data:image/png;base64,{encoded_image}'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        headline = request.form['headline']
        preprocessed_headline = preprocess(headline)
        print(preprocessed_headline)
        dominant_topic, top_words = predict_topic(preprocessed_headline)

        wordcloud_image = generate_wordcloud(top_words)

        return render_template('index.html', headline=headline, topic=dominant_topic, top_words=top_words, wordcloud_image=wordcloud_image)

    return render_template('index.html', headline=None, topic=None, top_words=None, wordcloud_image=None)


if __name__ == '__main__':
    app.run(debug=True)
