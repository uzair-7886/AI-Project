import pickle

with open('lda_result.pk', 'rb') as file:
        lda_topic_words = pickle.load(file)

print(lda_topic_words)