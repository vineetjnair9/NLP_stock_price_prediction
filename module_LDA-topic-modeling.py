# Source: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

import pandas as pd
import gensim
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import PorterStemmer
from nltk.stem.porter import *
np.random.seed(2018)
from gensim import corpora, models

import nltk
nltk.download('wordnet')

#%% Load dataset
data = pd.read_csv(r"C:\Users\vinee\OneDrive - Massachusetts Institute of Technology\MIT\Fall 2020\6.867\Project\amzn_text_embs_per_art.csv",usecols = ["title","text"])
data['index'] = data.index

#%% Pre-processing text
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#%% Preview sample article text after preprocessing

stemmer = SnowballStemmer("english")

doc_sample = data.iloc[0][1]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#%% Preprocess the text of all the articles
processed_articles = data['text'].map(preprocess)
processed_articles.head

#%% Preprocess all titles
processed_titles = data['title'].map(preprocess)
processed_titles.head


#%% TEXT
# Bag of words on the data set - Dictionary of 10 most common words
dictionary = gensim.corpora.Dictionary(processed_articles)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

#%% Filter out tokens
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)        

#%% Gensim doc2bow
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_articles]

#%% TF-IDF
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

#%% Running LDA using BOW
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=4)

#%% Words occuring in each topic and their relative weights
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

#%% Running LDA using TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)

#%%
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
#%% Performance evaluation by classifying sample document using LDA Bag of Words model
train_index = 0
processed_articles[0]

for index, score in sorted(lda_model[bow_corpus[train_index]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
    
#%% Performance evaluation by classifying sample document using LDA TF-IDF model.    
for index, score in sorted(lda_model_tfidf[bow_corpus[train_index]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
    
#%% Testing model on unseen document
unseen_document = 'Amazon sees record Cyber Monday sales'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    
    
    