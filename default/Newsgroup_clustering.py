import os
from default.tokenizer import LemmaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as sw
import csv
import string
from sklearn.cluster.k_means_ import MiniBatchKMeans
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

file_list = os.listdir("T-newsgroups")
dataset = {}
translation = str.maketrans(' ',' ',string.punctuation + string.digits)

for file in file_list:
    with open('T-newsgroups\\'+str(file)) as f:
        text = f.read()
        out = text.translate(translation)
        dataset[file] = out

#TF-IDF
lemmaTokenizer = LemmaTokenizer()
stop_words = sw.words('english')                                   
stop_words.extend(["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would'])
vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, max_df = 0.08, stop_words=stop_words)
tfidf_X = vectorizer.fit_transform(dataset.values())

#Clustering TF-IDF ( MiniBatchKMEANS n=4 best for now)
km = MiniBatchKMeans(n_clusters=4, init_size=1024, batch_size=2048, random_state=20).fit(tfidf_X)
assignments = km.predict(vectorizer.transform(dataset.values()))
clusters = MiniBatchKMeans(n_clusters=4, init_size=1024, batch_size=2048, random_state=20).fit_predict(tfidf_X)


def dump_to_file(filename, assignments, dataset):
    with open(filename, mode="w", newline="") as csvfile:
        
        # Headers
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ids, cluster in zip(dataset.keys(), assignments):
            writer.writerow({'Id': str(ids), 'Predicted': str(cluster)})

dump_to_file("result.csv", assignments, dataset)
print("Computed Finished")


"""
Get top words
"""
def get_top_keywords(data, clusters, labels, n_terms):
    
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    stopwords = set(STOPWORDS)

    for i, r in df.iterrows():
        
        comment_words = ""
        
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        
        words = [labels[t] for t in np.argsort(r)[-n_terms:]]
        
        for word in words:
            comment_words = comment_words + word + ' '
        
        wordcloud = WordCloud(width = 800, height = 800, 
        background_color ='white', 
        stopwords = stopwords, 
        min_font_size = 10).generate(comment_words) 
                
        # plot the WordCloud image                        
        plt.figure(figsize = (6, 6), facecolor = None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
  
        plt.show() 

get_top_keywords(tfidf_X, clusters, vectorizer.get_feature_names(), 20)
    