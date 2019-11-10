import os
from default.tokenizer import LemmaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as sw
import csv
import string
from sklearn.cluster.k_means_ import MiniBatchKMeans
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

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
stop_words.extend(["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa',
                   'wo', 'would', 'maynardramseycslaurentianca', 'henryzootorontoedu', 'u', 'right', 'dont',
                   'gebcadredslpittedu', 'n3jxp', 'bd', 'pk115050wvnvmswvnetedu', 'nntppostinghost',
                   'dyerursamajorspdcccom'
                   , 'imaharvardrayssdlinusm2cspdccdyer', 'coegalonlarcnasagov', 'fulkcsrochesteredu', 'hst', 'pm', 'batf',
                   'atf', 'mcovingtaiugaedu', 'n4tmi', 'warpedcsmontanaedu', 'ca', 'im', 'acad3alaskaedu',
                   'nsmcaacad3alaskaedu', 'jdwarnerjournalismindianaedublue', 'acadalaskaedu',
                    'imaharvardrayssdlinusmcspdccdyer', 'irvineuxhcsouiucedu', 'aejdcmuvmbitnet'
                      ,'njxp', 'nsmcaacadalaskaedu', 'ntmi', 'pkwvnvmswvnetedu', 'one', 'cdtrocketswstratuscom',
                   'cdtvosstratuscom','went', 'individu', 'true', 'apr', 'sound', 'posit', 'looking',
                    'net', 'info', 'mind', 'jack', 'address', 'radio', 'word', 'looking', 'cold',
                   'gari', 'sender', 'situat', 'guess', 'coupl', 'yet', 'sorri', 'yes', 'hour', 'especially',
                   'tom', 'april', 'certainli', 'close', 'robert', 'msg', 'mail', 'sender', 'write', 'move',
                   'scott', 'night', 'michael', 'expect', 'rule', 'accessdigexnet', 'far','abov', 'ani', 'becaus',
                    'befor', 'dure', 'imaharvardrayssdlinusm2cspdccdy', 'imaharvardrayssdlinusmcspdccdy',
                   'jdwarnerjournalismindianaedublu', 'onc', 'onli', 'ourselv', 'themselv', 'thi', 'veri', 'whi',
                   'yourselv', 'becau', 'imaharvardrayssdlinusm2cspdccdi', 'imaharvardrayssdlinusmcspdccdi',
                   'utzoohenry', 'name', 'giant', 'man', 'medium', 'told', 'taking', 'month', 'ago', 'friend', 'small'
                   'mailing', 'pretty', 'young', 'prbaccessdigexcom', 'message', 'toronto', 'rocketswstratuscom',
                   'aprkelvinjplnasagov', 'recently', 'likely', 'perhaps', 'special', 'old', 'normal', 'thank'
                   'rocketswstratuscom', 'common', 'usually', 'mailing', 'small', 'certainly', 'near',
                    'sorry', 'wasnt', 'anybody', 'looking', 'jrmgnvifasufledu'])

vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, min_df = 0.01, max_df = 0.08, stop_words=stop_words)
tfidf_X = vectorizer.fit_transform(dataset.values())

n_components = 80
# SVD
print("Reducing dimensions..")
svd = TruncatedSVD(n_components = n_components,random_state=42)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd,normalizer)
tfidf_X = lsa.fit_transform(tfidf_X)

#Clustering TF-IDF ( MiniBatchKMEANS n=4 best for now)
model = MiniBatchKMeans(n_clusters=4, init_size=1024, batch_size=2048, random_state=20)
model.fit(tfidf_X)
assignments = model.predict(lsa.transform(vectorizer.transform(dataset.values())))
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
Create WorkCloud
"""
def get_top_keywords(data, clusters, labels, n_terms, stop_words):
    
    try:
        df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    except:
        df = pd.DataFrame(data).groupby(clusters).mean()

    for i, r in df.iterrows():
        
        comment_words = ""
        
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        
        wordList = [labels[t] for t in np.argsort(r)[-50:]]
        words = []
        for i in range(len(wordList)):
            words.append(wordList[len(wordList)-1-i])

        for word in words:
            comment_words = comment_words + word + ' '
        
        wordcloud = WordCloud(width = 800, height = 800, 
        background_color ='white', 
        stopwords = stop_words, 
        min_font_size = 10).generate(comment_words) 
                
        # plot the WordCloud image                        
        plt.figure(figsize = (5, 5), facecolor = None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
  
        plt.show()

get_top_keywords(vectorizer.transform(dataset.values()), clusters, vectorizer.get_feature_names(),25 , stop_words)
    