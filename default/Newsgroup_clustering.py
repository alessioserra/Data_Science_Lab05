import os
from default.tokenizer import LemmaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as sw
import csv
import string
from sklearn.cluster.k_means_ import MiniBatchKMeans

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
    