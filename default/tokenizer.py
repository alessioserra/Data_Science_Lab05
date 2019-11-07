from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, documents):
        lemmas = []
        for t in word_tokenize(documents):
            t = t.strip()
            lemma = self.lemmatizer.lemmatize(t)
            lemmas.append(lemma)
        return lemmas
