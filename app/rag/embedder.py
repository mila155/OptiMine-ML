from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def encode(self, texts):
        return self.vectorizer.transform(texts).toarray()