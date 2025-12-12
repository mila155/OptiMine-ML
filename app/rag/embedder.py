from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts):
        if texts:
            self.vectorizer.fit(texts)
        else:
            # fallback: fit dummy text
            self.vectorizer.fit(["dummy"])

    def encode(self, texts):
        if not texts:
            texts = ["dummy"]
        return self.vectorizer.transform(texts).toarray()