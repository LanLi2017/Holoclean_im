# use tf-idf to evaluate how important a word is to a document
# concatenate the values in the attribute
import codecs
import math
import re
from collections import Counter
from operator import itemgetter
from pprint import pprint
import numpy as np

from gensim.models import Phrases, FastText
from gensim.models.phrases import Phraser
from gensim.parsing import remove_stopwords, strip_punctuation, strip_non_alphanum
from nltk.corpus import stopwords
import spacy
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
from spacy import displacy
from pandas import read_csv
from tqdm import tqdm


class Tfidf_Embed:
    def __init__(self, col: list):
        """
        fastText and tf-idf
        """
        self.emb_size = 10
        self.nlp = en_core_web_sm.load()
        self.col = col
        self.text = self.get_tokens()
        self.tokens = self.stick_terms()
        self.model = self.train_fastText()
        self.tfidf_sent_vectors = self.tfidf_w2v()

    def get_tokens(self):
        # create a list of documents with a list of tokenized words inside
        # col: list of rows
        # return: list of tokenized rows
        text = []
        for i in self.col:
            doc = self.nlp(remove_stopwords(strip_punctuation(strip_non_alphanum(str(i).lower()))))
            tokens = [token.text for token in doc]
            text.append(tokens)
        return text

    def stick_terms(self):
        # create relevant phrases from the list of sentences
        common_terms = ['of', 'with', 'without', 'and', 'or', 'the', 'a']
        phrases = Phrases(self.text, common_terms=common_terms, threshold=10, min_count=5)
        bigram = Phraser(phrases)
        # applying phraser to transform tokenized rows
        tokens = list(bigram[self.text])
        return tokens

    def train_fastText(self):
        # train fastText
        model = FastText(self.tokens, size=self.emb_size,min_count=1)
        return model

    def tfidf(self):
        text = []
        for i in tqdm(self.tokens):
            string = ' '.join(i)
            text.append(string)
        tf_idf_vec = TfidfVectorizer()
        final_tf_idf = tf_idf_vec.fit_transform(text)
        tfidf_feat = tf_idf_vec.get_feature_names()
        return final_tf_idf, tfidf_feat

    def tfidf_w2v(self):
        tfidf_sent_vectors = []  # the tfidf-w2v for each sentences in the column
        row = 0
        errors = 0
        final_tf_idf, tfidf_feat = self.tfidf()
        for sent in tqdm(self.tokens):
            sent_vec = np.zeros(self.emb_size)
            weight_sum = 0
            for word in sent:
                try:
                    vec = self.model.wv[word]
                    # obtain the tf_idf of a word in a sentence
                    tfidf = final_tf_idf[row, tfidf_feat.index(word)]
                    sent_vec += (vec * tfidf)
                    weight_sum += tfidf
                except:
                    errors += 1
                    pass
            sent_vec /= weight_sum
            tfidf_sent_vectors.append(sent_vec)
            row += 1
        return tfidf_sent_vectors


def main():
    data = read_csv('data/hospital.csv')
    attr_name = 'MeasureName'
    c = Tfidf_Embed(data[attr_name])
    pprint(c.tfidf_sent_vectors)
    # tfidf_vec = c.tfidf_sent_vectors
    # pprint(tfidf_vec)


if __name__ == '__main__':
    main()
