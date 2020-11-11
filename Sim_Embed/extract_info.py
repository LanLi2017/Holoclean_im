import csv
from pprint import pprint

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from Sim_Embed.weighted_emb import Computation
import numpy as np
import pandas as pd


def read_csv(file_p):
    # file_p: file path
    # with open(file_p, 'rt')as f:
    #     reader = csv.DictReader(f)
    #     data = list(reader)
    data = pd.read_csv(file_p,index_col=False)
    return data


def tfidf(data):
    # TF-IDF weighted Word2vec
    model = TfidfVectorizer()
    attribute = 'MeasureName'
    col_values = data[attribute]
    pprint(col_values)
    # computation = Computation(data)
    tokenize_col = []
    for v in col_values:
        print(v)
        tokenize_col.append(v.split())
    pprint(tokenize_col)
    model.fit(col_values)
    # converting a dictionary with word as a key, and the idf as a value
    dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))

    # word2vec min_count = 5 considers only words that occurred at least 5 times
    w2v_model = Word2Vec(tokenize_col, min_count=5, size=50, workers=4)
    w2v_words = list(w2v_model.wv.vocab)


    # TF-IDF weighted Word2Vec
    tfidf_feat = model.get_feature_names() # tfidf words/col-names
    # final_tf_idf is the sparse matrix with row=sentence, col=word and cell_val = tfidf

    tfidf_sent_vectors  = []
    row = 0
    for sent in tqdm(tokenize_col):
        sent_vec = np.zeros(50)
        weight_sum = 0
        for word in sent:
            if word in w2v_words and word in tfidf_feat:
                vec = w2v_model.wv[word]
                tf_idf = dictionary[word] * (sent.count(word)/len(sent))
                sent_vec += (vec * tf_idf)
                weight_sum += tf_idf
        if weight_sum != 0:
            sent_vec /= weight_sum
        tfidf_sent_vectors.append(sent_vec)
        row += 1
    pprint(tfidf_sent_vectors)
    return tfidf_sent_vectors


def main():
    data = read_csv('data/hospital.csv')
    # comment = []
    # for row in data:
    #     comment.append(row['MeasureName'])

    #
    # print(comment[0])
    tfidf(data)


if __name__ == '__main__':
    main()