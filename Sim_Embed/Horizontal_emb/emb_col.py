# use tf-idf to evaluate how important a word is to a document
# concatenate the values in the attribute
import codecs
import math
import re
from collections import Counter
from operator import itemgetter
from pprint import pprint
import numpy as np
import fasttext

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

from Sim_Embed.cal_similarity import pearson_corr, cos_simi


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
        # how to deal with nan?
        text = []
        for i in tqdm(self.tokens):
            string = ' '.join(i)
            text.append(string)
        tf_idf_vec = TfidfVectorizer()
        final_tf_idf = tf_idf_vec.fit_transform(text) # final_tf_idf is the sparse matrix with row=sentence, col=word and cell_val =tfidf
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


def remove_nan(embed_vec:list):
    # return the nan idx, and remove it jointly from two vectors
    nan_list = []
    for i,e in enumerate(embed_vec):
        for j in e:
            if math.isnan(j):
                nan_list.append(i)
    idx_nan = list(set(nan_list))
    return idx_nan


def cos_emb_vec(data, attr1, attr2, embed_size):
    '''
    data: csv file & pandas dataframe
    attr1: long sequence column
    attr2: short words column
    embed size: normally 10
    '''
    # MeasureName & Condition: [[0.25519639]]
    attr_name_1 = attr1
    # long sequence: tf-idf weighted word embedding
    embed1 = Tfidf_Embed(data[attr_name_1]).tfidf_sent_vectors

    # short word embedding
    attr_name_2 = attr2
    model_attr4 = Tfidf_Embed(data[attr_name_2]).train_fastText()
    w2v = []
    for word in data[attr_name_2]:
        try:
            vec = model_attr4.wv[word]
        except:
            vec = [math.nan] * embed_size
        w2v.append(vec)

    # get nan index
    idx_nan_1 = remove_nan(embed1)    # nan from long sequence
    idx_nan_2 = remove_nan(w2v)       # nan from short sequence
    join_nan = idx_nan_1 + list(set(idx_nan_2) - set(idx_nan_1))  # remove nan from both sides
    embed1_unan = [item for idx, item in enumerate(embed1) if idx not in set(join_nan)]
    w2v_unan = [item for idx, item in enumerate(w2v) if idx not in set(join_nan)]
    res1 = remove_nan(embed1_unan)
    res2 = remove_nan(w2v_unan)
    # should be none of nan after removing
    assert res1 == []
    assert res2 == []

    embed1_ary = np.array(embed1_unan)
    np_w2v = np.array(w2v_unan)
    assert embed1_ary.shape == np_w2v.shape

    # capture core info from long sequence column, weighted tf-idf word embedding
    # FastText embedding word into vectors
    # calculate the similarity using cosine similarity
    cos_res = []
    error = 0
    try:
        cos_res = cos_simi(embed1_ary, np_w2v)
        print(cos_res)
    except:
        error += 1
        pass
    assert error == 0
    return cos_res


def langmodel(data, attr, emb_size):
    model = Tfidf_Embed(data[attr]).train_fastText()
    w2v = []
    # wv: This object essentially contains the mapping between words and embeddings
    for word in data[attr]:
        try:
            vec = model.wv[word]
        except:
            vec = [math.nan] * emb_size
        w2v.append(vec)
    return w2v


def main():
    data = read_csv('../data/hospital.csv')
    colname_list = list(data.columns.values)
    print(colname_list)
    vec_1 = langmodel(data, 'MeasureName', 10)
    vec_tfidf = Tfidf_Embed(data['MeasureName']).tfidf_sent_vectors


    # vec_2 = langmodel(data, 'Condition', 10)
    idx_nan_1 = remove_nan(vec_1)  # nan from long sequence
    idx_nan_2 = remove_nan(vec_tfidf)  # nan from short sequence
    join_nan = idx_nan_1 + list(set(idx_nan_2) - set(idx_nan_1))  # remove nan from both sides
    vec_1_unan = [item for idx, item in enumerate(vec_1) if idx not in set(join_nan)]
    vec_2_unan = [item for idx, item in enumerate(vec_tfidf) if idx not in set(join_nan)]
    res1 = remove_nan(vec_1_unan)
    res2 = remove_nan(vec_2_unan)
    # # should be none of nan after removing
    assert res1 == []
    assert res2 == []

    vec_1_ary = np.array(vec_1_unan)
    vec_2_ary = np.array(vec_2_unan)
    assert vec_1_ary.shape == vec_2_ary.shape

    cos_res = []
    error = 0
    try:
        cos_res = cos_simi(vec_1_ary, vec_2_ary)
        print(cos_res)
    except:
        error += 1
        pass
    assert error == 0

    # cos1 = cos_emb_vec(data,'MeasureName', 'Condition', 10)
    #
    # cos2 = cos_emb_vec(data, 'MeasureName', 'City', 10)
    # assert cos2 < cos1

    # cos3 = cos_emb_vec(data, 'MeasureName', 'HospitalOwner', 10)
    # test_col = ['HospitalName','Address1','HospitalOwner','City','State','CountyName','EmergencyService','Condition']
    # cos_res = []
    # for col in test_col:
    #     cos_res.append(cos_emb_vec(data, 'MeasureName', col, 10))
    # pprint(cos_res)
    # for att, cos in zip(test_col, cos_res):
    #     print(f'{att} and "MeasureName" similarity is ==> {cos}')


if __name__ == '__main__':
    main()
