import math
from collections import namedtuple

import pandas as pd
from gensim.models import FastText

df = pd.read_csv('../data/hospital.csv')


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
    embed1 = FastText(data[attr_name_1], size=embed_size,min_count=1)

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


def col_():
    # Initialize some documents
    doc1 = {'Science': 0.8, 'History': 0.05, 'Politics': 0.15, 'Sports': 0.1}
    doc2 = {'News': 0.2, 'Art': 0.8, 'Politics': 0.1, 'Sports': 0.1}
    doc3 = {'Science': 0.8, 'History': 0.1, 'Politics': 0.05, 'News': 0.1}
    doc4 = {'Science': 0.1, 'Weather': 0.2, 'Art': 0.7, 'Sports': 0.1}
    collection = [doc1, doc2, doc3, doc4]
    df = pd.DataFrame(collection)
    # Fill missing values with zeros
    df.fillna(0, inplace=True)
    # Get Feature Vectors
    feature_matrix = df.as_matrix()

    # Fit DBSCAN
    db = DBSCAN(min_samples=1, metric='precomputed').fit(pairwise_distances(feature_matrix, metric='cosine'))
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

    # Find the representatives
    representatives = {}
    for label in set(labels):
        # Find indices of documents belonging to the same cluster
        ind = np.argwhere(labels == label).reshape(-1, )
        # Select these specific documetns
        cluster_samples = feature_matrix[ind, :]
        # Calculate their centroid as an average
        centroid = np.average(cluster_samples, axis=0)
        # Find the distance of each document from the centroid
        distances = [cosine(sample_doc, centroid) for sample_doc in cluster_samples]
        # Keep the document closest to the centroid as the representative
        representatives[label] = cluster_samples[np.argsort(distances), :][0]

    for label, doc in representatives.iteritems():
        print("Label : %d -- Representative : %s" % (label, str(doc)))


def model(row, attributes, emb_size=10):
    attr_language_model = {}
    for attribute in attributes:
        attr_corpus = list(zip(row[attribute].tolist()))
        model = FastText(attr_corpus, min_count=1, size=emb_size)  #
        attr_language_model[attribute] = model
    return attr_language_model


def emb(row):

    pass


Hos = namedtuple('Hospital',
                   ['ProviderNumber','HospitalName','Address1',
                    'City','State','ZipCode','HospitalType',
                    'HospitalOwner','Condition','MeasureName'
])
row = [10018,'callahan eye foundation hospital','1720 university blvd',
       'birmingham','al',35233,'acute care hospitals','voluntary non-profit - private',
       'surgical infection prevention',
       'surgery patients who were taking heart drugs caxxed beta bxockers '
       'before coming to the hospitax who were kept on the beta bxockers during the period just before and after their surgery'
]
Hos._make(row)
