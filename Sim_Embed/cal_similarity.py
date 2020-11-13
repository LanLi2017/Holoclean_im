# pearson's correlation
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score


def pearson_corr(col_1, col_2):
    # summarize the strength of the linear relationship between data samples
    # input two embedding matrix
    corr,_ = pearsonr(col_1,col_2)
    return corr


# spearson
def Spearson(col_1, col_2):
    # non-parametric statistic
    corr, _ = spearmanr(col_1, col_2)
    return corr


# cosine similarity
def cos_simi(col_1, col_2):
    # each word has its own axis, the cosine similarity then determines how similar the documents are
    cos_sim = cosine_similarity(col_1.reshape(1,-1), col_2.reshape(1,-1))
    return cos_sim


# jaccard similarity
def jacc_simi(col_1, col_2):
    # compare two binary vectors (sets
    # ) Jaccard similarity turns out to be useful by detecting duplciates
    jacc = jaccard_similarity_score(col_1, col_2)
    return jacc
