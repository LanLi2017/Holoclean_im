import math
from collections import Counter
from operator import itemgetter

from Sim_Embed.weighted_emb import TfIdf


def computeTF(self, curr_row):
    # current doc: sentences/words at current row
    tfDict = {}
    tokens = self.get_tokens(self.data[curr_row])
    print(f'term frequency for A is : {tokens}')
    tokensCount = len(tokens)
    set_tokens = set(tokens)
    dict_tokens = dict.fromkeys(set_tokens, 0)
    for word in tokens:
        dict_tokens[word] += 1

    for word, count in dict_tokens.items():
        tfDict[word] = count / float(tokensCount)
    pprint(tfDict)
    return tfDict


def computeIDF(self):
    docslist = []
    for row in self.data:
        tokenize_row = self.get_tokens(row)
        set_row = set(tokenize_row)
        dict_row = dict.fromkeys(set_row, 0)
        for word in dict_row:
            dict_row[word] += 1
        docslist.append(dict_row)
    print(f'now the doc list is : {docslist}')

    N = len(docslist)  # length of document
    idfDict = Counter()
    for doc in docslist:
        for word, val in doc.items():
            if word in self.stopwords:
                idfDict[word] = 0
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        if word in self.stopwords:
            idfDict[word] = 0
        else:
            idfDict[word] = math.log10(N / float(val))
    return idfDict


def computeTFIDF(self, curr_row):
    """ concatenate into one document
    """
    tfidf = {}
    idfs = self.computeIDF()
    tf_currow = self.computeTF(curr_row)
    for word, val in tf_currow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def get_doc_keywords(self, curr_row):
    """Retrieve terms and corresponding tf-idf for the specified document.
       The returned terms are ordered by decreasing tf-idf.
    """
    tfidf = self.computeTFIDF(curr_row)

    return sorted(tfidf.items(), key=itemgetter(1), reverse=True)


docA = "The cat sat on my face"
docB = "The dog sat on my bed"
tfidfA = TfIdf([docA,docB])
compute_keywords = tfidfA.get_doc_keywords(0)
print(compute_keywords)