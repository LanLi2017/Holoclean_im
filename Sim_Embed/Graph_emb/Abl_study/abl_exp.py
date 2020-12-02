# Without random walk; simply tuple embedding
# compare embedding quality
import pandas as pd

from Sim_Embed.Graph_emb.light_imp import multi_words
from embdi.EmbDI.embeddings import learn_embeddings

default_values = {
    'experiment_type': 'EQ',
    'smoothing_method': 'no',
    'smooth_k': '0.2',
    'inverse_k': '0.5',
    'smooth_t': '200',
    'log_base': '10',
    'window_size': '3',
    'n_dimensions': '30',
    'sentence_length': 10,
    'walks_strategy': 'basic',
    'write_walks': 'walk/small.walk',
    'ntop': '10',
    'ncand': '1',
    'max_rank': '3',
    'learning_method': 'skipgram',
    'training_algorithm': 'word2vec',
    'follow_sub': '',
    'task': 'train-test',
    'with_cid': 'all',
    'with_rid': 'first',
    'numeric': 'no',
    'backtrack': True,
    'run-tag': 'emb',
}


def generate_tuple(df: pd.DataFrame):
    # sentence == row_id + every row value
    sentences = []
    for elements in df.itertuples():
        sentence = []
        rid = elements[0] # row index
        sentence.append(rid)
        # token id
        element = elements[1:]
        for ele in element:
            sentence.append(ele)
        sentences.append(sentence)
    return sentences


def refine_sentence(sentences):
    # rid: idx__
    # token: tt__
    refine_sens = []
    for sentence in sentences:
        sens = []
        idxes, tokens = sentence[0],sentence[1:]
        idx_node = f'idx__{idxes}'
        sens.append(idx_node)
        for token in tokens:
            node_m = multi_words(token)
            token_node = f'tt__{node_m}'
            sens.append(token_node)
        refine_sens.append(sens)
    return refine_sens


def load_tuple_sentence(sentences:list, output_fp):
    # save sentences/walks to a
    fp = open(output_fp, 'w')
    for sentence in sentences:
        ws = ' '.join(str(v) for v in sentence)
        s = ws + '\n'
        fp.write(s)


def tuple_emb(output_file, walks):
    t = 'emb/' + output_file + '.emb'
    print('File: {}'.format(t))
    learn_embeddings(t, walks, write_walks=default_values['write_walks'],
                     dimensions=int(default_values['n_dimensions']),
                     window_size=int(default_values['window_size']),
                     training_algorithm=default_values['training_algorithm'],
                     learning_method=default_values['learning_method'],
                     )


def test_load_sentence():
    fin = 'dataset/med_demo.csv'
    walkin = 'walk/med_walk.walk'
    df = pd.read_csv(fin)
    sentences = generate_tuple(df)
    refine_res = refine_sentence(sentences)
    load_tuple_sentence(refine_res, walkin)


def main():
    # input sentences
    # output embedding file
    f_name = 'med'
    walk_p = 'walk/med_walk.walk'
    tuple_emb(f_name, walk_p)


if __name__ == '__main__':
    # test_load_sentence()
    main()