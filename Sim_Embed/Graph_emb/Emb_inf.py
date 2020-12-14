'''
This script is used to generate random walks starting from a given edgelist without the overhead required when running
the full algorithm. The parameters used here are the same as what is used in the main algorithm, so please refer to the
readme for more details.

@author: riccardo cappuzzo

'''
from embdi.EmbDI.embeddings import learn_embeddings
from embdi.EmbDI.utils import *
from embdi.EmbDI.graph import graph_generation, Graph
from embdi.EmbDI.sentence_generation_strategies import random_walks_generation
import pandas as pd

import argparse

from embdi.edgelist import EdgeList

# Default parameters
configuration = {
    'walks_strategy': 'basic',
    'flatten': 'all',
    'input_file': 'pipeline/edgelist/small_example.edgelist', # generate edge list file
    'n_sentences': 'default',
    'sentence_length': 10,
    'write_walks': True,
    'intersection': False,
    'backtrack': True,
    'walk_file': 'small_example', # walk file name
    'repl_numbers': False,
    'repl_strings': False,
    'follow_replacement': False,
    'mlflow': False,
    'learning_method': 'skipgram',
    'training_algorithm': 'word2vec',
    'window_size': '3',
    'n_dimensions': '30',
    'emb_f': 'pipeline/embedding/small.embedding',
}


def create_edge_list(csvfile, edgefile):
    df = pd.read_csv(csvfile)

    # edgefile = args.output_file
    edgefile = edgefile

    pref = ['3#__tn', '3$__tt', '5$__idx', '1$__cid']

    el = EdgeList(df, edgefile, pref)

    # Loading the graph to make sure it can load the edgelist.
    g = Graph(el.get_edgelist(), prefixes=pref)
    print(f'the cell list: {g.cell_list}')
    print(len(g.cell_list))
    print(f'the nodes: {g.get_node_list()}')
    print(f'the nodes lenth: {len(g.get_node_list())}')
    for node in g.get_node_list():
        print(g._get_node_type(node))

    print(f'the edges: {g.edges}')
    print(f'the node classes: {g.node_classes}')
    print(f'the numeric nodes: {g.node_is_numeric}')


def gen_sentences(csvfile):
    ''''../sim_embed/graph_emb/dataset/small_demo.csv'''
    df = pd.read_csv(csvfile)

    prefixes, edgelist = read_edgelist(configuration['input_file'])

    graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
    if configuration['n_sentences'] == 'default':
        #  Compute the number of sentences according to the rule of thumb.
        configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))
    walks = random_walks_generation(configuration, df, graph)
    return walks


def gen_embedding(walks, emb_p, emb_size):
    '''
    :param emb_p: path to save the embeddings file into.
    :param walks: path to the walks file (if write_walks == True), list of walks otherwise.
    :param emb_size: number of dimensions
    '''
    res = learn_embeddings(emb_p, walks, write_walks=configuration['write_walks'],
                           dimensions=int(emb_size),
                           window_size=int(configuration['window_size']),
                           training_algorithm=configuration['training_algorithm'],
                           learning_method=configuration['learning_method'],
                           )
    return res


def main():
    # csvfile = 'pipeline/dataset/small_demo.csv'
    # edgef = configuration['input_file']
    # create_edge_list(csvfile,edgef)
    # walk = gen_sentences(csvfile)  # generate walks/sentences
    walk_p = 'pipeline/walks/' + configuration['walk_file'] + '.walks' # 'walks': 'pipeline/walks/small_example.walks',
    print(walk_p)
    gen_embedding(walk_p, configuration['emb_f'],configuration['n_dimensions'])


if __name__ == '__main__':
    main()
