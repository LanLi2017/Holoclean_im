# Reimplement EmbDI
# https://gitlab.eurecom.fr/cappuzzo/embdi

import datetime
from pprint import pprint
import random

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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


class Edge:
    def __init__(self, node_from, node_to, weight_forward=1, weight_back=1):
        self.node_from = node_from
        self.node_to = node_to
        self.weight_forward = weight_forward
        self.weight_back = weight_back


class Node:
    def __init__(self, G: nx.Graph):
        '''
        input Graph G
        return metadata of nodes, edges
        '''
        # self.neighbor_name = []
        self.G = G
        # self.neighbor = {}
        self.start_from = []
        self.similar_tokens = []
        self.similar_distance = [1.0]
        self.nodeType = ''

    def node_metadata(self):
        # node_data: ('callahan eye foundation hospital', {'Type': 'TID'})
        node_list = list(self.G.nodes)
        node_data = list(self.G.nodes(data=True))
        return node_list, node_data

    def node_type(self, node_name):
        self.nodeType = self.G.nodes[node_name]['Type']
        return self.nodeType

    def edge_metadata(self):
        edge_list = [e for e in self.G.edges]
        edge_data = [[e for e in self.G.edges.data()]]  # no weight now
        return edge_data

    def find_neighbors(self, starting_node):
        '''
        # "using graphs and random walks allows us to have a richer and more diverse set of
        # neighborhoods"
        # neigh: attribute co-occurrance & row level relationships
        '''
        # neighbors = [n for n in self.G.neighbors(starting_node)]
        node_data = nx.get_node_attributes(self.G, 'Type')
        neighbors = {}

        for node in self.G.neighbors(starting_node):
            neighbor_type = node_data[node]
            neighbor_name = node
            neighbors[neighbor_name] = neighbor_type
        return neighbors

    def find_random_neighbor(self, starting_node):
        neighbor = self.find_neighbors(starting_node)
        neighbor_name = []
        for key, value in neighbor.items():
            neighbor_name.append(key)
        # rand_neigh = np.random.choice(self.neighbor_name, size=1)[0]
        rand_neigh = random.choice(neighbor_name)
        return rand_neigh

    def findNeighbor_RID(self, starting_node):
        # find the record id to starting_node
        neigh_nodes = self.find_neighbors(starting_node)
        filtered_nodes = {k: v for (k, v) in neigh_nodes.items() if v == 'RID'}
        rid = []
        for key, value in filtered_nodes.items():
            rid.append(key)
        return np.random.choice(rid, size=1)[0]

    def add_similar(self, other, distance):
        # TBD
        self.similar_tokens.append(other)
        self.similar_distance.append(distance)

    def normalize_token(self):
        '''# node merging: e.g. the U.S == United States '''
        pass


# missing value/ mismatched value node
def token_node(G, element, rid, attr_node, col_i, flatten=True):
    '''
    param G: current graph with attribute id; and record id
    param element: cell value
    param rid: record id/ row id
    param attr_node: list of attribute id nodes
    param col_i: index of column; map to attr_node
    param flatten: if True, split; else keep

    if cell value is multi-word: tokenize and add node; add edge
    if cell value is single-word: add node, add edge
    '''
    if flatten:
        pass
    else:
        e1 = Edge(rid, element)  # weight TBD
        G.add_node(element, Type='TID')
        G.add_edge(e1.node_from, e1.node_to)
        # nxg.add_edge(edge.node_from, edge.node_to)
        # print(f'element: {element} ==> in the column {attr_node[col_i]}')
        e2 = Edge(element, attr_node[col_i])
        G.add_edge(e2.node_to, e2.node_from)
    return G


def multi_words(node_v):
    try:
        node_split = node_v.split(' ')
        if len(node_split) > 1:
            # contain multiple words
            new_node_v = node_v.replace(" ", "_")
        else:
            new_node_v = node_v
        return new_node_v
    except:
        return node_v


def generate_graph(df: pd.DataFrame):
    G = nx.Graph()
    attr_node = []  # list of attribute id nodes
    for c in df.columns.values:
        # column id / attribute id
        attr_node.append(c)
    G.add_nodes_from(attr_node, Type='CID')
    for value in df.itertuples():
        # add record id
        rid = value[0]
        G.add_node(rid, Type='RID')
        # token id
        element = value[1:]
        for i, e in enumerate(element):
            token_node(G, e, rid, attr_node, i, flatten=False)
    return G


def replace_numeric_value(value, nodes):
    # TBD
    node_type = nodes.node_type(value)

    if node_type == 'TID':
        try:
            value = int(value)
        except ValueError:
            return value
        new_val = np.around(np.random.normal(loc=value, scale=1))
        cc = 0
        try:
            new_val = int(new_val)
        except OverflowError:
            return str(value)
        while new_val not in nodes.keys() and str(new_val) not in nodes.keys() and float(
                new_val) not in nodes.keys():
            if cc > 1:
                return str(value)
            new_val = np.around(np.random.normal(loc=value, scale=1))
            cc += 1
        return str(int(new_val))
    else:
        print(value)
        return value


def replace_string_value(value: Node):
    # replace similar token, node merge
    # e.g. the US == United Sates
    # if len(value.similar_tokens) > 1:
    #     return value, value.get_random_replacement()
    # else:
    #     return value
    pass


def refine_walk(walks: list, G: nx.Graph):
    # rewrite node name: rid: idx__ ; token: tt__; column: cid__
    refine_res = []
    nn = Node(G)

    for walk in walks:
        # [record node, token node, column node]
        node_list = []
        for node in walk:
            node_t = nn.node_type(node)  # return the node type: cid/tid/rid
            if node_t == 'CID':
                check_node = multi_words(node)
                new_node = f'cid__{check_node}'
                node_list.append(new_node)
            elif node_t == 'TID':
                check_node = multi_words(node)
                new_node = f'tt__{check_node}'
                node_list.append(new_node)
            elif node_t == 'RID':
                new_node = f'idx__{node}'
                node_list.append(new_node)
        refine_res.append(node_list)
    return refine_res


def random_walk(starting_node, rand_walk_length, G, repl_numbers=True):
    # random
    """
    starting_node: starting token node
    rand_walk_length: budget of random walk
    G: graph

    1. choosing a neighboring RID of Ti, Rj
    2. randomly choose from neighbors of node Rj
    3. a new neighbor of Ca will be chosen
    ... recursively until reach length
    """
    node = Node(G)
    r_start = node.findNeighbor_RID(starting_node)
    walk = [r_start, starting_node]
    currentNode = starting_node

    while len(walk) < rand_walk_length:
        nextNode = node.find_random_neighbor(currentNode)
        # if repl_numbers:
        #     nextNode = replace_numeric_value(nextNode, node)
        # if repl_strings:
        #     nextNode, replaced_node = replace_string_value(node.node_matadata()[0])
        # else:
        #     replaced_node = nextNode

        # if nextNode not in walk: # visited already
        #     continue
        # else:
        walk.append(nextNode)
        currentNode = nextNode

    return walk


def compute_n_sentences(nodes_list, sentence_length, factor=1000):
    """Compute the default number of sentences according to the rule of thumb:
    n_sentences = n_nodes * representation_factor // sentence_length

    :param sentence_length: target sentence length
    :param factor: "desired" number of occurrences of each node
    :return: n_sentences
    """
    print(type(sentence_length))
    n = len(nodes_list) * factor // sentence_length
    print('# Computing default number of sentences.\n{} sentences will be generated.'.format(n))
    return n


def generate_sentences(G, sentence_length):
    '''
    G: graph G
    '''
    # number of random walks; number of nodes
    sentences = []
    sentence_counter = 0
    count_cells = 0
    n = Node(G)
    nodes_list = n.node_metadata()[1]  # [('ProviderNumber', {'Type': 'CID'}), ('HospitalName', {'Type': 'CID'})
    cell_list = [x[0] for x in nodes_list if x[1] == {'Type': 'TID'}]  # only token is the cell

    number_sentences = compute_n_sentences(nodes_list, sentence_length)
    random_walks_per_node = number_sentences // len(cell_list)

    pbar = tqdm(desc='Sentence generation progress', total=len(cell_list) * random_walks_per_node)
    for cell in cell_list:
        r = []
        for i in range(random_walks_per_node):
            walk = random_walk(cell, sentence_length, G)
            r.append(walk)
        sentences += r
        sentence_counter += random_walks_per_node
        count_cells += 1
        pbar.update(random_walks_per_node)
    pbar.close()

    needed = number_sentences - sentence_counter
    if needed > 0:
        with tqdm(total=needed, desc='Completing fraction of random walks') as pbar:
            for count_cells in range(needed):
                # while needed > count_cells:
                cell = random.choice(cell_list)
                w = random_walk(cell, sentence_length, G)
                sen = [w]
                for r in sen:
                    sentences.append(r)
                pbar.update(1)
    return sentences


def embeddings_generation(walks):
    """
    Take the generated walks and train embeddings using the walks as training corpus.
    :param walks:
    :param configuration:
    :param dictionary:
    :return:
    """
    output_file = default_values['run-tag']

    t = 'embeddings/' + output_file + '.emb'
    print('File: {}'.format(t))
    learn_embeddings(t, walks, write_walks=default_values['write_walks'],
                     dimensions=int(default_values['n_dimensions']),
                     window_size=int(default_values['window_size']),
                     training_algorithm=default_values['training_algorithm'],
                     learning_method=default_values['learning_method'],
                     )


def position_MultiPartiteGraph(Graph, Parts):
    # Graph is a networkX Graph object, where the nodes have attribute 'agentType' with part name as a value
    # Parts is a list of names for the parts (to be shown as columns)
    # returns list of dictionaries with keys being networkX Nodes, values being x,y coordinates for plottingxPos = {}
    xPos = {}
    yPos = {}
    for index, Type in enumerate(Parts):
        xPos[Type] = index
        yPos[Type] = 0

    pos = {}
    for node, attrDict in Graph.nodes(data=True):
        Type = attrDict['Type']
        print('node: %s\tagentType: %s' % (node, Type))
        print('\t(x,y): (%d,%d)' % (xPos[Type], yPos[Type]))
        pos[node] = (xPos[Type], yPos[Type])
        yPos[Type] += 1

    return pos


def plot_Graph(G):
    nx.draw(G, pos=position_MultiPartiteGraph(G, ['RID', 'TID', 'CID']), with_labels=True)
    plt.show()


def test_wf():
    df = pd.read_csv('dataset/small_demo.csv')
    G = generate_graph(df)
    # plot_Graph(G)
    n = Node(G)
    # find neighbors
    nodes_list = n.node_metadata()[1]
    cell_list = [x[0] for x in nodes_list if x[1] == {'Type': 'TID'}]
    print(cell_list)
    print(len(cell_list))
    print(n.node_metadata()[1])
    nodes_ele = n.node_metadata()[0]
    print(f'nodes length: {len(nodes_ele)}')
    # neighbors = n.find_neighbors(35660)
    # print(neighbors)
    # rid = n.findNeighbor_RID(35660)
    # print(rid)
    # print(neighbors)
    # neighbor_name = []
    # for key, value in neighbors.items():
    #     neighbor_name.append(key)
    # ran_n = np.random.choice(neighbor_name, size=1)[0]
    # print(ran_n)


def test_generate_walk_file():
    df = pd.read_csv('dataset/small_demo.csv')
    G = generate_graph(df)
    # walk = random_walk(35660,10, G)
    sentences = generate_sentences(G, sentence_length=10)
    refine_res = refine_walk(sentences, G)
    walks_file = 'walk/small.walk'
    fp_walks = open(walks_file, 'w')
    for sen in refine_res:
        ws = ' '.join(str(v) for v in sen)
        s = ws + '\n'
        fp_walks.write(s)


def generate_embeddings():
    walks_file = 'walk/small.walk'
    embeddings_generation(walks_file)


def main(csv_file_path, walk_path):
    df = pd.read_csv(csv_file_path)
    G = generate_graph(df)
    # walk = random_walk(35660,10, G)
    sentences = generate_sentences(G, default_values['sentence_length'])
    refine_res = refine_walk(sentences, G)
    fp_walks = open(walk_path, 'w')
    for sen in refine_res:
        ws = ' '.join(str(v) for v in sen)
        s = ws + '\n'
        fp_walks.write(s)

    embeddings_generation(walk_path)


if __name__ == '__main__':
    # test_generate_walk_file()
    # generate_embeddings()
    # wf_test()
    main('dataset/small_demo.csv', 'walk/small.walk')
