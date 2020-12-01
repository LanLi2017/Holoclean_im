from pprint import pprint

from sklearn.metrics.pairwise import cosine_similarity

import holoclean
from Sim_Embed.Graph_emb import light_imp as embedding
from detect import NullDetector, ViolationDetector
import pandas as pd
import numpy as np


def req_holoclean(file,tb_name,constraints_f):
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=True,
        timeout=3 * 60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session
    # load dataset
    hc.load_data(tb_name, file)
    # load constraints violations
    hc.load_dcs(constraints_f)
    hc.ds.set_constraints(hc.get_dcs())

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector(), ViolationDetector()]
    # errors: list of errors by different detectors
    errors = hc.detect_errors(detectors)

    # errors: list of two dataframes : null errors df; violation errors df
    # ['_tid_': row index, 'attribute': column name]
    null_er_df = errors[0]
    vio_er_df = errors[1]
    return null_er_df, vio_er_df


# check values in tuple: null or violate
def check_tuple_v(row, col, null_coor, vio_coor):
    if (row, col) in null_coor:
        # if the tuple is missing a value, it might be the most likely candidate

        pass
    elif (row, col) in vio_coor:
        # if it's the violated
        # look at all the candidates from the column
        # and identify the most likely one based on the context of the tuple and domain
        pass


def save_error_file(err_list:list, error_file):
    fp_walks = open(error_file, 'w')
    for error in err_list:
        ws = ','.join(str(v) for v in error)
        s = ws + '\n'
        fp_walks.write(s)


def load_error_file(error_file):
    '''
    open error_file;
    load the error list: [(rid, cid), ...]

    '''
    error_list = []
    with open(error_file, 'r')as f:
        for line in f.readlines():
            line_str = line.strip('\n')
            error_t = line_str.split(',')
            error_list.append((int(error_t[0]), error_t[1]))
    return error_list


def test_savefile():
    # save error into files
    # no need to ask Holoclean system every time
    file, tb_name, constraints_f = 'dataset/small_demo.csv', 'hospital', 'constraints/hospital_constraints.txt'
    null_er_df, vio_er_df = req_holoclean(file, tb_name, constraints_f)
    null_coor = error_coordinate(null_er_df) # (rid, cid) record id, column id
    null_error_fp = 'Error/Null_err.txt'
    save_error_file(null_coor, null_error_fp)
    vio_coor = error_coordinate(vio_er_df)
    vio_error_fp = 'Error/Violates_err.txt'
    save_error_file(vio_coor, vio_error_fp)


def error_coordinate(error_df: pd.DataFrame):
    # return [row_index, column_name]
    # error_df: dataframe
    coor = []  # list of tuples: save errors location
    for tp in error_df.itertuples():
        row_idx = tp[1]
        col_name = tp[2]
        coor.append((row_idx, col_name))
    # assert len(coor) == 227
    return coor


def cal_pairwise(error_list):
    # calculate pairwise record id
    rid_err_list = []
    for (rid, cid) in error_list:
        # initial_v = get_element(rid,cid, df)
        rid_err_list.append(rid)
    rid_list = list(set(rid_err_list))
    pair_wise_res = []
    for i in range(len(rid_list)):
        for j in range(i + 1, len(rid_list)):
            pair_wise_res.append((rid_list[i], rid_list[j]))
    return pair_wise_res


def _load_emb(model_file, start_str='idx__'):
    with open(model_file, 'r', encoding='utf-8') as fp:
        s = fp.readline()
        _, dimensions = s.strip().split(' ')
        viable_idx = []
        for i, row in enumerate(fp):
            if i >= 0:
                idx, vec = row.split(' ', maxsplit=1)
                if idx.startswith(start_str):
                    try:
                        prefix, n = idx.split('__')
                        n = int(n)
                    except ValueError:
                        continue
                    viable_idx.append(row)
        # viable_idx = [row for idx, row in enumerate(fp) if idx > 0 and row.startswith('idx_')]

    f = 'embeddings/dump/indices.emb'
    with open(f, 'w', encoding='utf-8') as fp:
        fp.write('{} {}\n'.format(len(viable_idx), dimensions))
        for _ in viable_idx:
            fp.write(_)

    return f, viable_idx


def clean_embedding(emb_file):
    # extract rid and corresponding embedding
    emb_dict = {}
    with open(emb_file, 'r') as fp:
        n, dim = map(int, fp.readline().split())
        for idx, line in enumerate(fp):
            k, v = line.rstrip().split(' ', maxsplit=1)
            key = k.split('__')[1]
            vector = list(map(float, v.split(' ')))
            emb_dict[key] = vector
    return emb_dict


def cal_pairwise_rid(error_list, emb_file):
    # calculate distances between record/row id
    '''
    params error_list: [(rid, cid)]
    rid_embeddings: every index embeddings
    '''
    # extract rid and corresponding embedding
    rid_dict = clean_embedding(emb_file)
    # rid_dict: {rid: rid_embedding}

    # calculate pairwise record id
    pair_wise_res = cal_pairwise(error_list)

    # use embeddings to calculate cosine-similarity pairwisely for each index
    distance_res = {}
    for (rid_1, rid_2) in pair_wise_res:
        emb_1 = np.array(rid_dict[str(rid_1)])
        emb_2 = np.array(rid_dict[str(rid_2)])
        distance = cosine_similarity(emb_1.reshape(1,-1), emb_2.reshape(1,-1))
        distance_res[f'{rid_1}__{rid_2}'] = distance

    pprint(distance_res)

    return distance_res


def test_loademb():
    # load index/record embedding
    f,idx_emb = _load_emb('embeddings/emb.emb',start_str='idx__')


def test_cal_distance():
    emb_file = 'embeddings/dump/indices.emb'
    rid_dict = clean_embedding(emb_file)
    # rid_dict: {rid: rid_embedding}
    # use embeddings to calculate cosine-similarity pairwisely for each index
    distance_res = {}

    emb_1 = np.array(rid_dict[str(1)])
    emb_2 = np.array(rid_dict[str(2)])
    distance = cosine_similarity(emb_1.reshape(1, -1), emb_2.reshape(1, -1))
    distance_res[f'{1}__{2}'] = distance
    print(distance_res)


def main():
    # load error
    error_file = 'Error/Violates_err.txt'
    vio_error_list = load_error_file(error_file)
    emb_file = 'embeddings/dump/indices.emb'
    rid_l = cal_pairwise_rid(vio_error_list, emb_file)
    # print(rid_l)


if __name__ == '__main__':
    # test_loademb()
    # test_cal_distance()
    # main_cal_distance()
    # main_savefile()
    main()
