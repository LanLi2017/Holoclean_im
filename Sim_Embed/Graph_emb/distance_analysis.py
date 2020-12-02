from pprint import pprint

import pandas as pd


def find_skip_pos(slice_col):
    # slice the dataset
    from_list = list(slice_col)
    start_v = from_list[0]
    skip_point = []  # index list for skipping to next node
    count = 0
    for from_node in from_list:
        if from_node == start_v:
            count += 1
            continue
        else:
            skip_point.append(count - 1)
            start_v = from_node
            count += 1
    if skip_point[:-1] != len(from_list):
        skip_point.append(len(from_list))
    return skip_point


# this is to analysis the generated distance file under 'distance/' folder
def extract_top_cand(n, distance_f):
    # according to the cosine similarity
    # return the closest n records
    '''
    params n: top n candidates

    '''
    df = pd.read_csv(distance_f, index_col=False)
    new_df = df.sort_values(by=['record_from', 'cos_distance'])

    slice_col = new_df['record_from']
    skip_pos = find_skip_pos(slice_col)
    start_id = 0
    top_n = []
    for id in skip_pos:
        slice_df = new_df[start_id: id+1]
        slice_row_count = slice_df.shape[0]
        if slice_row_count <= n:
            top_n_cand = slice_df
        else:
            top_n_cand = slice_df[slice_row_count-n:slice_row_count+1]

        for cand in top_n_cand.itertuples():
            cand_v = cand[1:]
            top_n.append(cand_v)

        # print(f'range is : {[start_id,id+1]}')
        start_id = id+1
    return top_n


def save_candidates(top_n, fp):
    # save top_n distances between records
    from_list = []
    to_list = []
    dis_list = []
    for (from_node, to_node, distance) in top_n:
        from_list.append(from_node)
        to_list.append(to_node)
        dis_list.append(distance)

    col_dict = {'record_from': from_list, 'record_to': to_list, 'distance': dis_list}
    df = pd.DataFrame(col_dict)
    df.to_csv(fp, index=False)

    assert len(top_n) == df.shape[0]


def main():
    n = 1
    top_n = extract_top_cand(n, 'distance/med_distance.csv') # most similar records
    fp = f'distance/top_candidates/med_top_{n}_cand.csv'
    save_candidates(top_n, fp)


if __name__ == '__main__':
    main()
