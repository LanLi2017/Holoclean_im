from pprint import pprint

import holoclean
from detect import NullDetector, ViolationDetector
import pandas as pd

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
hc.load_data('hospital', '../../testdata1/hospital.csv')
# load constraints violations
hc.load_dcs('../../testdata1/hospital_constraints.txt')
hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
# errors: list of errors by different detectors
errors = hc.detect_errors(detectors)

# errors: list of two dataframes : null errors df; violation errors df
# ['_tid_': row index, 'attribute': column name]
null_er_df = errors[0]
vio_er_df = errors[1]


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


def tuple_emb():
    pass


def main():
    null_coor = error_coordinate(null_er_df)
    vio_coor = error_coordinate(vio_er_df)


if __name__ == '__main__':
    main()
