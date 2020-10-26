import csv
from pprint import pprint
import pandas as pd


def split_data(f_path):
    slice_d = []
    with open(f_path, 'r')as f:
        data = csv.reader(f)
        for d in data:
            slice_d.append(d)
    df = pd.DataFrame(slice_d[1:], columns=slice_d[0])
    return df.sample(frac=0.0016)


def get_test_data():
    df = split_data('PPP_all_state.csv')
    df.to_csv('ppp_slice.csv', index=False)
    # data = split_data('../testdata/PPP_Data.csv')
    # df = pd.DataFrame(data[1:], columns=data[0])
    #
    # df.to_csv('ppp_slice.csv', index=False)


def main():
    get_test_data()
    # test_ppp = csv.writer(data, )


if __name__ == '__main__':
    main()



