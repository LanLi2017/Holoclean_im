import glob
import math
import os
from pprint import pprint

import pandas as pd
import numpy as np

grd = '../Holoclean_test_input/hospital.csv' # compared with the augmented csv file

df_grd = pd.read_csv(grd)
# x_y = [[x, df_grd.columns[y]] for x, y in zip(*np.where(df_grd.values))]
# pprint(x_y)
# total_cells = grd_array.size
# # assert total_cells == 19000
grd_shape = df_grd.shape
grd_size = df_grd.size
assert grd_shape == (1000,19)
assert grd_size == 19000

na_count = df_grd.isna().values.sum()
na_frac = na_count/grd_size
assert na_count == 2227


def infile_path(file_folder):
    infiles = sorted(glob.glob(os.path.join(file_folder, '*.csv')))
    # assert infiles == ['../Missing/Aug_missing_v0_frac0.1.csv','../Missing/Aug_missing_v1_frac0.2.csv','../Missing/Aug_missing_v2_frac0.3.csv','../Missing/Aug_missing_v3_frac0.4.csv']
    return infiles


def diff_mism(df_grd, df_aug, grd_count):
    # difference of mismached
    # dont touch na
    df_grd_replace_na = df_grd.replace(np.nan,'',regex=True)
    na_df_grd = df_grd_replace_na.isna().values.sum()
    assert na_df_grd == 0
    df_aug_replace_na = df_aug.replace(np.nan, '', regex=True)
    na_df_aug = df_aug_replace_na.isna().values.sum()
    assert na_df_aug == 0
    # diff = df_grd_replace_na[~df_grd_replace_na.isin(df_aug_replace_na)]
    diff = df_grd_replace_na == df_aug_replace_na
    colname = list(diff.columns.values)
    # count = diff.value_counts().loc[False]
    diff_count = 0
    for col in colname:
        try:
            diff_count += diff[col].value_counts().loc[False]
        except:
            print(col)
            pass
        # count += diff[col].value_counts().loc[False]
    assert diff_count == grd_count
    # assert len(diff) == 0
    # diff = df_aug_replace_na[df_aug_replace_na != df_grd_replace_na]


def diff(df_grd, df_aug):
    # df_grd: the original dirty input
    # df_aug: the dirty input after augmentation
    # try df_aug - df_grd to make sure the augment fration is correct
    df_grd_stack = df_grd.stack(dropna=False)
    na_coor_df_grd = [list(x) for x in df_grd_stack.index[df_grd_stack.isna()]]

    df_aug_stack = df_aug.stack(dropna=False)
    na_coor_df_aug = [list(x) for x in df_aug_stack.index[df_aug_stack.isna()]]

    # na_coor_df_aug - na_coor_df_grd
    diff = [x for x in na_coor_df_aug if not x in na_coor_df_grd]

    return len(diff)


def cal_diff(file_p, grd_count_list, res_grd):
    res = []
    count_mis = []
    infiles = infile_path(file_p)
    for file in infiles:
        mis_file = pd.read_csv(file)
        mis_file_shape = mis_file.shape
        mis_file_size = mis_file.size
        assert mis_file_shape == (1000, 19)
        assert mis_file_size == 19000
        na_count_mis = mis_file.isna().values.sum()
        count_mis.append(na_count_mis) # make sure the augment count is correct

        mis_aug = diff(df_grd, mis_file) # difference based on original input
        diff_frac = mis_aug/ mis_file_size
        res.append(diff_frac)

    assert count_mis == grd_count_list
    assert res == res_grd


def test():
    file_p = [
        '../Missing/',
        '../Mismatch/',
        '../Mixed/'
    ]
    res_missing_grd = [0.1, 0.2, 0.3, 0.4]
    res_mismatch_grd = [0.1, 0.2, 0.3, 0.4]
    res_mixed_grd = [0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.8]
    grd_count_list = [4127,6027,7927,9827]
    # grd_count_list1 = []

    # test missing augment
    cal_diff(file_p[0], grd_count_list,res_missing_grd)

    # test mismatch augment
    grd_count_list_mismatch = [int(x * 19000) for x in res_mismatch_grd]
    mismatch_file = file_p[1]
    infiles = infile_path(mismatch_file)
    for i, infile in enumerate(infiles):
        df = pd.read_csv(infile)
        nan_count = df.isna().values.sum()
        assert nan_count == 2227 # should not change the missing values of original dataset
        assert nan_count == df_grd.isna().values.sum()
        df_stack = df.stack(dropna=False)
        nan_df = [list(x) for x in df_stack.index[df_stack.isna()]]
        df_grd_stack = df_grd.stack(dropna=False)
        nan_df_grd = [list(x) for x in df_grd_stack.index[df_grd_stack.isna()]]
        assert len(nan_df_grd) == 2227
        diff = [x for x in nan_df if not x in nan_df_grd]
        diff1 = [x for x in nan_df_grd if not x in nan_df]
        assert nan_df == nan_df_grd
        assert len(diff) == 0

        # check if their null coordinates are the same
        mismatch_count = grd_count_list_mismatch[i]
        diff_mism(df_grd, df, mismatch_count)

    # test mixed augment
    # cal_diff(file_p[2], grd_count_list1, res_mixed_grd)
    mixed_f = file_p[2]
    mixed_file = infile_path(mixed_f)
    grd_count_list_mixed = [int(x * 19000) for x in res_mixed_grd]
    # mixed_1 = '../Mixed/Aug_missing_0.1_mismatch_v0_frac0.1.csv'
    mixed_2 = '../Mixed/Aug_missing_0.3_mismatch_v3_frac0.4.csv'

    aug_mixed_test = pd.read_csv(mixed_2)
    nan_count = aug_mixed_test.isna().values.sum()
    assert nan_count == 0.3*19000 + 2227
    # diff_mism(df_grd, aug_mixed, grd_count_list_mixed[-1])

    for i,file in enumerate(mixed_file):
        aug_mixed = pd.read_csv(file)

        diff_mism(df_grd,aug_mixed,grd_count_list_mixed[i])

