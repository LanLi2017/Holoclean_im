# this is to create a dirty hospital dataset
import csv
import random
from pprint import pprint
import pandas as pd
import numpy as np


def cal_nan(grd_tb, baseline_tb):
    # what's the proportion missing values for ground truth dataset?: grd_tb
    grd = pd.read_csv(grd_tb)
    size_tb = grd.size
    count_na_grd = grd.isna().values.sum()
    grd_stack = grd.stack(dropna=False)
    na_coor_grd = [list(x) for x in grd_stack.index[grd_stack.isna()]]
    prop_grd = count_na_grd/size_tb
    print(f'nan in ground truth/clean data: {count_na_grd} ==> proportion {prop_grd}')

    # what's the proportion missing values HoloClean used?: baseline_tb
    baseline = pd.read_csv(baseline_tb)
    count_na_base = baseline.isna().values.sum()
    baseline_stack = baseline.stack(dropna=False)
    na_coor_base = [list(x) for x in baseline_stack.index[baseline_stack.isna()]]
    prop_base = count_na_base/size_tb
    print(f'nan in holoclean baseline: {count_na_base} ==> proportion {prop_base}')

    diff_null_list = [i for i in na_coor_grd + na_coor_base if i not in na_coor_grd or i not in na_coor_base]
    print(f'if the nan position is equal? ==> {diff_null_list}')
    print(f'how many different null values? ==> {len(diff_null_list)}')
    return prop_base,size_tb,grd,na_coor_grd


def get_missing_sample(grd, rand_coord,i,frac):
    # rand_coord: list of lists, randomly generated coordinates for missing value replacement
    # [[19, 'Score'], [26, 'Score'], [27, 'Score'],...]
    count_na_grd = grd.isna().values.sum()
    print(f'before replacement: nan count {count_na_grd}')
    for _,v in enumerate(rand_coord):
        row_id = v[0]
        col_name = v[1]
        print(f'row is {row_id} in column {col_name}')
        print(f'before replace: {grd[col_name].iloc[row_id]}')
        grd[col_name].iloc[row_id] = None
        print(f'after replace: {grd[col_name].iloc[row_id]}')
    count_na_grd_af = grd.isna().values.sum()
    print(f'after replacement: nan count {count_na_grd_af}')
    grd.to_csv(f'Missing/Aug_missing_v{i}_frac{frac}.csv')


def get_nonmatch_sample(df, col_name,nonmatch_frac):
    # impute mis-match data into the missing sample
    rows = len(df.index)
    # how many dirty data will be imputated?
    replace_count = int(nonmatch_frac * rows)
    # dirty room_type and price; suppose correlations
    # shared room < private room < Entire home/apt
    # highest: 200 < 800< 2000+
    avai_choices_rm_tp = list(set(df[col_name]))
    # remove 'nan'
    cleaned_ava_choices = [x for x in avai_choices_rm_tp if str(x) != 'nan']
    col_index = df.columns.get_loc(col_name)
    # random replacement indexes
    ran_rep_index =random.sample(range(0,rows), replace_count)
    # df.loc[ran_rep_index, 'room_type'] = random.choice(avai_choices_rm_tp)
    for id in ran_rep_index:
        # print(f'cleaned available choices; {cleaned_ava_choices}')
        if not cleaned_ava_choices:
            continue
        rv = random.choice(cleaned_ava_choices)
        df.iloc[id, col_index] = rv # first two columns of data frame with all rows

    return df


def aug_mis_mim():
    # slice complete data
    # delete values from current table
    # impute mimatched pattern values + missing values
    file = 'Ground_truth/hospital_clean.csv'
    frac_1 = np.arange(0.8,0.95,0.1) # 0.5, 0.6, 0.7, [0.8, 0.9]
    frac_2 = np.arange(0.05, 0.2, 0.1) # 0.05, 0.15, [0.25, 0.35, 0.45
    pair_fracs = []
    for f in frac_1:
        pair_fracs.extend(list(zip([f for _ in range(5)],frac_2)))
    pprint(pair_fracs)
    i = 0
    for frac in frac_1:
        # add missing values
        df_input = get_missing_sample(file, frac)
        df_cp = df_input.copy()
        df_replace = pd.DataFrame()
        # input the column names which we want them to be dirtier.
        dirty_cols = df_input.columns.values.tolist()[1:]

        # add mismatched
        for frac_mis in frac_2:
            for col in dirty_cols:
                df_replace = get_nonmatch_sample(df_input, col, frac_mis)
                # changes = df_replace.compare(df_cp)
                df_replace.to_csv(f'Tune_res/hos_sample_dirty_{i}.csv', index=False)
            i = i+1
            print(i)
    return pair_fracs


def aug_mis():
    # only augment missing value
    prop_base, size_tb,grd,na_coor_grd = cal_nan('Ground_truth/hospital_clean.csv', 'Holoclean_test_input/hospital.csv')
    frac_2 = np.arange(0.1,0.5, 0.1)  # missing 0.1, 0.2, 0.3, 0.4 [original]

    for i,v in enumerate(frac_2):
        grd_cp = grd.copy()
        rand_count = int(size_tb * v)
        x_y = [[x, grd_cp.columns[y]] for x,y in zip(*np.where(grd_cp.values))]
        un_nan_xy = [coor for coor in x_y if coor not in na_coor_grd]
        rand_coord = random.sample(un_nan_xy, k=rand_count) # randomly generate coordinates [[19, 'Score'],...]
        get_missing_sample(grd_cp, rand_coord,i,v)


def aug_mimatch():
    # only augment mimatched values
    
    pass


def main_():
    cal_nan('Ground_truth/hospital_clean.csv', 'Holoclean_test_input/hospital.csv')


if __name__ == '__main__':
    # main_2()
    # main_()
    aug_mis()



