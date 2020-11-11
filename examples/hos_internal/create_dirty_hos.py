# this is to create a dirty hospital dataset
import csv
import random
from pprint import pprint
import pandas as pd
import numpy as np


def get_missing_sample(file, frac):
    # abb_slice_clean.csv is the clean csv which still covers some weird symbols
    # clean it again: abb_sli_clean2.csv
    # delete randomly
    df = pd.read_csv(file)
    cols = df.columns.values.tolist()
    # sample isnull
    for col in cols[1:]:
        df[col] = df[col].sample(frac=frac)
    pprint(df)
    return df


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


def main_():
    # slice complete data
    # delete values from current table
    # impute mimatched pattern values
    file = 'recon_hospital_clean.csv'
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


if __name__ == '__main__':
    # main_2()
    main_()



