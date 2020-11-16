# this is to create a dirty hospital dataset
import csv
import random
from pprint import pprint
import pandas as pd
import numpy as np
import logging

logging.basicConfig(format="%(asctime)s - [%(levelname)5s] - %(message)s",
                    datefmt='%H:%M:%S',
                    filename='Logs/data_augmentation.log',
                    filemode='a',
                    level=logging.DEBUG)


def cal_nan(grd_tb, baseline_tb):
    # what's the proportion missing values for ground truth dataset?: grd_tb
    grd = pd.read_csv(grd_tb)
    size_tb = grd.size
    count_na_grd = grd.isna().values.sum()
    grd_stack = grd.stack(dropna=False)
    na_coor_grd = [list(x) for x in grd_stack.index[grd_stack.isna()]]
    prop_grd = count_na_grd/size_tb
    logging.info(f'nan in ground truth/clean data: {count_na_grd} ==> proportion {prop_grd}')
    print(f'nan in ground truth/clean data: {count_na_grd} ==> proportion {prop_grd}')

    # what's the proportion missing values HoloClean used?: baseline_tb
    baseline = pd.read_csv(baseline_tb)
    count_na_base = baseline.isna().values.sum()
    baseline_stack = baseline.stack(dropna=False)
    na_coor_base = [list(x) for x in baseline_stack.index[baseline_stack.isna()]]
    prop_base = count_na_base/size_tb
    logging.info(f'nan in holoclean baseline: {count_na_base} ==> proportion {prop_base}')
    print(f'nan in holoclean baseline: {count_na_base} ==> proportion {prop_base}')

    diff_null_list = [i for i in na_coor_grd + na_coor_base if i not in na_coor_grd or i not in na_coor_base]
    logging.info(f'if the nan position is equal? ==> {diff_null_list}')
    return prop_base,size_tb,grd,na_coor_base,baseline


def get_missing_sample(base, rand_coord,i,frac):
    # rand_coord: list of lists, randomly generated coordinates for missing value replacement
    # [[19, 'Score'], [26, 'Score'], [27, 'Score'],...]
    count_na_grd = base.isna().values.sum()
    # logging.info(f'before replacement: nan count {count_na_grd}')
    for _,v in enumerate(rand_coord):
        row_id = v[0]
        col_name = v[1]
        logging.info(f'row is {row_id} in column {col_name}')
        logging.info(f'before replace: {base[col_name].iloc[row_id]}')
        base[col_name].iloc[row_id] = np.nan
        logging.info(f'after replace: {base[col_name].iloc[row_id]}')
    count_na_grd_af = base.isna().values.sum()
    logging.info(f'after replacement: nan count {count_na_grd_af}')
    # base['ProviderNumber'] = base['ProviderNumber'].astype(pd.Int64Dtype())
    # base['ZipCode'] = base['ZipCode'].astype(pd.Int64Dtype())
    # base['PhoneNumber'] = base['PhoneNumber'].astype(pd.Int64Dtype())
    # base.to_csv(f'Missing/Aug_missing_v{i}_frac{round(frac,1)}.csv',index=False)
    return base


def domain_value(colname, df):
    # return all possible values in a column as a domain
    domain = list(set(df[colname]))
    avail_values = [v for v in domain if str(v) != 'nan']
    return avail_values


def get_mismatch_sample(base, rand_coord, i, frac, col_list, domain_list):
    # rand_coord: list of lists, randomly generated coordinates for mismatched values
    # [[19, 'Score'], [26, 'Score'], [27, 'Score'],...]
    count_na_grd = base.isna().values.sum()
    logging.info(f'before replacement: nan count {count_na_grd}')
    error = 0
    for _, v in enumerate(rand_coord):
        row_id = v[0]
        col_name = v[1]
        logging.info(f'row is {row_id} in column {col_name}')
        domain_choices = domain_list[col_list.index(col_name)]
        # remove its initial value
        initial_v = base[col_name].iloc[row_id]
        ava_choices = [x for x in domain_choices if x != initial_v ]
        # logging.info(f'before replace: {initial_v}')
        print(f'before replace: {initial_v}')
        # random replacement indexes
        replace_v =random.choice(ava_choices)
        base[col_name].iloc[row_id] = replace_v
        # df.loc[ran_rep_index, 'room_type'] = random.choice(avai_choices_rm_tp)
        # logging.info(f'after replace: {replace_v}')
        print(f'after replace: {replace_v}')
    count_na_grd_af = base.isna().values.sum()
    logging.info(f'after replacement: nan count {count_na_grd_af}')
    # base.to_csv(f'Mismatch/Aug_mismatch_v{i}_frac{round(frac, 1)}.csv', index=False)


def aug_mis():
    # only augment missing value
    prop_base, size_tb,grd,na_coor_base,baseline = cal_nan('Ground_truth/hospital_clean_recon.csv', 'Holoclean_test_input/hospital.csv')
    frac_2 = np.arange(0.1,0.5, 0.1)  # missing 0.1, 0.2, 0.3, 0.4 [original]

    for i,v in enumerate(frac_2):
        baseline_cp = baseline.copy()
        rand_count = int(size_tb * v)
        x_y = [[x, baseline_cp.columns[y]] for x,y in zip(*np.where(baseline_cp.values))]
        un_nan_xy = [coor for coor in x_y if coor not in na_coor_base]
        rand_coord = random.sample(un_nan_xy, k=rand_count) # randomly generate coordinates [[19, 'Score'],...]
        get_missing_sample(baseline_cp, rand_coord,i,v)


def aug_mimatch():
    # only augment mimatched values
    frac_2 = np.arange(0.1, 0.5, 0.1)  # missing 0.1, 0.2, 0.3, 0.4 [original]
    prop_base, size_tb, grd, na_coor_base, baseline = cal_nan('Ground_truth/hospital_clean_recon.csv',
                                                             'Holoclean_test_input/hospital.csv')
    # group the available choices in a column
    col_list = list(baseline.columns.values)
    domain_list = [domain_value(colname, baseline) for colname in col_list]

    for i, v in enumerate(frac_2):
        baseline_cp = baseline.copy()
        rand_count = int(size_tb * v)
        print(rand_count)
        x_y = [[x, baseline_cp.columns[y]] for x, y in zip(*np.where(baseline_cp.values))] # enumerate all of cells in a table
        un_nan_xy = [coor for coor in x_y if coor not in na_coor_base] # remove nan cell
        rand_coord = random.sample(un_nan_xy, k=rand_count)  # randomly generate coordinates [[19, 'Score'],...]
        get_mismatch_sample(baseline_cp, rand_coord, i, v, col_list, domain_list)


def get_mismatch_sample1(aug_data_cp, rand_coord1, j, frac_mismatch, col_list, domain_list, frac_missing):
    count_na_grd = aug_data_cp.isna().values.sum()
    logging.info(f'before replacement: nan count {count_na_grd}')

    for _, v in enumerate(rand_coord1):
        row_id = v[0]
        col_name = v[1]
        logging.info(f'row is {row_id} in column {col_name}')
        domain_choices = domain_list[col_list.index(col_name)]
        # remove its initial value
        initial_v = aug_data_cp[col_name].iloc[row_id]
        ava_choices = [x for x in domain_choices if x != initial_v]
        logging.info(f'before replace: {initial_v}')
        # random replacement indexes
        replace_v = random.choice(ava_choices)
        aug_data_cp[col_name].iloc[row_id] = replace_v
        # df.loc[ran_rep_index, 'room_type'] = random.choice(avai_choices_rm_tp)
        logging.info(f'after replace: {replace_v}')
    count_na_grd_af = aug_data_cp.isna().values.sum()
    logging.info(f'after replacement: nan count {count_na_grd_af}')
    aug_data_cp.to_csv(f'Mixed/Aug_missing_{frac_missing}_mismatch_v{j}_frac{round(frac_mismatch, 1)}.csv', index=False)


def aug_mis_mim():
    # impute mimatched pattern values + missing values
    frac_1 = [0.1,0.2,0.3,0.4] # 0.5, 0.6, 0.7, [0.8, 0.9]
    frac_2 = [0.1,0.2,0.3,0.4] # 0.05, 0.15, [0.25, 0.35, 0.45

    prop_base, size_tb, grd, na_coor_base, baseline = cal_nan('Ground_truth/hospital_clean_recon.csv',
                                                              'Holoclean_test_input/hospital.csv')

    for i,frac in enumerate(frac_1):
        # add missing values
        baseline_cp = baseline.copy()
        rand_count = int(size_tb * frac)
        x_y = [[x, baseline_cp.columns[y]] for x, y in zip(*np.where(baseline_cp.values))]
        un_nan_xy = [coor for coor in x_y if coor not in na_coor_base]
        rand_coord = random.sample(un_nan_xy, k=rand_count)  # randomly generate coordinates [[19, 'Score'],...]
        aug_data = get_missing_sample(baseline_cp, rand_coord, i, frac) # current status of data after aug missing values

        # add mismatched
        for j,frac_mismatch in enumerate(frac_2):
            aug_data_cp = aug_data.copy()

            # group the available choices in a column
            col_list = list(aug_data_cp.columns.values)
            domain_list = [domain_value(colname, aug_data_cp) for colname in col_list]

            rand_count1 = int(size_tb * frac_mismatch) # how many values need to be replaced?
            mixed_stack = aug_data_cp.stack(dropna=False)
            na_coor_base1 = [list(x) for x in mixed_stack.index[mixed_stack.isna()]] # current null values in new table
            un_nan_xy1 = [coor for coor in x_y if coor not in na_coor_base1]  # remove nan cell
            rand_coord1 = random.sample(un_nan_xy1, k=rand_count1)  # randomly generate coordinates [[19, 'Score'],...]
            get_mismatch_sample1(aug_data_cp, rand_coord1, j, frac_mismatch, col_list, domain_list, frac)


def main_():
    cal_nan('Ground_truth/hospital_clean_recon.csv', 'Holoclean_test_input/hospital.csv')


def demo3():
    res_mixed_grd = [0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.8]
    count_list = [x*19000 for x in res_mixed_grd]
    print(count_list)


def demo2():
    df = pd.read_csv('Mismatch/Aug_mismatch_v0_frac0.1.csv')
    nan_count = df.isna().values.sum()

    df1 = pd.read_csv('Holoclean_test_input/hospital.csv')
    nan_count_compare = df1.isna().values.sum()

    df1replace_na = df1.replace(np.nan, '', regex=True)
    na_df1 = df1replace_na.isna().values.sum()

    df_replace_na = df.replace(np.nan, '', regex=True)
    na_df = df_replace_na.isna().values.sum()

    diff = df_replace_na==df1replace_na
    pprint(diff)
    colname = list(diff.columns.values)

    # count = diff.value_counts().loc[False]
    count = 0
    for col in colname:
        try:
            count += diff[col].value_counts().loc[False]
        except:
            print(col)
            pass
        # count += diff[col].value_counts().loc[False]
    print(count)


def demo():
    df = pd.read_csv('Missing/Aug_missing_v0_frac0.1.csv')
    nan_count = df.isna().values.sum()
    df1 = pd.read_csv('Holoclean_test_input/hospital.csv')
    nan_count_compare = df1.isna().values.sum()

    colname = list(df1.columns.values)

    df1_stack = df1.stack(dropna=False)
    na_coor_df1 = [list(x) for x in df1_stack.index[df1_stack.isna()]]
    print(len(na_coor_df1))

    df_stack = df.stack(dropna=False)
    na_coor_df = [list(x) for x in df_stack.index[df_stack.isna()]]
    print(len(na_coor_df))
    # df - df1
    diff =[x for x in na_coor_df if not x in na_coor_df1]

    print(len(diff))
    # print(len(diff))
    # res = (df equ df1)
    # pprint(res)
    # # print(list(res.columns.values))
    # # print(res.ProviderNumber.value_counts())
    # colname = list(res.columns.values)
    # print(res['Address2'])
    # print(colname)
    # count = 0
    # for col in colname:
    #     count_false = res[col].value_counts().loc[False]
    #     print(count_false)
    #     count += count_false
    # print(count)


if __name__ == '__main__':
    # main_2()
    # main_()
    # aug_mis()
    # aug_mimatch()
    # aug_mis_mim()
    # demo2()
    demo3()



