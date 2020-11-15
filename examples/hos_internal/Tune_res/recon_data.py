import csv
from csv import DictReader
from itertools import groupby
from operator import itemgetter

from io import StringIO
from pprint import pprint

import pandas as pd
from dateutil.parser import parse as parse_datetime


def guess_type(value):
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    try:
        return parse_datetime(value)
    except (ValueError, OverflowError):
        pass

    return value # Fallback to string


# file_content = """
# tid,attribute,correct_val
# 0,id,21597594
# 0,name_grel_star,Eclectic
# 0,host_id,100952192
# 0,host_name,Diana
# 1,id,222222
# 1,name_grel_star,room
# """.strip()

def main():
    # file_p = '../../testdata1/hospital_clean_recon.csv'
    file_p = 'recon_data/hospital_clean.csv'
    # with open(file_p) as csvf:
    #     file_content = csvf.read().strip('\n')
    #
    # with StringIO(file_content, newline='')as f:
    #     reader = DictReader(f)
    #     data = list(reader)
    with open(file_p,'rt',newline='',encoding='utf-8')as f:
        reader = DictReader(f)
        data = list(reader)
    data.sort(key=itemgetter('tid'))
    table = []
    for _,records in groupby(data, itemgetter('tid')):
        table.append({
            record['attribute']: record['correct_val']
            for record in records
        })

    df = pd.DataFrame(table)
    # for pn in df['ProviderNumber']:
    #     str(pn)

    df.to_csv('Ground_truth/hospital_clean_recon.csv', index=False)
    # df.to_csv('recon_hospital_100_clean.csv', index=False)


if __name__ == '__main__':
    main()

