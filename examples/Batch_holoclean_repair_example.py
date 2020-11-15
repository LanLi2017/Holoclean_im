import glob
import os
import sys
from pprint import pprint
import matplotlib.pyplot as plt
sys.path.append('../')
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *
import numpy as np
import pandas as pd
# import logging
# logging.basicConfig(filename='hospital.log',level=logging.DEBUG)



# 2. Load training data and denial constraints.
# hc.load_data('hospital', 'hos_internal/hospital.csv')
# hc.load_data('hospital', '../testdata1/Adult20.csv')
# hc.load_data('hospital', 'hos_internal/recon_hospital_clean.csv')
# hc.load_data('hospital','hos_internal/recon_hospital_100_clean.csv')
# hc.load_data('hospital', '../testdata1/hospital.csv')
frac_1 = np.arange(0.8,0.95,0.1) # [0.8, 0.9]
frac_2 = np.arange(0.05, 0.2, 0.1) # [0.05, 0.15]
pair_fracs = []
for f in frac_1:
    pair_fracs.extend(list(zip([1-f for _ in range(5)],frac_2)))
pprint(pair_fracs)
missing = []
mim = []
for pair in pair_fracs:
    missing.append(pair[0])
    mim.append(pair[1])


# plt.hist(precision)
# plt.hist(recall)
# plt.ylabel('Probability')
# plt.show()
# precision = []
# recall = []
# repair_recall = []
# f1 = []
#
# hc = holoclean.HoloClean(
#         db_name='holo',
#         domain_thresh_1=0,
#         domain_thresh_2=0,
#         weak_label_thresh=0.99,
#         max_domain=10000,
#         cor_strength=0.6,
#         nb_cor_strength=0.8,
#         epochs=10,
#         weight_decay=0.01,
#         learning_rate=0.001,
#         threads=1,
#         batch_size=1,
#         verbose=True,
#         timeout=3*60000,
#         feature_norm=False,
#         weight_norm=False,
#         print_fw=True
#     ).session
#
# hc.load_data('hospital','hos_internal/Tune_res/hos_sample_dirty_0.csv')
# hc.load_dcs('../testdata1/hospital_constraints.txt')
# # hc.load_dcs('../testdata/adult_constraints.txt')
# hc.ds.set_constraints(hc.get_dcs())
# #
# # # 3. Detect erroneous cells using these two detectors.
# detectors = [NullDetector(),ViolationDetector()]
# hc.detect_errors(detectors)
# #
# # 4. Repair errors utilizing the defined features.
# hc.setup_domain()
# featurizers = [
#     InitAttrFeaturizer(),
#     OccurAttrFeaturizer(),
#     FreqFeaturizer(),
#     ConstraintFeaturizer(),
# ]
# #
# hc.repair_errors(featurizers)
#
# # 5. Evaluate the correctness of the results.
# # hc.evaluate(fpath='../testdata/hospital_clean_recon.csv',
# # hc.evaluate(fpath='../usecase/usecase3/hospital_clean_recon.csv',
# eval_report = hc.evaluate(fpath='../usecase/usecase3/hos_clean_create.csv',
#             tid_col='tid',
#             attr_col='attribute',
#             val_col='correct_val')
# precision.append(eval_report.precision)
# recall.append(eval_report.recall)
#
# print(precision, recall)

eval_report = []
precision = []
recall = []
repaired_recall = []
f1 = []
path = 'hos_internal/Tune_res/'
# path ='../log2/'
infiles = sorted(glob.glob(os.path.join(path, '*.csv')), key=os.path.getmtime)
print(infiles)

for infile in infiles:
    # 1. Setup a HoloClean session.
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
        timeout=3*60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session
    hc.load_data('hospital',infile)
    hc.load_dcs('../testdata1/hospital_constraints.txt')
    # hc.load_dcs('../testdata/adult_constraints.txt')
    hc.ds.set_constraints(hc.get_dcs())

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector(),ViolationDetector()]
    hc.detect_errors(detectors)
    #
    # 4. Repair errors utilizing the defined features.
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        ConstraintFeaturizer(),
    ]
    #
    hc.repair_errors(featurizers)

    # 5. Evaluate the correctness of the results.
    # hc.evaluate(fpath='../testdata/hospital_clean_recon.csv',
    # hc.evaluate(fpath='../usecase/usecase3/hospital_clean_recon.csv',
    eval_report = hc.evaluate(fpath='../usecase/usecase3/hos_clean_create.csv',
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val')
    precision.append(eval_report.precision)
    recall.append(eval_report.recall)
    repaired_recall.append(eval_report.repair_recall)
    f1.append(eval_report.f1)

d = {'Missing': missing, 'Mismatched': mim, 'precision': precision, 'recall': recall, 'repaired_recall':repaired_recall, 'F1': f1}
df = pd.DataFrame(d)
df.to_csv('Missing-Eval_report.csv')
pprint(df)
