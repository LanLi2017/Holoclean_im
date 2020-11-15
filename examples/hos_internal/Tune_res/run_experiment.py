from pprint import pprint
import holoclean
import numpy as np
import glob
import os
from detect import NullDetector, ViolationDetector
from repair.featurize import *
import pandas as pd


# path = 'Missing/'
# path = 'Mismatch/'
path = 'Mixed/'
infiles = sorted(glob.glob(os.path.join(path, '*.csv')), key=os.path.getmtime)
print(infiles)

missing = [0.1,0.2,0.3,0.4]
mismatch = [0.1,0.2,0.3,0.4]
pair_fracs = []
for f in missing:
    pair_fracs.extend(list(zip([f for _ in range(4)],mismatch)))

missing_list = []
mismatch_list = []
for pair in pair_fracs:
    missing_list.append(pair[0])
    mismatch_list.append(pair[1])

eval_report = []
precision = []
recall = []
repaired_recall = []
f1 = []

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
    hc.load_dcs('constraints/hospital_constraints.txt')
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
    eval_report = hc.evaluate(fpath='Ground_truth/hospital_clean.csv',
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val')
    precision.append(eval_report.precision)
    recall.append(eval_report.recall)
    repaired_recall.append(eval_report.repair_recall)
    f1.append(eval_report.f1)

# d = {'Missing': missing, 'precision': precision, 'recall': recall, 'repaired_recall':repaired_recall, 'F1': f1}
# d = {'Mismatch': mismatch, 'precision': precision, 'recall': recall, 'repaired_recall':repaired_recall, 'F1': f1}
d = {'Mismatch': mismatch_list, 'missing': missing_list,'precision': precision, 'recall': recall, 'repaired_recall':repaired_recall, 'F1': f1}
df = pd.DataFrame(d)
df.to_csv('eval_report/Mixed-Eval_report.csv')
pprint(df)