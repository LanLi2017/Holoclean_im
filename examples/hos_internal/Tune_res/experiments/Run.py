import numpy as np

import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import InitAttrFeaturizer, OccurAttrFeaturizer, FreqFeaturizer, ConstraintFeaturizer
import pandas as pd
prune_thread = np.arange(0.1,1.1,0.1)

eval_report = []
precision = []
recall = []
repaired_recall = []
f1 = []

for thread in prune_thread:
    # 1. Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=thread,
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
    hc.load_data('hospital', '../Holoclean_test_input/hospital.csv')
    hc.load_dcs('../constraints/hospital_constraints.txt')
    # hc.load_dcs('../testdata/adult_constraints.txt')
    hc.ds.set_constraints(hc.get_dcs())

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector(), ViolationDetector()]
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
    eval_report = hc.evaluate(fpath='../Ground_truth/hospital_clean.csv',
                              tid_col='tid',
                              attr_col='attribute',
                              val_col='correct_val')
    precision.append(eval_report.precision)
    recall.append(eval_report.recall)
    repaired_recall.append(eval_report.repair_recall)
    f1.append(eval_report.f1)

d = {'prune_threshold': prune_thread, 'precision': precision, 'recall': recall, 'repaired_recall':repaired_recall, 'F1': f1}
# d = {'Mismatch': mismatch_list, 'missing': missing_list,'precision': precision, 'recall': recall, 'repaired_recall':repaired_recall, 'F1': f1}
df = pd.DataFrame(d)
df.to_csv('../eval_report/tune_prune_coocurrance.csv')
# df.to_csv('eval_report/Mixed-Eval_report.csv')
print(df)
