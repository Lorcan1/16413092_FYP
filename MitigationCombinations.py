import sys

sys.path.insert(1, "../")

import pandas as pd
import numpy as np

np.random.seed(0)

import numpy
import random

import pprint
import json
import traceback
from collections import defaultdict

from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# from aif360.algorithms.inprocessing import AdversarialDebiasing

from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, Reweighing
from aif360.algorithms.inprocessing import MetaFairClassifier, PrejudiceRemover, GerryFairClassifier, \
    ExponentiatedGradientReduction
from grid_search_reduction_fixed import GridSearchReduction
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, \
    RejectOptionClassification

from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, \
    load_preproc_data_german, load_preproc_data_compas
from aif360.datasets import BinaryLabelDataset
import time
import gc

import queue
import itertools

from multiprocessing import Process, Queue, Pool
import multiprocessing

# Globals for the datasets
bank_train = None
bank_test = None
bank_valid = None

adult_train = None
adult_test = None
adult_valid = None

german_train = None
german_test = None
german_valid = None

compas_train = None
compas_test = None
compas_valid = None

ricci_train = None
ricci_test = None
ricci_valid = None


def set_default(obj):
    # print(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def log_error(stage, exc_type, exc_value, exc_traceback):
    if len(sys.argv) > 1:
        name = sys.argv[1] + '.log'
    else:
        name = "StandardErrorLog.log"
    f = open(name, "a+")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    f.write('[' + current_time + '] ' + stage + '\n')
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    f.close()


def log(message, dictionary=False):
    if len(sys.argv) > 1:
        name = sys.argv[1] + '.log'
    else:
        name = "StandardLog.log"
    f = open(name, "a+")
    if (dictionary):
        message.pop('dataset', None)
        message.pop('dataset_test', None)
        message.pop('dataset_valid', None)
        # print(message)
        print(json.dumps(str(message), default=set_default), file=f)
    else:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        f.write('[' + current_time + '] ' + message + '\n')
    f.close()


def top_single(my_list):  # not parrallelized function
    counter = 0
    a = []

    for i in my_list:
        log('Here is the row: ' + str(i))
        log('Here is the counter ' + str(counter))

        dataset_name = ""
        if i[0] == 0:
            dataset_name = "Bank"
        elif i[0] == 1 or i[0] == 2:
            dataset_name = "Adult"
        elif i[0] == 3 or i[0] == 4:
            dataset_name = "German"
        elif i[0] == 5 or i[0] == 6:
            dataset_name = "Compas"
        else:
            dataset_name = "Ricci"
        print("Now looking at dataset: ", dataset_name)

        top_data_d = data_f(i[0])  # main pipeline of bma functions called here
        top_pre_d = pre(i[1], top_data_d)

        passed = 0

        log(str(i) + ' pre complete')
        if (sanity_check(top_pre_d)):
            passed += 1

            try:
                # log('throwing error')
                # raise TypeError('somthing')
                top_in_d = in_p(i[2], top_pre_d)
                top_class_d = classifier(i[3], top_in_d)

                log(str(i) + ' in/class complete')

                if (sanity_check(top_class_d)):
                    passed += 1
                    try:
                        top_post_d = post(i[4], top_class_d)  # bias mitigation functions
                        log(str(i) + ' post complete')
                        top_sort_d = sorter(top_post_d)
                        passed += 1
                    except:
                        log('caught error in post')
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        log_error('post', exc_type, exc_value, exc_traceback)
                        top_sort_d = resolve_failed(top_class_d, 'post', i)
                else:
                    log('Sanity Check Failed in in/class' + str(i))
                    top_sort_d = resolve_failed(top_class_d, 'in/class', i)
            except:
                log('caught error in in/class')
                exc_type, exc_value, exc_traceback = sys.exc_info()
                log_error('in/class', exc_type, exc_value, exc_traceback)
                top_sort_d = resolve_failed(top_pre_d, 'in/class', i)

        else:
            log('Sanity Check Failed in pre' + str(i))
            top_sort_d = resolve_failed(top_pre_d, 'pre', i)

        # print(top_sort_d)
        log(message=top_sort_d, dictionary=True)

        a.append(top_sort_d)
    return a


def top(my_list, out_queue):  # parrallelized function
    counter = 0
    a = []

    for i in my_list:
        log('Here is the row: ' + str(i))
        log('Here is the counter ' + str(counter))

        dataset_name = ""
        if i[0] == 0:
            dataset_name = "Bank"
        elif i[0] == 1 or i[0] == 2:
            dataset_name = "Adult"
        elif i[0] == 3 or i[0] == 4:
            dataset_name = "German"
        elif i[0] == 5 or i[0] == 6:
            dataset_name = "Compas"
        else:
            dataset_name = "Ricci"
        print("Now looking at dataset: ", dataset_name)

        #        if i[0] == 0:
        #            dataset = bank_dataset
        #        elif i[0] == 1:
        #            dataset =  adult_dataset
        #        elif i[0] == 2:
        #            dataset = adult_dataset
        #        elif i[0] == 3:
        #            dataset = german_dataset
        #        elif i[0] == 4:
        #            dataset = german_dataset
        #        elif i[0] == 5:
        #            dataset = compas_dataset
        #        elif i[0] == 6:
        #            dataset = compas_dataset
        #        elif i[0] == 7:
        #            dataset = ricci_dataset

        top_data_d = data_f(i[0])  # main pipeline of bma functions called here
        top_pre_d = pre(i[1], top_data_d)

        passed = 0

        log(str(i) + ' pre complete')
        if (sanity_check(top_pre_d)):
            passed += 1

            try:
                # log('throwing error')
                # raise TypeError('somthing')
                top_in_d = in_p(i[2], top_pre_d)
                top_class_d = classifier(i[3], top_in_d)

                log(str(i) + ' in/class complete')

                if (sanity_check(top_class_d)):
                    passed += 1
                    try:
                        top_post_d = post(i[4], top_class_d)  # bias mitigation functions
                        log(str(i) + ' post complete')
                        top_sort_d = sorter(top_post_d)
                        passed += 1
                    except:
                        log('caught error in post')
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        log_error('post', exc_type, exc_value, exc_traceback)
                        top_sort_d = resolve_failed(top_class_d, 'post', i)
                else:
                    log('Sanity Check Failed in in/class' + str(i))
                    top_sort_d = resolve_failed(top_class_d, 'in/class', i)
            except:
                log('caught error in in/class')
                exc_type, exc_value, exc_traceback = sys.exc_info()
                log_error('in/class', exc_type, exc_value, exc_traceback)
                top_sort_d = resolve_failed(top_pre_d, 'in/class', i)

        else:
            log('Sanity Check Failed in pre' + str(i))
            top_sort_d = resolve_failed(top_pre_d, 'pre', i)

        # print(top_sort_d)
        log(message=top_sort_d, dictionary=True)

        out_queue.put(top_sort_d)  # result returned through queue
        # log(pprint.pformat(top_sort_d))

        top_sort_d.clear()
        top_data_d.clear()
        top_pre_d.clear()

        if passed > 1:
            top_in_d.clear()
            top_class_d.clear()

        if passed > 2:
            top_post_d.clear()

        top_data_d = None
        top_pre_d = None
        top_in_d = None
        top_class_d = None
        top_post_d = None
        top_sort_d = None

        counter = counter + 1


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def multi_run_wrapper(args):
    return top(*args)


def main(runParallel=False, random_state=None, run_name="Run0", stratified_sampling=True):
    # setting up the datasets
    bank_dataset = BankDataset()
    adult_dataset = AdultDataset()
    german_dataset = GermanDataset()
    compas_dataset = CompasDataset()
    df = pd.read_csv('ricci.csv')

    df = df.replace('B', 0)
    df = df.replace('H', 0)
    df = df.replace('W', 1)
    df = df.replace('Captain', 1)
    df = df.replace('Lieutenant', 0)

    ricci_dataset = BinaryLabelDataset(favorable_label=1,
                                       unfavorable_label=0,
                                       df=df,
                                       label_names=['Promotion'],
                                       protected_attribute_names=['Race'],
                                       unprivileged_protected_attributes=['0'])

    testSize = .4
    if (len(sys.argv) < 3):
        stratified = stratified_sampling
    else:
        stratified = sys.argv[2].lower()

    log('Stratified sampling enabled: ' + str(stratified))

    global bank_train, bank_test, bank_valid
    global adult_train, adult_test, adult_valid
    global german_train, german_test, german_valid
    global compas_train, compas_test, compas_valid
    global ricci_train, ricci_test, ricci_valid

    bank_train, bank_vt = sample_data(bank_dataset, testSize, stratified, random_state)
    bank_test, bank_valid = sample_data(bank_vt, .5, stratified, random_state)

    adult_train, adult_vt = sample_data(adult_dataset, testSize, stratified, random_state)
    adult_test, adult_valid = sample_data(adult_vt, .5, stratified, random_state)

    german_train, german_vt = sample_data(german_dataset, testSize, stratified, random_state)
    german_test, german_valid = sample_data(german_vt, .5, stratified, random_state)

    compas_train, compas_vt = sample_data(compas_dataset, testSize, stratified, random_state)
    compas_test, compas_valid = sample_data(compas_vt, .5, stratified, random_state)

    ricci_train, ricci_vt = sample_data(ricci_dataset, testSize, stratified, random_state)
    ricci_test, ricci_valid = sample_data(ricci_vt, .5, stratified, random_state)

    # for the German dataset, we need to re-set the labels to work properly with all mitigation strategies
    german_train.labels = german_train.labels - 1
    german_valid.labels = german_valid.labels - 1
    german_test.labels = german_test.labels - 1

    german_train.favorable_label = german_train.favorable_label - 1
    german_valid.favorable_label = german_valid.favorable_label - 1
    german_test.favorable_label = german_test.favorable_label - 1

    german_train.unfavorable_label = german_train.unfavorable_label - 1
    german_valid.unfavorable_label = german_valid.unfavorable_label - 1
    german_test.unfavorable_label = german_test.unfavorable_label - 1

    # 1 = dataset
    # 2 = pre
    # 3= in_p
    # 4 = class
    # 5 = post

    a_list = []

    dataset, pre_proc, in_proc, class_proc, post_proc = 8, 3, 4, 4, 4
    # dataset, pre_proc, in_proc, class_proc, post_proc = 2, 0, 0, 2, 0
    # data, pre_p, in_p, class_p, post_p = 8, 1, 1, 2, 1
    l = [(a, b, c, d, e) for a in range(dataset) for b in range(pre_proc) for c in range(in_proc) for d in
         range(class_proc) for e in range(post_proc)]  # create list of inputs

    for x in l:
        if (x[2] == 0 and x[3] != 0) or (x[2] != 0 and x[3] == 0):  # cant have in-processing and classifier
            #            if x[4] is not 3:                                            #ROC was calculated seperatedly as it is a memory hog
            # if (x[2] == 1) or (x[2] == 2):  # run only MFC
            a_list.append(x)

#    a_list = [(3, 0, 0, 2, 0), (3, 0, 1, 0, 1), (3, 0, 2, 0, 1)]
    #    a_list = [(3, 0, 1, 0, 0)]

    print(a_list)
    if runParallel:
        # upper bound to n cores -- added by Simon to be nice on SONIC
        if (len(sys.argv) < 4):
            num_proc = 12
        else:
            num_proc = int(sys.argv[3])

        # return number of cores present on machine
        cpu_num = int(max(1, min(num_proc, multiprocessing.cpu_count()) / 4))
        log('Using ' + str(cpu_num) + ' cores')

        # randomly shuffle  input list before splitting to achieve a more equal runtime during parallelization
        random.shuffle(a_list)

        # split input array
        five = numpy.array_split(numpy.array(a_list), cpu_num)

        m = multiprocessing.Manager()
        processes = []
        out_queue0 = m.Queue()
        numb_list = []
        counter = 0

        for x in five:
            numb_list.append((five[counter]))
            counter = counter + 1

        input_list = list(zip(numb_list, itertools.repeat(out_queue0)))

        pool = Pool(cpu_num)
        pool.map(multi_run_wrapper, input_list)

        result = []

        while out_queue0.qsize() != 0:  # prevent deadlock
            result.append(out_queue0.get())

        pool.close()
    else:
        print('starting sequential processing')
        result = top_single(a_list)
        print('sequential processing stopped')
    dfTemp0 = append_func(result)  # append results to dataframe
    print(dfTemp0)
    dfFinal0, dfFinal1 = df_sort(dfTemp0)  # clean dataframe, second dataframe is identical bar being ranked differently
    output(dfFinal0, dfFinal1, run_name, stratified)

    return None


def df_sort(dataframe):  # all dataframe cleaning handled here
    dataframe = df_format(dataframe)  # fix bma names
    dataframe = df_orig_value(dataframe)  # return orig fairness values for theil, av and eop
    dataframe = fair_checker(dataframe)  # return score column

    dataframe1 = dataframe.copy()
    dataframe = rank_df0(dataframe)
    dataframe1 = rank_df1(
        dataframe1)  # rank_df` is merged excpet overall performance and fairness rank calculated sep. before joining

    temp = ovr_rank_df(dataframe)
    temp2 = ovr_rank_df(dataframe1)

    dfFinal = merge0(temp, dataframe)
    dfFinal1 = merge1(temp2, dataframe1)
    return dfFinal, dfFinal1


def df_orig_value(dataframe):  # appends unmitgated score to equivlaent mitigated row
    orig_score_df = dataframe[
        (dataframe['Pre'] == '-') & (dataframe['In_p'] == '-') & (dataframe['Post'] == '-')]  # find unmitigated rows
    orig_score_df = orig_score_df[
        ['Theil Index', 'Average Odds Difference', 'Equal Opportunity Difference', 'Classifier', 'Dataset',
         'Sens_Attr']]
    orig_score_df = orig_score_df.rename(columns={"Theil Index": "Orig Theil",
                                                  "Average Odds Difference": "Orig Av Odds",
                                                  "Equal Opportunity Difference": "Orig Eq Opp Diff"})
    dataframe.loc[dataframe.In_p != '-', 'Classifier'] = "Logistic Regression"
    new_df = pd.merge(dataframe, orig_score_df, how='left', on=['Classifier', 'Dataset', 'Sens_Attr'])
    new_df.loc[new_df.In_p != '-', 'Classifier'] = "-"

    return new_df


def merge0(dataframe1,
           dataframe2):  # mereges temp dataframe containing rank averaged across datasets, with original datafame
    new_df = pd.merge(dataframe1, dataframe2, how='left', on=['Pre', 'In_p', 'Post', 'Classifier'])
    tempdf = new_df.pop('Ovr_Rank')
    new_df['Ovr_Rank'] = tempdf
    new_df['Rank'] = new_df.groupby('Dataset')['Rank'].rank(method='min', ascending=True)
    new_df = new_df.sort_values(by=['Dataset', 'Rank'])

    return new_df


def merge1(dataframe1, dataframe2):  # similar to above but with perf-rank and fair-rank

    new_df = pd.merge(dataframe1, dataframe2, how='left', on=['Pre', 'In_p', 'Post', 'Classifier'])
    new_df['Ovr_Rank'] = (new_df['Perf_Rank'] + new_df['Fair_Rank']) / 2
    new_df['Ovr_Rank'] = new_df.groupby('Dataset')['Ovr_Rank'].rank(method='min', ascending=True)
    new_df = new_df.rename(columns={"Rank_y": "Rank"})
    new_df['Rank'] = new_df.groupby('Dataset')['Rank'].rank(method='min', ascending=True)
    new_df = new_df.sort_values(by=['Dataset', 'Rank'])

    return new_df


def data_f(data_used):
    if data_used == 0:
        nam = 'Bank'
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig_train = bank_train
        dataset_orig_test = bank_test
        dataset_orig_valid = bank_valid
    elif data_used == 1:
        nam = 'Adult'
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig_train = adult_train
        dataset_orig_test = adult_test
        dataset_orig_valid = adult_valid
    elif data_used == 2:
        nam = 'Adult'
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig_train = adult_train
        dataset_orig_test = adult_test
        dataset_orig_valid = adult_valid
    elif data_used == 3:
        nam = 'German'
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig_train = german_train
        dataset_orig_test = german_test
        dataset_orig_valid = german_valid
    elif data_used == 4:
        nam = 'German'
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig_train = german_train
        dataset_orig_test = german_test
        dataset_orig_valid = german_valid
    elif data_used == 5:
        nam = 'Compas'
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig_train = compas_train
        dataset_orig_test = compas_test
        dataset_orig_valid = compas_valid
    elif data_used == 6:
        nam = 'Compas'
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig_train = compas_train
        dataset_orig_test = compas_test
        dataset_orig_valid = compas_valid
    elif data_used == 7:
        nam = 'Ricci'
        privileged_groups = [{'Race': 1}]
        unprivileged_groups = [{'Race': 0}]
        dataset_orig_train = ricci_train
        dataset_orig_test = ricci_test
        dataset_orig_valid = ricci_valid

    sens = str(privileged_groups[0])
    sens = sens.split(":")
    sens[0] = sens[0].replace("'", ' ')
    sens[0] = sens[0].replace("{", ' ')  # returns privileged group as a string
    sens[0] = sens[0].strip()

    sens = sens[0]

    start = 0
    start = time.perf_counter()  # begin timer

    data_d = {'dataset': dataset_orig_train,
              'dataset_test': dataset_orig_test,
              'dataset_valid': dataset_orig_valid,
              'dataset_used': nam,
              'privileged_groups': privileged_groups,
              'unprivileged_groups': unprivileged_groups,
              'sens': sens,
              'start': start}

    dataset = None
    dataset_orig_train = None
    dataset_orig_test = None
    nam = None
    privileged_groups = None
    unprivileged_groups = None
    sens = None
    start = None
    data_used = None
    pro_used = None

    return data_d


def two_dimensional_stratified_sample(dataset, testSize, random_state=None):
    df_conv, _ = dataset.convert_to_dataframe()
    y = dataset.label_names[0]
    protected = dataset.protected_attribute_names[0]

    strata = df_conv[y].astype(str) + "-" + df_conv[protected].astype(str)

    y = df_conv.pop(y)

    X_train, X_test, y_train, y_test = train_test_split(df_conv, y, test_size=testSize, stratify=strata,
                                                        random_state=random_state)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    trainAIF = BinaryLabelDataset(favorable_label=dataset.favorable_label,
                                  unfavorable_label=dataset.unfavorable_label,
                                  df=train,
                                  label_names=dataset.label_names,
                                  protected_attribute_names=dataset.protected_attribute_names,
                                  unprivileged_protected_attributes=dataset.unprivileged_protected_attributes)

    testAIF = BinaryLabelDataset(favorable_label=dataset.favorable_label,
                                 unfavorable_label=dataset.unfavorable_label,
                                 df=test,
                                 label_names=dataset.label_names,
                                 protected_attribute_names=dataset.protected_attribute_names,
                                 unprivileged_protected_attributes=dataset.unprivileged_protected_attributes)

    return trainAIF, testAIF


def stratified_sample(dataset, testSize, random_state=None):
    df_conv, _ = dataset.convert_to_dataframe()
    y = dataset.label_names[0]
    y = df_conv.pop(y)
    X_train, X_test, y_train, y_test = train_test_split(df_conv, y, test_size=testSize, stratify=y,
                                                        random_state=random_state)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    trainAIF = BinaryLabelDataset(favorable_label=dataset.favorable_label,
                                  unfavorable_label=dataset.unfavorable_label,
                                  df=train,
                                  label_names=dataset.label_names,
                                  protected_attribute_names=dataset.protected_attribute_names,
                                  unprivileged_protected_attributes=dataset.unprivileged_protected_attributes)

    testAIF = BinaryLabelDataset(favorable_label=dataset.favorable_label,
                                 unfavorable_label=dataset.unfavorable_label,
                                 df=test,
                                 label_names=dataset.label_names,
                                 protected_attribute_names=dataset.protected_attribute_names,
                                 unprivileged_protected_attributes=dataset.unprivileged_protected_attributes)

    return trainAIF, testAIF


def sample_data(dataset, testSize, stratified=True, random_state=None):
    if (stratified):
        return two_dimensional_stratified_sample(dataset, testSize, random_state)
    else:
        return dataset.split([1 - testSize], shuffle=True, seed=random_state)


def get_total_finish(fin_dict):  # keeps track of total time for each combination

    total_finish = fin_dict.get('total_finish')
    if total_finish is None:
        total_finish = 0
    else:
        total_finish = total_finish
    fin_dict = None

    return total_finish


def namer(bma_a, name_dict, name):  # combines names for rows with multiple bmas

    if bma_a is None:
        bma_a = []
        bma_a.append(name)
    else:
        bma_a.append(name)

    if name_dict.get('count') is None:
        count = 1
        bma_a.append(count)
        count += 1

    else:
        count = name_dict.get('count')
        bma_a.append(count)
        count += 1

    name_dict = None
    name = None

    return bma_a, count


def sanity_check(p_dict):
    # check if the previous part of the pipeline broke the training data
    data = p_dict['dataset']

    fail = True
    y = data.labels

    if (len(np.unique(y)) == 1):
        fail = False

    return fail


def resolve_failed(p_dict, stage, i):
    results = {'pre': lookup_pre(i[1]),
               'in': lookup_in(i[2]),
               'post': lookup_post(i[4]),
               'mean_diff': None,
               'dis_impact': None,
               'theil': None,
               'av_odds': None,
               'eq_opp_diff': None,
               'mean_diff_orig': None,
               'dis_impact_orig': None,
               'acc': None,
               'prec': None,
               'rec': None,
               'auc': None,
               'class_name': lookup_class(i[3]),
               'data_used_name': p_dict['dataset_used'],
               'sens_name': p_dict['sens'],
               'acc_check': 'Fail',
               't': {0},
               'failed': stage}

    return results


def lookup_pre(i):
    pre = []

    if i == 1:
        pre.append('di')
        pre.append(i)
    elif i == 2:
        pre.append('rw')
        pre.append(i)
    else:
        return None

    return pre


def lookup_in(i):
    in_p = []

    if i == 1:
        # in_p.append('mfc_sr')
        in_p.append('egr_fp')
        in_p.append(i)
    elif i == 2:
        # in_p.append('mfc_fdr')
        in_p.append('gsr_eo')
        in_p.append(i)
    elif i == 3:
        in_p.append('pr')
        in_p.append(i)
    else:
        return None

    return in_p


def lookup_class(i):
    if i == 1:
        return 'Logistic Regression'
    elif i == 2:
        return 'Random Forest'
    elif i == 3:
        return 'Naive Bayes'
    else:
        return None


def lookup_post(i):
    post = []
    if i == 1:
        post.append('cpp')
        post.append(i)
    elif i == 2:
        post.append('eop')
        post.append(i)
    elif i == 3:
        post.append('roc')
        post.append(i)
    else:
        return None

    return post


def pre(bma, p_dict):  # applies pre-processing BMA

    data = p_dict['dataset']
    data_used = p_dict['dataset_used']
    unprivileged_groups = p_dict['unprivileged_groups']
    privileged_groups = p_dict['privileged_groups']
    sens = p_dict['sens']
    start = p_dict['start']

    dataset_test = p_dict['dataset_test']
    dataset_valid = p_dict['dataset_valid']

    if bma == 1:
        DI = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=sens)
        dataset_t = DI.fit_transform(data)
        nam = 'di'
        DI = None
    elif bma == 2:
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_t = RW.fit_transform(data)
        nam = 'rw'
        RW = None
    elif bma == 0:  # no-preprcossing
        return p_dict

    finish = 0
    finish = time.perf_counter()
    total_finish = get_total_finish(p_dict)

    total_finish = total_finish + finish

    pre_a = p_dict.get('pre')
    post = p_dict.get("post")
    in_p = p_dict.get('in')
    name = p_dict.get('class')

    pre_a, count = namer(pre_a, p_dict, nam)

    p_dict = None

    pre_d = {'dataset': dataset_t,
             'dataset_used': data_used,
             'privileged_groups': privileged_groups,
             'unprivileged_groups': unprivileged_groups,
             'pre': pre_a,
             'in': in_p,
             'post': post,
             'count': count,
             'sens': sens,
             'dataset_test': dataset_test,
             'dataset_valid': dataset_valid,
             'class': name,
             'start': start,
             'finish': total_finish}

    data = None
    dataset_t = None
    data_used = None
    privileged_groups = None
    unprivileged_groups = None
    pre_a = None
    in_p = None
    post = None
    count = None
    sens = None
    name = None
    start = None
    finish = None
    total_finish = None
    dataset_test = None
    dataset_valid = None
    bma = None
    nam = None

    return pre_d


def in_p(bma, in_dict):  # applies in-processing classifier

    data = in_dict['dataset']
    data_used = in_dict['dataset_used']
    unprivileged_groups = in_dict['unprivileged_groups']
    privileged_groups = in_dict['privileged_groups']
    sens = in_dict['sens']
    dataset_test = in_dict['dataset_test']
    dataset_valid = in_dict['dataset_valid']
    start = in_dict['start']

    if bma == 1:
        print("Running ExponentiatedGradientReduction")
        # MFC = MetaFairClassifier(tau=0, sensitive_attr= sens, type = 'sr')
        # MFC = MetaFairClassifier(tau=0.8, sensitive_attr= sens, type = 'sr')
        # MFC = MFC.fit(data)
        #
        # data_pred_valid = MFC.predict(dataset_valid)
        # data_pred = MFC.predict(dataset_test)
        # C = 10
        # print_flag = False
        # gamma = .01
        # max_iterations = 10
        #
        # GF = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
        #      max_iters=max_iterations, heatmapflag=False)
        # GF = GF.fit(data, early_termination=True)
        #
        # data_pred_valid = GF.predict(dataset_valid)
        # data_pred = GF.predict(dataset_test)

        estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
        constraint = "EqualizedOdds"
        EGR = ExponentiatedGradientReduction(estimator=estimator, constraints=constraint)
        EGR = EGR.fit(data)

        data_pred_valid = EGR.predict(dataset_valid)
        data_pred = EGR.predict(dataset_test)

        pred = data_pred.labels
        pred_prob = data_pred.scores
        pred_valid = data_pred_valid.labels
        nam = 'egr_fp'
        EGR = None
    if bma == 2:
        print("Running GridSearchReduction")
        # MFC2 = MetaFairClassifier(tau=0, sensitive_attr= sens, type = 'fdr')
        # MFC2 = MetaFairClassifier(tau=0.8, sensitive_attr= sens, type = 'fdr')
        # MFC2 = MFC2.fit(data)
        #
        # data_pred_valid = MFC2.predict(dataset_valid)
        # data_pred = MFC2.predict(dataset_test)

        # EGR = ExponentiatedGradientReduction()
        # EGR = EGR.fit(data)
        #
        # data_pred_valid = EGR.predict(dataset_valid)
        # data_pred = EGR.predict(dataset_test)
        estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
        GSR = GridSearchReduction(estimator=estimator, constraints="EqualizedOdds")
        GSR = GSR.fit(data)

        data_pred_valid = GSR.predict(dataset_valid)
        data_pred = GSR.predict(dataset_test)

        pred = data_pred.labels

        pred_prob = data_pred.scores
        pred_valid = data_pred_valid.labels
        nam = 'gsr_eo'
        GSR = None
    elif bma == 3:
        print("Running PrejudiceRemover")
        # PR = PrejudiceRemover(sensitive_attr= sens, eta=25.0)
        PR = PrejudiceRemover(sensitive_attr=sens, eta=1.0)
        PR = PR.fit(data)

        data_pred_valid = PR.predict(dataset_valid)
        data_pred = PR.predict(dataset_test)

        pred_prob = data_pred.scores
        pred_prob_val = data_pred_valid.scores

        pred_thres = 0.5
        y_valid_pred = np.zeros_like(data_pred_valid.labels)
        y_valid_pred[pred_prob_val >= pred_thres] = data_pred_valid.favorable_label
        y_valid_pred[~(pred_prob_val >= pred_thres)] = data_pred_valid.unfavorable_label
        data_pred_valid.labels = y_valid_pred

        y_test_pred = np.zeros_like(data_pred.labels)
        y_test_pred[pred_prob >= pred_thres] = data_pred.favorable_label
        y_test_pred[~(pred_prob >= pred_thres)] = data_pred.unfavorable_label
        data_pred.labels = y_test_pred

        pred = data_pred.labels
        pred_valid = data_pred_valid.labels
        nam = 'pr'
        PR = None
    elif bma == 0:
        return in_dict

    finish = 0
    finish = time.perf_counter()
    total_finish = get_total_finish(in_dict)
    total_finish = total_finish + finish

    pre = in_dict.get('pre')
    in_p_a = in_dict.get('in_p_a')
    post = in_dict.get('post')
    name = in_dict.get('class')

    in_p_a, count = namer(in_p_a, in_dict, nam)

    classified_metric = class_met(dataset_test, data_pred, unprivileged_groups, privileged_groups)

    in_dict = None

    in_d = {'dataset': data_pred,
            'dataset_used': data_used,
            'privileged_groups': privileged_groups,
            'unprivileged_groups': unprivileged_groups,
            'pre': pre,
            'in': in_p_a,
            'post': post,
            'count': count,
            'sens': sens,
            'dataset_test': dataset_test,
            'dataset_valid': dataset_valid,
            'data_pred': data_pred,
            'data_pred_valid': data_pred_valid,
            'pred': pred,
            'pred_prob': pred_prob,
            'pred_valid': pred_valid,
            'class': name,
            'class_met': classified_metric,
            'start': start,
            'finish': total_finish}

    data = None
    data_used = None
    privileged_groups = None
    unprivileged_groups = None
    pre = None
    in_p_a = None
    post = None
    count = None
    sens = None
    dataset_test = None
    dataset_valid = None
    name = None
    start = None
    total_finish = None
    data_pred = None
    data_pred_valid = None
    pred = None
    pred_valid = None
    classified_metric = None
    bma = None
    finish = None
    nam = None

    return in_d


def classifier(clss, class_dict):
    data = class_dict['dataset']
    data_used = class_dict['dataset_used']
    unprivileged_groups = class_dict['unprivileged_groups']
    privileged_groups = class_dict['privileged_groups']
    sens = class_dict['sens']
    dataset_test = class_dict['dataset_test']
    dataset_valid = class_dict['dataset_valid']
    start = class_dict['start']

    pred_thres = .5
    data_pred = dataset_test.copy(deepcopy=True)
    data_pred_valid = dataset_valid.copy(deepcopy=True)

    if clss == 1:

        lr = LogisticRegression()
        lr = lr.fit(data.features, data.labels.ravel(),
                    sample_weight=data.instance_weights)  # fitted on train(transformed) datatset

        pred_valid = lr.predict(dataset_valid.features).reshape(-1, 1)
        pred = lr.predict(dataset_test.features).reshape(-1, 1)
        pos_ind = np.where(lr.classes_ == data.favorable_label)[0][0]
        pred_prob = lr.predict_proba(dataset_test.features)[:, pos_ind].reshape(-1, 1)
        pred_prob_val = lr.predict_proba(dataset_valid.features)[:, pos_ind].reshape(-1, 1)

        data_pred_valid.scores = pred_prob_val
        data_pred.scores = pred_prob

        y_valid_pred = np.zeros_like(data_pred_valid.labels)
        y_valid_pred[pred_prob_val >= pred_thres] = data_pred_valid.favorable_label
        y_valid_pred[~(pred_prob_val >= pred_thres)] = data_pred_valid.unfavorable_label
        data_pred_valid.labels = y_valid_pred

        y_test_pred = np.zeros_like(data_pred.labels)
        y_test_pred[pred_prob >= pred_thres] = data_pred.favorable_label
        y_test_pred[~(pred_prob >= pred_thres)] = data_pred.unfavorable_label
        data_pred.labels = y_test_pred

        name = 'Logistic Regression'
        lr = None
    elif clss == 2:
        rf = RandomForestClassifier(n_estimators=100,
                                    max_features='sqrt')
        rf = rf.fit(data.features, data.labels.ravel(),
                    sample_weight=data.instance_weights)  # fitted on train(transformed) datatset

        pred_valid = rf.predict(dataset_valid.features).reshape(-1, 1)
        pos_ind = np.where(rf.classes_ == data.favorable_label)[0][0]
        pred_prob = rf.predict_proba(dataset_test.features)[:, pos_ind].reshape(-1, 1)
        pred = rf.predict(dataset_test.features).reshape(-1, 1)
        pred_prob_val = rf.predict_proba(dataset_valid.features)[:, pos_ind].reshape(-1, 1)

        data_pred_valid.scores = pred_prob_val
        data_pred.scores = pred_prob

        y_valid_pred = np.zeros_like(data_pred_valid.labels)
        y_valid_pred[pred_prob_val >= pred_thres] = data_pred_valid.favorable_label
        y_valid_pred[~(pred_prob_val >= pred_thres)] = data_pred_valid.unfavorable_label
        data_pred_valid.labels = y_valid_pred

        y_test_pred = np.zeros_like(data_pred.labels)
        y_test_pred[pred_prob >= pred_thres] = data_pred.favorable_label
        y_test_pred[~(pred_prob >= pred_thres)] = data_pred.unfavorable_label
        data_pred.labels = y_test_pred
        name = 'Random Forest'
        rf = None
    elif clss == 3:
        nb = GaussianNB()
        nb = nb.fit(data.features, data.labels.ravel(),
                    sample_weight=data.instance_weights)  # fitted on train(transformed) datatset

        pred_valid = nb.predict(dataset_valid.features).reshape(-1, 1)
        pos_ind = np.where(nb.classes_ == data.favorable_label)[0][0]
        pred_prob = nb.predict_proba(dataset_test.features)[:, pos_ind].reshape(-1, 1)
        pred = nb.predict(dataset_test.features).reshape(-1, 1)
        pred_prob_val = nb.predict_proba(dataset_valid.features)[:, pos_ind].reshape(-1, 1)

        data_pred_valid.scores = pred_prob_val
        data_pred.scores = pred_prob

        y_valid_pred = np.zeros_like(data_pred_valid.labels)
        y_valid_pred[pred_prob_val >= pred_thres] = data_pred_valid.favorable_label
        y_valid_pred[~(pred_prob_val >= pred_thres)] = data_pred_valid.unfavorable_label
        data_pred_valid.labels = y_valid_pred

        y_test_pred = np.zeros_like(data_pred.labels)
        y_test_pred[pred_prob >= pred_thres] = data_pred.favorable_label
        y_test_pred[~(pred_prob >= pred_thres)] = data_pred.unfavorable_label
        data_pred.labels = y_test_pred
        name = 'Naive Bayes'
        nb = None

    elif clss == 0:
        return class_dict

    finish = 0
    finish = time.perf_counter()

    total_finish = get_total_finish(class_dict)
    total_finish = total_finish + finish

    pre = class_dict.get('pre', None)
    in_p = class_dict.get('in', None)
    post = class_dict.get('post', None)
    count = class_dict.get('count')

    classified_metric = class_met(dataset_test, data_pred, unprivileged_groups, privileged_groups)

    class_dict = None

    class_d = {'dataset': data,
               'dataset_used': data_used,
               'privileged_groups': privileged_groups,
               'unprivileged_groups': unprivileged_groups,
               'pre': pre,
               'in': in_p,
               'post': post,
               'count': count,
               'sens': sens,
               'dataset_test': dataset_test,
               'dataset_valid': dataset_valid,
               'data_pred': data_pred,
               'data_pred_valid': data_pred_valid,
               'pred': y_test_pred,
               'pred_prob': pred_prob,
               'class': name,
               'class_met': classified_metric,
               'start': start,
               'finish': total_finish}

    data = None
    data_used = None
    privileged_groups = None
    unprivileged_groups = None
    pre = None
    in_p = None
    post = None
    count = None
    sens = None
    dataset_test = None
    dataset_valid = None
    name = None
    start = None
    total_finish = None
    data_pred = None
    data_pred_valid = None
    pred = None
    classified_metric = None
    clss = None
    finish = None

    return class_d


def post(bma, post_dict):  # applies post-processing algorithms

    data = post_dict['dataset']
    data_used = post_dict['dataset_used']
    unprivileged_groups = post_dict['unprivileged_groups']
    privileged_groups = post_dict['privileged_groups']
    sens = post_dict['sens']
    dataset_test = post_dict['dataset_test']
    dataset_valid = post_dict['dataset_valid']
    data_pred = post_dict.get('data_pred')
    data_pred_valid = post_dict.get('data_pred_valid')
    pred = post_dict.get('pred')
    pred_prob = post_dict.get('pred_prob')
    start = post_dict['start']

    dataset_valid = dataset_valid.copy(deepcopy=True)
    dataset_test = dataset_test.copy(deepcopy=True)
    data_pred = data_pred.copy(deepcopy=True)
    data_pred_valid = data_pred_valid.copy(deepcopy=True)

    if bma == 1:
        cost_constraint = "fnr"
        CPP = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                             unprivileged_groups=unprivileged_groups,
                                             cost_constraint=cost_constraint,
                                             seed=None)
        CPP = CPP.fit(dataset_valid, data_pred_valid)
        data_pred = CPP.predict(data_pred)
        data_pred_valid = CPP.predict(data_pred_valid)
        pred = data_pred.labels
        pred_prob = data_pred.scores
        nam = 'cpp'
        CPP = None
    elif bma == 2:
        EOP = EqOddsPostprocessing(privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups,
                                   seed=None)
        EOP = EOP.fit(dataset_valid, data_pred_valid)
        data_pred = EOP.predict(data_pred)
        data_pred_valid = EOP.predict(data_pred_valid)
        pred = data_pred.labels
        pred_prob = data_pred.scores
        nam = 'eop'
        EOP = None
    elif bma == 3:
        ROC = RejectOptionClassification(privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups)
        ROC = ROC.fit(dataset_valid, data_pred_valid)
        data_pred = ROC.predict(data_pred)
        data_pred_valid = ROC.predict(data_pred_valid)
        pred = data_pred.labels
        pred_prob = data_pred.scores
        nam = 'roc'
        ROC = None
    elif bma == 0:
        return post_dict

    finish = 0
    finish = time.perf_counter()
    total_finish = get_total_finish(post_dict)
    total_finish = total_finish + finish

    pre = post_dict.get("pre")

    in_p = post_dict.get('in')
    post_a = post_dict.get("post")
    name = post_dict.get('class')

    post_a, count = namer(post_a, post_dict, nam)

    classified_metric = class_met(dataset_test, data_pred, unprivileged_groups, privileged_groups)

    post_dict = None

    post_d = {'dataset': data_pred,
              'dataset_used': data_used,
              'privileged_groups': privileged_groups,
              'unprivileged_groups': unprivileged_groups,
              'pre': pre,
              'in': in_p,
              'post': post_a,
              'count': count,
              'sens': sens,
              'dataset_test': dataset_test,
              'data_pred': data_pred,
              'dataset_valid': dataset_valid,
              'pred': pred,
              'pred_prob': pred_prob,
              'class': name,
              'class_met': classified_metric,
              'start': start,
              'finish': total_finish}

    data_used = None
    privileged_groups = None
    unprivileged_groups = None
    pre = None
    in_p = None
    post_a = None
    count = None
    sens = None
    dataset_test = None
    dataset_valid = None
    name = None
    start = None
    total_finish = None
    data_pred = None
    pred = None
    classified_metric = None
    bma = None
    finish = None
    nam = None

    return post_d


def class_met(cm_dataset, classified_dataset, unprivileged_groups,
              privileged_groups):  # returns 'classifed metric' which is used for AIF360 fairness metrics

    classified_metric = ClassificationMetric(cm_dataset,
                                             classified_dataset,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    cm_dataset = None
    classified_dataset = None
    unprivileged_groups = None
    privileged_groups = None

    return classified_metric


def sorter(sort_dict):  # prepare for dataframe, delete datasets from memory

    if ('failed' in sort_dict):

        sort_d = {'pre': sort_dict['pre'],
                  'in': sort_dict['in'],
                  'post': sort_dict['post'],
                  'mean_diff': None,
                  'dis_impact': None,
                  'theil': None,
                  'av_odds': None,
                  'eq_opp_diff': None,
                  'mean_diff_orig': None,
                  'dis_impact_orig': None,
                  'acc': None,  # hold out, labels of testing data vs labels of transformed
                  'prec': None,
                  'rec': None,
                  'auc': None,
                  'class_name': sort_dict.get('class'),
                  'data_used_name': sort_dict['dataset_used'].title(),
                  'sens_name': sort_dict['sens'].title(),
                  'acc_check': True,
                  't': {0},
                  'failed': sort_dict['failed']}

    else:

        pre = sort_dict['pre']
        in_p = sort_dict['in']
        post = sort_dict['post']
        data_pred = sort_dict['data_pred']
        unprivileged_groups = sort_dict['unprivileged_groups']
        privileged_groups = sort_dict['privileged_groups']
        classified_metric = sort_dict['class_met']
        dataset_test = sort_dict['dataset_test']
        pred = sort_dict['pred']
        pred_prob = sort_dict['pred_prob']
        data_used = sort_dict['dataset_used']
        class_name = sort_dict.get('class')
        sens = sort_dict['sens']
        finish = sort_dict['finish']
        start = sort_dict['start']
        t = {round(finish - start, 2)}

        # df = bank_test.convert_to_dataframe()
        # print('in nsorter' + str(df[0].shape[0]) + ' rows, ' + str(df[0].shape[1]) + ' cols')
        # print(df[0].columns)

        # print (pred)
        # print (pred_prob)

        mean_diff, dis_impact = metric(data_pred, unprivileged_groups, privileged_groups)  # calculate fairness scores
        theil, av_odds, eq_opp_diff = classified_metric_func(classified_metric)  # calculate fairness scores
        mean_diff_orig, dis_impact_orig = metric(dataset_test, unprivileged_groups,
                                                 privileged_groups)  # get the original scores
        acc, prec, rec = apr_score(dataset_test, pred)  # calculate performance metrics
        data_used_name = data_used.title()
        if np.unique(data_pred.labels)[1] != data_pred.favorable_label:
            auc = roc_auc_score(dataset_test.labels, 1 - pred_prob)
        else:
            auc = roc_auc_score(dataset_test.labels, pred_prob)
        maj_class = count_maj(dataset_test, data_used)  # get majority class percentage
        acc_check = acc_checker(maj_class, acc)  # compare majority class to accuracy
        data_used = data_used.strip()
        sens_name = sens.title()

        sort_dict = None

        sort_d = {'pre': pre,
                  'in': in_p,
                  'post': post,
                  'mean_diff': mean_diff,
                  'dis_impact': dis_impact,
                  'theil': theil,
                  'av_odds': av_odds,
                  'eq_opp_diff': eq_opp_diff,
                  'mean_diff_orig': mean_diff_orig,
                  'dis_impact_orig': dis_impact_orig,
                  'acc': acc,  # hold out, labels of testing data vs labels of transformed
                  'prec': prec,
                  'rec': rec,
                  'auc': auc,
                  'class_name': class_name,
                  'data_used_name': data_used_name,
                  'sens_name': sens_name,
                  'acc_check': acc_check,
                  't': t,
                  'failed': None}
        pre = None
        in_p = None
        post = None
        data_pred = None
        privileged_groups = None
        unprivileged_groups = None
        classified_metric = None
        dataset_test = None
        pred = None
        data_used = None
        class_name = None
        sens = None
        start = None
        finish = None
        t = None
        data_used_name = None
        sens = None
        mean_diff = None
        dis_impact = None
        theil = None
        av_odd = None
        eq_opp_diff = None
        mean_diff_orig = None
        dis_impact_orig = None
        acc = None
        prec = None
        rec = None
        class_name = None
        data_used_name = None
        sens_name = None
        auc = None

        maj_class = None
        acc_check = None

    return sort_d


def append_func(result):  # append results to dataframe

    cols = ['Pre', 'In_p', 'Post', 'Mean Difference', 'Disparate Impact', 'Theil Index', 'Average Odds Difference',
            'Equal Opportunity Difference', 'Orig Mean Difference', 'Orig Disparate Impact', 'Accuracy', 'Precision',
            'Recall', 'AUC',
            'Classifier', 'Dataset', 'Sens_Attr', 'Valid', 'Time', 'Failed']

    dfTest = pd.DataFrame(columns=cols)

    for x in result:
        temp_dict = defaultdict()

        temp_dict['Pre'] = [x['pre']]
        temp_dict['In_p'] = [x['in']]
        temp_dict['Post'] = [x['post']]

        temp_dict['Mean Difference'] = [x['mean_diff']]
        temp_dict['Disparate Impact'] = [x['dis_impact']]
        temp_dict['Theil Index'] = [x['theil']]
        temp_dict['Average Odds Difference'] = [x['av_odds']]
        temp_dict['Equal Opportunity Difference'] = [x['eq_opp_diff']]
        temp_dict['Orig Mean Difference'] = [x['mean_diff_orig']]
        temp_dict['Orig Disparate Impact'] = [x['dis_impact_orig']]

        temp_dict['Accuracy'] = [x['acc']]
        temp_dict['Precision'] = [x['prec']]
        temp_dict['Recall'] = [x['rec']]
        temp_dict['AUC'] = [x['auc']]

        temp_dict['Classifier'] = [x['class_name']]
        temp_dict['Dataset'] = [x['data_used_name']]
        temp_dict['Sens_Attr'] = [x['sens_name']]

        temp_dict['Valid'] = [x['acc_check']]

        temp_dict['Time'] = [x['t']]
        temp_dict['Failed'] = [x['failed']]

        dfTest = pd.concat([dfTest, pd.DataFrame.from_dict(temp_dict)], ignore_index=True)

    return dfTest


def metric(dataset, unprivileged_groups, privileged_groups):  # calculate metric for AIF360 fairness calculations
    metric = BinaryLabelDatasetMetric(dataset,
                                      unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)
    mean_diff = metric.mean_difference()
    dis_impact = metric.disparate_impact()

    return mean_diff, dis_impact


def rank_df0(dataframe):  # rank columns

    dataframe = abso_val(dataframe, 'Disparate Impact')
    dataframe = abso_val(dataframe, 'Mean Difference')
    dataframe['Mean Difference'] = dataframe['Mean Difference'].astype('float')
    dataframe = abso_val(dataframe, 'Theil Index')
    dataframe = abso_val(dataframe, 'Average Odds Difference')
    dataframe = abso_val(dataframe, 'Equal Opportunity Difference')

    dataframe = rename_func(dataframe)

    dataframe['Acc_Rank'] = dataframe.groupby('Dataset')['Accuracy'].rank(method='min', na_option='keep',
                                                                          ascending=False)
    dataframe['Prec_Rank'] = dataframe.groupby('Dataset')['Precision'].rank(method='min', ascending=False)
    dataframe['Rec_Rank'] = dataframe.groupby('Dataset')['Recall'].rank(method='min', ascending=False)
    dataframe['Auc_Rank'] = dataframe.groupby('Dataset')['AUC'].rank(method='min', ascending=False)

    dataframe['Rank'] = (dataframe['Acc_Rank'] + dataframe['Prec_Rank'] + dataframe['Rec_Rank'] + dataframe[
        'Mean_Rank'] + dataframe['Dis_Rank']
                         + dataframe['Theil_Rank'] + dataframe['Aod_Rank'] + dataframe['Eod_Rank'] + dataframe[
                             'Auc_Rank']) / 9

    dataframe = dataframe.drop(
        ['Acc_Rank', 'Prec_Rank', 'Rec_Rank', 'Auc_Rank', 'Mean_Rank', 'Dis_Rank', 'Theil_Rank', 'Aod_Rank',
         'Eod_Rank'], axis=1)

    return dataframe


def abso_val(dataframe, column):  # gets the absolute value of a column, ranks it and drops the column

    dataframe['abso_column'] = dataframe[column].abs()

    if column == 'Disparate Impact':
        dataframe[column + '_Rank'] = dataframe.groupby('Dataset')['abso_column'].rank(method='min', na_option='keep',
                                                                                       ascending=False)
    else:
        dataframe[column + '_Rank'] = dataframe.groupby('Dataset')['abso_column'].rank(method='min', na_option='keep',
                                                                                       ascending=True)

    dataframe = dataframe.drop(['abso_column'], axis=1)

    return dataframe


def rename_func(dataframe):
    dataframe = dataframe.rename(columns={"Disparate Impact_Rank": "Dis_Rank",
                                          "Mean Difference_Rank": "Mean_Rank",
                                          "Theil Index_Rank": "Theil_Rank",
                                          "Average Odds Difference_Rank": "Aod_Rank",
                                          "Equal Opportunity Difference_Rank": "Eod_Rank"})
    return dataframe


def rank_df1(dataframe):  # ranks performance and fairness seperatedly

    dataframe['Acc_Rank'] = dataframe.groupby('Dataset')['Accuracy'].rank(method='min', na_option='keep',
                                                                          ascending=False)
    dataframe['Prec_Rank'] = dataframe.groupby('Dataset')['Precision'].rank(method='min', ascending=False)
    dataframe['Rec_Rank'] = dataframe.groupby('Dataset')['Recall'].rank(method='min', ascending=False)
    dataframe['Auc_Rank'] = dataframe.groupby('Dataset')['AUC'].rank(method='min', ascending=False)
    dataframe['Perf_Rank'] = (dataframe['Acc_Rank'] + dataframe['Prec_Rank'] + dataframe['Rec_Rank'] + dataframe[
        'Auc_Rank']) / 4

    dataframe['Mean_Rank'] = dataframe.groupby('Dataset')['Mean Difference'].rank(method='min', ascending=True)
    dataframe['Dis_Rank'] = dataframe.groupby('Dataset')['Disparate Impact'].rank(method='min', ascending=False)
    dataframe['Fair_Rank'] = (dataframe['Mean_Rank'] + dataframe['Dis_Rank']) / 2
    dataframe['Rank'] = (dataframe['Perf_Rank'] + dataframe['Fair_Rank']) / 2

    dataframe = dataframe.drop(['Acc_Rank', 'Prec_Rank', 'Rec_Rank', 'Mean_Rank', 'Dis_Rank', 'Auc_Rank'], axis=1)

    return dataframe


def ovr_rank_df(dataframe):  # retruns temp dataframe, of ranks averaged across datasets

    dfTemp = dataframe.groupby(['Pre', 'In_p', 'Post', 'Classifier'])[['Rank']].mean()
    dfTemp = dfTemp.rename(columns={"Rank": "Ovr_Rank"})
    dfTemp['Ovr_Rank'] = dfTemp['Ovr_Rank'].rank(method='min', ascending=True)

    return dfTemp


def count_maj(count_dataset, string):  # returns majority class

    count_dataset.labels.flatten()

    if string == 'german':
        count0 = np.count_nonzero(count_dataset.labels == 1)
        count_1 = np.count_nonzero(count_dataset.labels == 2)
    else:
        count0 = np.count_nonzero(count_dataset.labels == 0)
        count_1 = np.count_nonzero(count_dataset.labels == 1)

    if count0 > count_1:
        maj_class = count0 / (count0 + count_1)
    else:
        maj_class = count_1 / (count0 + count_1)

    count_dataset = None

    return maj_class


def apr_score(data, pred):  # returns performance metrics

    if pred is not None:
        acc = accuracy_score(data.labels, pred)
        prec = precision_score(data.labels, pred)
        recc = recall_score(data.labels, pred)
    else:
        acc = None
        prec = None
        recc = None

    data = None
    pred = None

    return acc, prec, recc


def acc_checker(maj, acc):  # compares majority class to accuracy

    if acc is not None:
        if acc > maj:
            check = 'Valid'
        else:
            check = 'Fail'
    else:
        check = None

    maj = None
    acc = None

    return check


def fair_checker(df):  # returns 'score'

    List = ['Mean Difference', 'Disparate Impact', 'Theil Index', 'Average Odds Difference',
            'Equal Opportunity Difference']
    origList = ['Orig Mean Difference', 'Orig Disparate Impact', 'Orig Theil', 'Orig Av Odds', 'Orig Eq Opp Diff']

    comparison_column = np.where(df['Mean Difference'].abs() < df['Orig Mean Difference'].abs(), 1, 0)
    comparison_column1 = np.where(df['Disparate Impact'].abs() < df['Orig Disparate Impact'].abs(), 1, 0)
    comparison_column2 = np.where(df['Theil Index'].abs() < df['Orig Theil'].abs(), 1, 0)
    comparison_column3 = np.where(df['Average Odds Difference'].abs() < df['Orig Av Odds'].abs(), 1, 0)
    comparison_column4 = np.where(df['Equal Opportunity Difference'].abs() < df['Orig Eq Opp Diff'].abs(), 1, 0)

    df['Score'] = comparison_column + comparison_column1 + comparison_column2 + comparison_column3 + comparison_column4

    return df


def classified_metric_func(classified_metric):  # returns fairness metrics

    if classified_metric is not None:
        theil = classified_metric.theil_index()
        av_odds = classified_metric.average_odds_difference()
        eq_opp_diff = classified_metric.equal_opportunity_difference()
    else:
        theil = np.NaN
        av_odds = np.NaN
        eq_opp_diff = np.NaN
    classified_metric = None

    return theil, av_odds, eq_opp_diff


def df_format(df):
    df = df.fillna({'Pre': '-', 'In_p': '-', 'Post': '-', 'Classifier': '-', 'Valid': '-'}).fillna(np.nan)
    df['Pre'] = df['Pre'].apply(lambda x: "".join([str(e) for e in x]))
    df['In_p'] = df['In_p'].apply(lambda x: "".join([str(e) for e in x]))
    df['Post'] = df['Post'].apply(lambda x: "".join([str(e) for e in x]))

    return df


def output(df, df2, run_name=None, stratified=True):
    if len(sys.argv) > 1:
        name1 = sys.argv[1] + "-output.csv"
        name2 = sys.argv[1] + "-output1.csv"
    else:
        if stratified:
            name1 = run_name + "StratifiedSampling-Output.csv"
            name2 = run_name + "StratifiedSampling-Output1.csv"
        else:
            name1 = run_name + "RandomSampling-Output.csv"
            name2 = run_name + "RandomSampling-Output1.csv"
    df.to_csv(name1, index=False)
    df2.to_csv(name2, index=False)
    return True


def rebuild_from_log(filename, output=None):
    f = open(filename, 'r')

    print('reading: ' + filename)

    results = []

    for line in f:
        if line.startswith('\"{'):
            line = line[1:-2].replace("\'", "\"")
            line = line.replace("None", "null")
            line = line.replace("{", "\"{")
            line = line.replace("}", "}\"")
            line = line[1:-1]
            # print(line)
            result = json.loads(line)
            results.append(result)

    f.close()

    dfTemp0 = append_func(results)  # append results to dataframe
    df, _ = df_sort(dfTemp0)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    runParallel = False

    stratified = False

    random_state = 5
    num_runs = 10
    for run in range(0, 2):
        run_name = "Run" + str(run)
        random_state = run
        main(runParallel, random_state, run_name, stratified)
# name = "Run2Strat"
# for i in range(10):
#    rebuild_from_log('/Users/scaton/Documents/Papers/FairMLComp/logs/'+name+'-'+str(i+1)+'.log', '/Users/scaton/Documents/Papers/FairMLComp/logs/'+name+'-'+str(i+1)+'.csv')
