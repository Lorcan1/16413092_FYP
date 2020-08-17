import sys
sys.path.insert(1, "../")  

import pandas as pd
import numpy as np
np.random.seed(0)

import numpy
import random

import pprint
import json

from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.metrics import BinaryLabelDatasetMetric , ClassificationMetric
from aif360.algorithms.inprocessing import AdversarialDebiasing

from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, Reweighing
from aif360.algorithms.inprocessing import MetaFairClassifier, PrejudiceRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification

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

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.datasets import BinaryLabelDataset
import time
import gc

import queue
import itertools

from multiprocessing import Process, Queue, Pool
import multiprocessing

#Globals for the datasets
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


def log(message, dictionary=True):
    name = sys.argv[1] + '.log'
    f=open(name, "a+")
    if (dictionary):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        f.write('[' + current_time + '] ' + message + '\n')
    else:
        print(message, file=f)
    f.close()

def top(my_list, out_queue): #parrallelized function 
    counter = 0
    a = []

    for i in my_list:
        log('Here is the row: ' + str(i))
        log('Here is the counter ' + str(counter))
        
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
    
        top_data_d = data_f(i[0])       #main pipeline of bma functions called here
        top_pre_d =  pre(i[1], top_data_d)
        log(str(i) + ' pre complete')
        if (sanity_check(top_pre_d)):
            top_in_d = in_p(i[2], top_pre_d)
            top_class_d = classifier(i[3],top_in_d)
            log(str(i) + ' in/class complete')
            
            if (sanity_check(top_class_d)):
                top_post_d = post(i[4], top_class_d) #bias mitigation functions
                log(str(i) + ' post complete')
                top_sort_d = sorter(top_post_d)
            else:
                log('Sanity Check Failed in in/class' + str(i))
                top_sort_d = resolve_failed(top_class_d, 'in/class')
        else:
            log('Sanity Check Failed in pre' + str(i))
            top_sort_d = resolve_failed(top_pre_d, 'pre')


        out_queue.put(top_sort_d)              #result returned through queue
        #log(pprint.pformat(top_sort_d))
        log(top_sort_d, False)

        top_data_d.clear()
        top_pre_d.clear()
        top_in_d.clear()
        top_class_d.clear()
        top_post_d.clear()
        top_sort_d.clear()

        top_data_d = None
        top_pre_d = None
        top_in_d = None
        top_class_d = None
        top_post_d = None
        top_sort_d = None
    
        counter = counter + 1
    
 
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:] 


def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]       
            
def multi_run_wrapper(args):
   return top(*args)
        
def main():
    #setting up the datasets
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

    ricci_dataset = BinaryLabelDataset(favorable_label='1',      
                                       unfavorable_label='0',
                                       df=df,
                                       label_names=['Promotion'],
                                       protected_attribute_names=['Race'],
                                       unprivileged_protected_attributes=['0'])
    
    testSize = .4
    if (len(sys.argv) < 3): 
        stratified= True
    else:
        stratified = sys.argv[2].lower() == 'true'
    
    log('Stratified sampling enabled: ' + str(stratified))
    
    global bank_train, bank_test, bank_valid
    global adult_train, adult_test, adult_valid
    global german_train, german_test, german_valid
    global compas_train, compas_test, compas_valid
    global ricci_train, ricci_test, ricci_valid
    
    bank_train, bank_vt = sample_data(bank_dataset, testSize, stratified)
    bank_test, bank_valid = sample_data(bank_vt, .5, stratified)
    
    adult_train, adult_vt = sample_data(adult_dataset, testSize, stratified)
    adult_test, adult_valid = sample_data(adult_vt, .5, stratified)
    
    german_train, german_vt = sample_data(german_dataset, testSize, stratified)
    german_test, german_valid = sample_data(german_vt, .5, stratified)
    
    compas_train, compas_vt = sample_data(compas_dataset, testSize, stratified)
    compas_test, compas_valid = sample_data(compas_vt, .5, stratified)
    
    ricci_train, ricci_vt = sample_data(ricci_dataset, testSize, stratified)
    ricci_test, ricci_valid = sample_data(ricci_vt, .5, stratified)
    
    #1 = dataset
    #2 = pre
    #3= in_p
    #4 = class
    #5 = post
     
    a_list = []
    m,j,k = 2,3,4
    l = [(a,b,c,d,e)  for a in range(8) for b in range(j) for c in range(j) for d in range(k)for e in range(k)] #create list of inputs

    for x in l:
        if x[2] is 0 and x[3] is not 0 or x[2] is not 0 and x[3] is 0:   #cant have in-processing and classifier
#            if x[4] is not 3:                                            #ROC was calculated seperatedly as it is a memory hog
            a_list.append(x)

    # upper bound to n cores -- added by Simon to be nice on SONIC
    if (len(sys.argv) < 4):
        num_proc = 12
    else:
        num_proc = int(sys.argv[3])
        
    #return number of cores present on machine
    cpu_num = int(max(1, min(num_proc, multiprocessing.cpu_count()) / 4))
    log('Using ' + str(cpu_num) + ' cores')
    
    #randomly shuffle  input list before splitting to achieve a more equal runtime during parallelization
    random.shuffle(a_list)    

    #split input array                  
    five = numpy.array_split(numpy.array(a_list),cpu_num)      

    m = multiprocessing.Manager()   
    processes = []
    out_queue0 = m.Queue()
    numb_list = []
    counter = 0

    for x in five:
        numb_list.append((five[counter]))                 
        counter = counter + 1

    input_list = list(zip(numb_list,itertools.repeat(out_queue0)))

    
    pool = Pool(cpu_num)
    pool.map(multi_run_wrapper,input_list)

    result = []

    while out_queue0.qsize() != 0:              #prevent deadlock
          result.append(out_queue0.get())

    pool.close()
    dfTemp0 = append_func(result)               #append results to dataframe
    dfFinal0, dfFinal1 = df_sort(dfTemp0)       #clean dataframe, second dataframe is identical bar being ranked differently
    output(dfFinal0,dfFinal1)

    return None

def df_sort(dataframe):      #all dataframe cleaning handled here
    dataframe = df_format(dataframe)  #fix bma names
    dataframe  = df_orig_value(dataframe) #return orig fairness values for theil, av and eop     
    dataframe = fair_checker(dataframe)   #return score column

    dataframe1 = dataframe.copy()
    dataframe = rank_df0(dataframe)
    dataframe1 = rank_df1(dataframe1) #rank_df` is merged excpet overall performance and fairness rank calculated sep. before joining

    temp = ovr_rank_df(dataframe)
    temp2 = ovr_rank_df(dataframe1)

    dfFinal = merge0(temp , dataframe)    
    dfFinal1 = merge1(temp2, dataframe1)                    
    return dfFinal, dfFinal1

def df_orig_value(dataframe):  #appends unmitgated score to equivlaent mitigated row
    orig_score_df = dataframe[(dataframe['Pre'] == '-') & (dataframe['In_p'] == '-') & (dataframe['Post'] == '-')]   #find unmitigated rows
    orig_score_df = orig_score_df[['Theil Index', 'Average Odds Difference' , 'Equal Opportunity Difference',  'Classifier', 'Dataset', 'Sens_Attr']]
    orig_score_df = orig_score_df.rename(columns={"Theil Index" : "Orig Theil",
                                          "Average Odds Difference" : "Orig Av Odds",
                                          "Equal Opportunity Difference" : "Orig Eq Opp Diff"})
    dataframe.loc[dataframe.In_p != '-', 'Classifier'] = "Logistic Regression"
    new_df = pd.merge(dataframe, orig_score_df , how = 'left',  on = ['Classifier', 'Dataset' , 'Sens_Attr'])
    new_df.loc[new_df.In_p != '-', 'Classifier'] = "-"

    return new_df

def merge0(dataframe1, dataframe2):  #mereges temp dataframe containing rank averaged across datasets, with original datafame
    new_df = pd.merge(dataframe1, dataframe2 , how = 'left',  on = ['Pre', 'In_p' , 'Post', 'Classifier'])
    tempdf = new_df.pop('Ovr_Rank')
    new_df['Ovr_Rank'] = tempdf
    new_df['Rank'] = new_df.groupby('Dataset')['Rank'].rank(method = 'min',ascending=True)
    new_df = new_df.sort_values(by=['Dataset', 'Rank'])
    
    return new_df

def merge1(dataframe1, dataframe2): #similar to above but with perf-rank and fair-rank
    
    new_df = pd.merge(dataframe1, dataframe2 , how = 'left',  on = ['Pre', 'In_p' , 'Post', 'Classifier'])
    new_df['Ovr_Rank'] = (new_df['Perf_Rank'] + new_df['Fair_Rank'])/2
    new_df['Ovr_Rank'] = new_df.groupby('Dataset')['Ovr_Rank'].rank(method = 'min',ascending=True)
    new_df = new_df.rename(columns={"Rank_y": "Rank"})
    new_df['Rank'] = new_df.groupby('Dataset')['Rank'].rank(method = 'min',ascending=True)
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
    sens[0] = sens[0].replace("{", ' ')    #returns privileged group as a string
    sens[0] = sens[0].strip()

    sens = sens[0]

    start = 0
    start = time.perf_counter()       #begin timer
    
    data_d = {'dataset' : dataset_orig_train,
         'dataset_test' : dataset_orig_test,
         'dataset_valid': dataset_orig_valid,
         'dataset_used' : nam,
         'privileged_groups' : privileged_groups, 
         'unprivileged_groups' : unprivileged_groups,                                           
         'sens' : sens , 
          'start' : start}

    dataset = None
    dataset_orig_train = None
    dataset_orig_test = None  
    nam= None
    privileged_groups = None 
    unprivileged_groups = None                                         
    sens = None 
    start = None
    data_used = None
    pro_used = None

    return data_d 

def stratified_sample(dataset, testSize):
    df_conv, _ = dataset.convert_to_dataframe()
    y = dataset.label_names[0]
    y = df_conv.pop( y )
    X_train, X_test, y_train, y_test = train_test_split(df_conv, y, test_size=testSize, stratify=y)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    trainAIF = BinaryLabelDataset( favorable_label=dataset.favorable_label,
                                       unfavorable_label=dataset.unfavorable_label,
                                       df=train,
                                       label_names=dataset.label_names,
                                       protected_attribute_names=dataset.protected_attribute_names,
                                       unprivileged_protected_attributes=dataset.unprivileged_protected_attributes)
    
    testAIF = BinaryLabelDataset( favorable_label=dataset.favorable_label,
                                       unfavorable_label=dataset.unfavorable_label,
                                       df=test,
                                       label_names=dataset.label_names,
                                       protected_attribute_names=dataset.protected_attribute_names,
                                       unprivileged_protected_attributes=dataset.unprivileged_protected_attributes)
    
    return trainAIF, testAIF

def sample_data(dataset, testSize, stratified=True):
    if (stratified):
        return stratified_sample(dataset, testSize)
    else:
        return dataset.split([1-testSize], shuffle=False)

def get_total_finish(fin_dict): #keeps track of total time for each combination
    
    total_finish = fin_dict.get('total_finish')
    if total_finish is None:
        total_finish = 0
    else:
        total_finish = total_finish
    fin_dict = None

    return total_finish

def namer(bma_a, name_dict, name): #combines names for rows with multiple bmas
    
    if bma_a is None: 
            bma_a = []
            bma_a.append(name)
    else:
        bma_a.append(name)
    
    if  name_dict.get('count') is None:
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
    #check if the previous part of the pipeline broke the training data
    data = p_dict['dataset'] 
    
    fail = False
    
    y = data.labels
    if (len(np.unique(y)) == 1):
        fail=True
        
    return fail

def resolve_failed(p_dict, stage):
    p_dict['failed']=stage
    return p_dict
    

def pre(bma, p_dict): #applies pre-processing BMA

    data = p_dict['dataset'] 
    data_used = p_dict['dataset_used']
    unprivileged_groups = p_dict['unprivileged_groups']
    privileged_groups = p_dict['privileged_groups']
    sens = p_dict['sens']
    start = p_dict['start']

    dataset_test = p_dict['dataset_test']
    dataset_valid = p_dict['dataset_valid']
     
    if bma == 1:
        DI = DisparateImpactRemover(repair_level=1.0, sensitive_attribute = sens)
        dataset_t = DI.fit_transform(data)
        nam = 'di'
        DI = None      
    elif bma == 2:
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_t = RW.fit_transform(data)
        nam = 'rw'
        RW = None    
    elif bma == 0:   #no-preprcossing
        return p_dict
        
    finish = 0    
    finish = time.perf_counter()
    total_finish = get_total_finish(p_dict)
    
    total_finish = total_finish + finish
    
    pre_a = p_dict.get('pre')
    post= p_dict.get("post")
    in_p = p_dict.get('in')
    name = p_dict.get('class')

    pre_a, count = namer(pre_a, p_dict, nam)
    
    p_dict = None
        
    pre_d = {'dataset' : dataset_t,
         'dataset_used' : data_used,
          'privileged_groups' : privileged_groups, 
          'unprivileged_groups' : unprivileged_groups,
          'pre' : pre_a,
          'in': in_p,
          'post' : post,
          'count' : count,
          'sens' : sens,
          'dataset_test' : dataset_test,
          'dataset_valid': dataset_valid,
          'class' : name,
          'start' : start,
          'finish' : total_finish}

    data = None
    dataset_t= None
    data_used= None
    privileged_groups= None
    unprivileged_groups= None
    pre_a= None
    in_p= None
    post= None
    count= None
    sens= None
    name= None
    start= None
    finish = None
    total_finish = None
    dataset_test = None
    dataset_valid = None
    bma = None
    nam = None

    return pre_d 


def in_p(bma, in_dict): #applies in-processing classifier
        
    data = in_dict['dataset']
    data_used = in_dict['dataset_used']
    unprivileged_groups = in_dict['unprivileged_groups']
    privileged_groups = in_dict['privileged_groups']
    sens = in_dict['sens']
    dataset_test = in_dict['dataset_test']
    dataset_valid = in_dict['dataset_valid']
    start = in_dict['start']      

    if bma == 1:
        MFC = MetaFairClassifier(tau=0, sensitive_attr= sens, type = 'sr')
        #MFC = MetaFairClassifier(tau=0.8, sensitive_attr= sens, type = 'sr')
        MFC = MFC.fit(data)
        
        data_pred_valid = MFC.predict(dataset_valid)
        data_pred = MFC.predict(dataset_test)
        
        pred = data_pred.labels
        pred_valid = data_pred_valid.labels
        nam = 'mfc_sr'
        MFC = None
    if bma == 2:
        MFC2 = MetaFairClassifier(tau=0, sensitive_attr= sens, type = 'fdr')
        #MFC2 = MetaFairClassifier(tau=0.8, sensitive_attr= sens, type = 'fdr')
        MFC2 = MFC2.fit(data)
        
        data_pred_valid = MFC2.predict(dataset_valid)
        data_pred = MFC2.predict(dataset_test)
            
        pred = data_pred.labels
        pred_valid = data_pred_valid.labels
        nam = 'mfc_fdr'
        MFC2 = None       
    elif bma == 3:
        #PR = PrejudiceRemover(sensitive_attr= sens, eta=25.0)
        PR = PrejudiceRemover(sensitive_attr= sens, eta=1.0)
        PR = PR.fit(data)
        
        data_pred = PR.predict(dataset_valid)
        data_pred = PR.predict(dataset_test)
            
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
    in_p_a  = in_dict.get('in_p_a')
    post = in_dict.get('post')
    name = in_dict.get('class')
              
    in_p_a , count = namer(in_p_a, in_dict, nam) 

    classified_metric = class_met(dataset_test, data_pred, unprivileged_groups, privileged_groups) 
    
    in_dict = None
  
    in_d = {'dataset' : data_pred,
         'dataset_used' : data_used,
         'privileged_groups' : privileged_groups, 
         'unprivileged_groups' : unprivileged_groups,
         'pre' : pre,
         'in' : in_p_a, 
         'post' : post,
         'count' : count,
         'sens' : sens,
         'dataset_test' : dataset_test,
         'dataset_valid': dataset_valid,
         'data_pred' : data_pred,
         'data_pred_valid' : data_pred_valid,
         'pred' : pred,
         'pred_valid' : pred_valid,
         'class' : name,
         'class_met' : classified_metric,
         'start' : start,
         'finish' : total_finish}

    data = None
    data_used= None
    privileged_groups= None
    unprivileged_groups= None
    pre= None
    in_p_a= None
    post= None
    count= None
    sens= None
    dataset_test= None
    dataset_valid = None
    name= None
    start= None
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

    if clss == 1:
        
        lr = LogisticRegression()
        lr = lr.fit(data.features, data.labels.ravel()) #fitted on train(transformed) datatset   
        
        pred_valid = lr.predict(dataset_valid.features)  
        pred = lr.predict(dataset_test.features)  
                
        data_pred = dataset_test.copy()
        data_pred.labels = pred   
        
        data_pred_valid = dataset_valid.copy()
        data_pred_valid.labels = pred_valid
                
        name = 'Logistic Regression'
        lr = None        
    elif clss == 2:
        rf = RandomForestClassifier(n_estimators=100, 
                               max_features = 'sqrt')
        rf = rf.fit(data.features, data.labels.ravel())   
        
        pred_valid = rf.predict(dataset_valid.features)  
        pred = rf.predict(dataset_test.features)  
                
        data_pred = dataset_test.copy()
        data_pred.labels = pred   
        
        data_pred_valid = dataset_valid.copy()
        data_pred_valid.labels = pred_valid
        name = 'Random Forest'
        rf = None        
    elif clss == 3: 
        nb = GaussianNB()
        nb = nb.fit(data.features,data.labels.ravel())   
        
        pred_valid = nb.predict(dataset_valid.features)  
        pred = nb.predict(dataset_test.features)  
                
        data_pred = dataset_test.copy()
        data_pred.labels = pred   
        
        data_pred_valid = dataset_valid.copy()
        data_pred_valid.labels = pred_valid
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
    
    class_dict =  None

    class_d = {'dataset' : data,
         'dataset_used' : data_used,
         'privileged_groups' : privileged_groups, 
         'unprivileged_groups' : unprivileged_groups,
         'pre' : pre,
         'in' : in_p,
         'post' : post,
         'count': count,
         'sens' : sens,
         'dataset_test' : dataset_test,
         'dataset_valid': dataset_valid,
         'data_pred' : data_pred,
         'data_pred_valid' : data_pred_valid,
         'pred' : pred,
         'class' : name,
         'class_met' : classified_metric,
         'start' : start,
         'finish' : total_finish}

    data = None
    data_used= None
    privileged_groups= None
    unprivileged_groups= None
    pre= None
    in_p= None
    post= None
    count= None
    sens= None
    dataset_test= None
    dataset_valid = None
    name= None
    start= None
    total_finish = None
    data_pred = None
    data_pred_valid = None
    pred = None
    classified_metric = None
    clss = None
    finish = None
          
    return class_d

def post(bma, post_dict): #applies post-processing algorithms

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
    start = post_dict['start']
        
    if bma == 1:
        cost_constraint = "fnr"
        CPP = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                         unprivileged_groups = unprivileged_groups,
                                         cost_constraint=cost_constraint,
                                         seed=None)
        CPP = CPP.fit(dataset_valid, data_pred_valid)   
        data_pred = CPP.predict(dataset_test)  
        nam = 'cpp'
        CPP = None       
    elif bma == 2:
        EOP = EqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     seed=None)
        EOP= EOP.fit(dataset_valid, data_pred_valid)  
        data_pred = EOP.predict(dataset_test)
        nam = 'eop'
        EOP = None
    elif bma == 3:
        ROC = RejectOptionClassification(privileged_groups = privileged_groups,
                                 unprivileged_groups = unprivileged_groups)
        ROC = ROC.fit(dataset_valid, data_pred_valid)  
        data_pred = ROC.predict(dataset_test)
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
        
    post_a, count= namer(post_a, post_dict, nam) 

    classified_metric = class_met(dataset_test, data_pred, unprivileged_groups, privileged_groups)

    post_dict = None
    
    post_d = {'dataset' : data_pred,
         'dataset_used' : data_used,
         'privileged_groups' : privileged_groups, 
         'unprivileged_groups' : unprivileged_groups,
         'pre' : pre,
         'in' : in_p,
         'post' : post_a,
         'count' : count,
         'sens' : sens,
         'dataset_test' : dataset_test,
         'data_pred' : data_pred,
         'dataset_valid' : dataset_valid,
         'pred' : pred,
         'class' : name,
         'class_met' : classified_metric,
         'start' : start,
         'finish' : total_finish}

    data_used= None
    privileged_groups= None
    unprivileged_groups= None
    pre= None
    in_p= None
    post_a= None
    count= None
    sens= None
    dataset_test= None
    dataset_valid = None
    name= None
    start= None
    total_finish = None
    data_pred = None
    pred = None
    classified_metric = None
    bma = None
    finish = None
    nam = None

    return post_d

def class_met(cm_dataset, classified_dataset, unprivileged_groups, privileged_groups): #returns 'classifed metric' which is used for AIF360 fairness metrics

    classified_metric = ClassificationMetric(cm_dataset, 
                                             classified_dataset,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    cm_dataset = None
    classified_dataset = None
    unprivileged_groups = None
    privileged_groups = None

    return classified_metric
            
def sorter(sort_dict): #prepare for dataframe, delete datasets from memory

    if ('failed' in sort_dict):
        
        sort_d = { 'pre' : sort_dict['pre'], 
                   'in': sort_dict['in'],
                   'post' : sort_dict['post'],     
                   'mean_diff' : None,
                   'dis_impact': None,
                   'theil'  : None,
                   'av_odds' :None,
                    'eq_opp_diff'  : None, 
                    'mean_diff_orig' : None,
                    'dis_impact_orig' : None,
                    'acc' : None, #hold out, labels of testing data vs labels of transformed 
                    'prec': None,
                    'rec' : None,
                    'auc' : None,
                    'class_name' : sort_dict.get('class'),
                    'data_used_name' : sort_dict['dataset_used'].title(),
                    'sens_name' : sort_dict['sens'].title(),
                    'acc_check' : True,
                    't' : None,
                    'failed' : sort_dict['failed']}
        
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
        data_used = sort_dict['dataset_used']
        class_name = sort_dict.get('class')
        sens = sort_dict['sens']
        finish = sort_dict['finish']
        start = sort_dict['start']
        t = {round(finish-start, 2)}
    
        mean_diff, dis_impact = metric(data_pred, unprivileged_groups, privileged_groups) #calculate fairness scores
        theil, av_odds, eq_opp_diff = classified_metric_func(classified_metric) #calculate fairness scores
        mean_diff_orig, dis_impact_orig = metric(dataset_test, unprivileged_groups, privileged_groups) #get the original scores
        acc, prec, rec = apr_score(dataset_test, pred) #calculate performance metrics
        data_used_name = data_used.title()
        auc = roc_auc_score(dataset_test.labels, pred) #get auc 
        maj_class = count_maj(dataset_test, data_used) #get majority class percentage
        acc_check = acc_checker(maj_class, acc) #compare majority class to accuracy
        data_used = data_used.strip()
        sens_name = sens.title()
    
        sort_dict = None
    
        sort_d = { 'pre' : pre, 
                   'in': in_p,
                   'post' : post,     
                   'mean_diff' : mean_diff,
                   'dis_impact': dis_impact,
                   'theil'  : theil,
                   'av_odds' :av_odds,
                    'eq_opp_diff'  : eq_opp_diff, 
                    'mean_diff_orig' : mean_diff_orig,
                    'dis_impact_orig' : dis_impact_orig,
                    'acc' : acc, #hold out, labels of testing data vs labels of transformed 
                    'prec': prec,
                    'rec' : rec,
                    'auc' : auc,
                    'class_name' : class_name,
                    'data_used_name' :data_used_name,
                    'sens_name' : sens_name,
                    'acc_check' : acc_check,
                    't' : t,
                    'failed' : None}
        pre= None
        in_p= None
        post= None
        data_pred = None
        privileged_groups= None
        unprivileged_groups= None
        classified_metric = None
        dataset_test= None
        pred = None
        data_used= None
        class_name = None
        sens= None
        start= None
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

def append_func(result): #append results to dataframe
    
    dfTest = pd.DataFrame(columns = [ 'Pre', 'In_p', 'Post', 'Mean Difference' , 'Disparate Impact', 'Theil Index',
                                     'Average Odds Difference',   'Equal Opportunity Difference', 'Accuracy',
                                      'Precision', 'Recall', 'Classifier', 'Dataset', 'Sens_Attr', 'Valid' , 'Time'])
    for x in result:
               
        pre = x['pre']
        in_p = x['in']
        post = x['post']

        mean_diff = x['mean_diff']
        dis_impact = x['dis_impact']
        theil = x['theil']
        av_odds = x['av_odds']
        eq_opp_diff = x['eq_opp_diff']
        mean_diff_orig = x['mean_diff_orig']
        dis_impact_orig = x['dis_impact_orig']

        acc = x['acc']
        prec = x['prec']
        rec = x['rec']
        auc = x['auc']
 
        class_name = x['class_name']
        data_used_name = x['data_used_name']
        sens_name = x['sens_name']

        acc_check = x['acc_check']

        t = x['t']

        dfTest = dfTest.append({ 'Pre' : pre, 
                                 'In_p': in_p,
                                 'Post' : post,     
                                 'Mean Difference' : mean_diff,
                                 'Disparate Impact': dis_impact,
                                 'Theil Index'  : theil,
                                 'Average Odds Difference' :av_odds,
                                 'Equal Opportunity Difference'  : eq_opp_diff, 
                                 'Orig Mean Difference' : mean_diff_orig,
                                 'Orig Disparate Impact' : dis_impact_orig,
                                 'Accuracy' : acc, #hold out, labels of testing data vs labels of transformed 
                                 'Precision': prec,
                                 'Recall' : rec,
                                 'AUC' : auc,
                                 'Classifier' : class_name,
                                 'Dataset' :data_used_name,
                                 'Sens_Attr' : sens_name,
                                 'Valid' : acc_check,
                                 'Time' : t}, ignore_index = True)

    return dfTest


def metric(dataset, unprivileged_groups, privileged_groups): #calculate metric for AIF360 fairness calculations
    metric = BinaryLabelDatasetMetric(dataset, 
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    mean_diff = metric.mean_difference()
    dis_impact = metric.disparate_impact()
    
    return mean_diff, dis_impact
    
def rank_df0(dataframe): #rank columns 
 
    dataframe = abso_val(dataframe, 'Disparate Impact')
    dataframe = abso_val(dataframe, 'Mean Difference')
    dataframe['Mean Difference'] = dataframe['Mean Difference'].astype('float')
    dataframe = abso_val(dataframe, 'Theil Index')
    dataframe = abso_val(dataframe, 'Average Odds Difference')
    dataframe = abso_val(dataframe, 'Equal Opportunity Difference')
    
    dataframe = rename_func(dataframe)
      
    dataframe['Acc_Rank'] = dataframe.groupby('Dataset')['Accuracy'].rank(method = 'min',na_option = 'keep', ascending=False)
    dataframe['Prec_Rank'] = dataframe.groupby('Dataset')['Precision'].rank(method = 'min',ascending=False)
    dataframe['Rec_Rank'] = dataframe.groupby('Dataset')['Recall'].rank(method = 'min',ascending=False)
    dataframe['Auc_Rank'] = dataframe.groupby('Dataset')['AUC'].rank(method = 'min',ascending=False)
    
    dataframe['Rank'] = (dataframe['Acc_Rank'] + dataframe['Prec_Rank'] + dataframe['Rec_Rank'] + dataframe['Mean_Rank'] + dataframe['Dis_Rank']
                         + dataframe['Theil_Rank'] +  dataframe['Aod_Rank'] + dataframe['Eod_Rank']+ dataframe['Auc_Rank'])/9
    
    dataframe = dataframe.drop(['Acc_Rank', 'Prec_Rank', 'Rec_Rank','Auc_Rank', 'Mean_Rank', 'Dis_Rank','Theil_Rank', 'Aod_Rank', 'Eod_Rank'], axis = 1)       
        
    return dataframe

def abso_val(dataframe, column):#gets the absolute value of a column, ranks it and drops the column
    
    dataframe['abso_column'] = dataframe[column].abs()
    
    if column == 'Disparate Impact':
        dataframe[column + '_Rank'] = dataframe.groupby('Dataset')['abso_column'].rank(method = 'min',na_option = 'keep', ascending=False)
    else:
        dataframe[column + '_Rank'] = dataframe.groupby('Dataset')['abso_column'].rank(method = 'min',na_option = 'keep', ascending=True)
        
    dataframe = dataframe.drop(['abso_column'], axis = 1)
    
    return dataframe

def rename_func(dataframe):
    
    dataframe = dataframe.rename(columns={"Disparate Impact_Rank": "Dis_Rank",
                                          "Mean Difference_Rank": "Mean_Rank",
                                          "Theil Index_Rank" : "Theil_Rank",
                                          "Average Odds Difference_Rank" : "Aod_Rank",
                                          "Equal Opportunity Difference_Rank" : "Eod_Rank"})
    return dataframe


def rank_df1(dataframe): #ranks performance and fairness seperatedly
        
        dataframe['Acc_Rank'] = dataframe.groupby('Dataset')['Accuracy'].rank(method = 'min',na_option = 'keep', ascending=False)
        dataframe['Prec_Rank'] = dataframe.groupby('Dataset')['Precision'].rank(method = 'min',ascending=False)
        dataframe['Rec_Rank'] = dataframe.groupby('Dataset')['Recall'].rank(method = 'min',ascending=False)
        dataframe['Auc_Rank'] = dataframe.groupby('Dataset')['AUC'].rank(method = 'min',ascending=False)
        dataframe['Perf_Rank'] = (dataframe['Acc_Rank'] + dataframe['Prec_Rank'] + dataframe['Rec_Rank'] + dataframe['Auc_Rank'])/4

        dataframe['Mean_Rank'] = dataframe.groupby('Dataset')['Mean Difference'].rank(method = 'min',ascending=True)
        dataframe['Dis_Rank'] = dataframe.groupby('Dataset')['Disparate Impact'].rank(method = 'min',ascending=False)
        dataframe['Fair_Rank'] =  (dataframe['Mean_Rank'] + dataframe['Dis_Rank'])/2                        
        dataframe['Rank'] = (dataframe['Perf_Rank'] + dataframe['Fair_Rank'])/2

        dataframe = dataframe.drop(['Acc_Rank', 'Prec_Rank', 'Rec_Rank', 'Mean_Rank', 'Dis_Rank','Auc_Rank'], axis = 1)

        return dataframe

def ovr_rank_df(dataframe): #retruns temp dataframe, of ranks averaged across datasets
    
    dfTemp = dataframe.groupby(['Pre', 'In_p', 'Post', 'Classifier'])[['Rank']].mean()
    dfTemp = dfTemp.rename(columns={"Rank": "Ovr_Rank"})
    dfTemp['Ovr_Rank'] = dfTemp['Ovr_Rank'].rank(method = 'min',ascending=True)

    return dfTemp

def count_maj(count_dataset, string): #returns majority class
    
        count_dataset.labels.flatten()
        
        if string is 'german':
            count0 = np.count_nonzero(count_dataset.labels == 1)
            count_1 = np.count_nonzero(count_dataset.labels == 2)
        else:
            count0 = np.count_nonzero(count_dataset.labels == 0)
            count_1 = np.count_nonzero(count_dataset.labels == 1)
        
        if count0 > count_1: 
            maj_class = count0/(count0 + count_1)
        else: 
            maj_class = count_1/(count0 + count_1)

        count_dataset = None
            
        return maj_class

def apr_score(data, pred):#returns performance metrics
   
    if pred is not None:
        acc = accuracy_score(data.labels, pred)
        prec = precision_score(data.labels, pred, average = 'weighted')
        recc = recall_score(data.labels, pred, average = 'weighted')
    else:
        acc = None
        prec = None
        recc = None

    data = None
    pred = None
        
    return acc, prec, recc

def acc_checker(maj, acc): #compares majority class to accuracy
    
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
    
def fair_checker(df): #returns 'score'

    List = ['Mean Difference', 'Disparate Impact','Theil Index','Average Odds Difference','Equal Opportunity Difference']
    origList = ['Orig Mean Difference','Orig Disparate Impact','Orig Theil','Orig Av Odds','Orig Eq Opp Diff']

    comparison_column = np.where(df['Mean Difference'].abs() < df['Orig Mean Difference'].abs(), 1, 0)
    comparison_column1 = np.where(df['Disparate Impact'].abs() < df['Orig Disparate Impact'].abs(), 1, 0)
    comparison_column2 = np.where(df['Theil Index'].abs() < df['Orig Theil'].abs(), 1, 0)
    comparison_column3 = np.where(df['Average Odds Difference'].abs() < df['Orig Av Odds'].abs(), 1, 0)
    comparison_column4 = np.where(df['Equal Opportunity Difference'].abs() < df['Orig Eq Opp Diff'].abs(), 1, 0)

    df['Score'] = comparison_column + comparison_column1 +  comparison_column2 + comparison_column3 + comparison_column4
   
    return df

def classified_metric_func(classified_metric): #returns fairness metrics
    
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
    
    df = df.fillna({'Pre':'-', 'In_p':'-', 'Post':'-', 'Classifier' : '-', 'Valid' : '-'}).fillna(np.nan)
    df['Pre'] = df['Pre'].apply(lambda x: "".join([str(e) for e in x]))
    df['In_p'] = df['In_p'].apply(lambda x: "".join([str(e) for e in x]))
    df['Post'] = df['Post'].apply(lambda x: "".join([str(e) for e in x]))
        
    return df

def output(df, df2):
    
    name1 = sys.argv[1] + "-output.csv"
    name2 = sys.argv[1] + "-output1.csv"
    
    df.to_csv(name1, index=False)
    df2.to_csv(name2, index=False)
    return True

if __name__ == "__main__":
   main()

