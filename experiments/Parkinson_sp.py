import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from ..data_loaders.pt_data_loader_img import Dataset_Spec_img, Dataset_Spec_img_CL, Dataset_Spec_txtmat, Dataset_Spec_txtmat_CL
from ..data_loaders.pt_data_loader_frp import Dataset_FPR, Dataset_FPR_CL
from ..training.pt_training import Training_model
from ..validation.evaluation import one_run_validate_model_multinstance, multi_run_validate_model_multinstance
from ..training.pt_training import StratifiedBatchSampler
from ..utils.utils import StratifiedGroupKFold_local
import os
import wandb
import pickle

def load_patients_sp(group_file):

    groups_df = pd.read_csv(group_file,sep='\t')

    groups_df.drop(groups_df[groups_df.Class == 'MSA'].index, inplace=True)
    groups_df.drop(groups_df[groups_df.Class == 'control joven'].index, inplace=True)
    groups_df.loc[groups_df.Class == 'Parkinson (T)',['Class']] = 'Parkinson'
    groups_df.loc[groups_df.Class == 'Parkinson (RA)',['Class']] = 'Parkinson'

    dictOfClass = dict(zip(np.unique(groups_df.Class), [1,0]))

    return groups_df, dictOfClass

def read_sp(dir_img, groups_df, sp_inc = [1],axis='x'):
    SP_folders = os.listdir(dir_img)
    X_sp_ids = []
    label = []
    Participant = []
    
    for sp in SP_folders:
        #----- fin sp number ---------
        n_sp = int(sp.split('_')[1])
        if n_sp in sp_inc:
            #----- set axis for each sp according to the sp number -------------
            if axis == 'x':
                axis = 'Axis_x'
            elif axis=='y':
                axis = 'Axis_y'

            sp_dir = os.path.join(dir_img,sp,axis)
            img_files = os.listdir(sp_dir+os.sep)
            for img in img_files:
                #-----remove .txt from file name-----
                img = img.split('.')[0]
                try:
                    label.append(groups_df[groups_df.Patient==img].Class.values[0])
                    X_sp_ids.append(sp + '_'+ axis +'_' + img)
                    Participant.append(img)
                except:
                    print('Participant excluded {}'.format(img))
    return np.stack(X_sp_ids), np.stack(label), np.stack(Participant)

def data_partition(X_sp_ids,label,Participant, n_splits=10):

    #train_idx, test_idx = next(StratifiedGroupKFold(n_splits=n_splits,shuffle=True).split(X_sp_ids,label,Participant))
    train_idx, test_idx = StratifiedGroupKFold_local(label,Participant,n_splits=n_splits,shuffle=True)
    X_train = X_sp_ids[train_idx]
    X_test = X_sp_ids[test_idx]
    y_train = label[train_idx]
    y_test = label[test_idx]
    p_train = Participant[train_idx]
    p_test = Participant[test_idx]

    return X_train, X_test, y_train, y_test, p_train, p_test

def Create_data_loaders( X_train, X_test, y_train, y_test,dictOfClass,dir_img,batch_size=4, scheme=1, repeat = 10):

    params_dloader_training = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0}

    params_dloader_test = {'batch_size': 2*batch_size,
          'shuffle': False,
          'num_workers': 0}

    X_train_r = X_train
    y_train_r = y_train
    X_val_r = X_test
    y_val_r = y_test
    for _ in range(2,repeat):
        X_train_r = np.r_[X_train_r,X_train]
        y_train_r = np.r_[y_train_r,y_train]
        X_val_r = np.r_[X_val_r,X_test]
        y_val_r = np.r_[y_val_r,y_test]

    res =  X_train_r.shape[0]%batch_size
    if res > 0:
        X_train_r = X_train_r[:-res]
        y_train_r = y_train_r[:-res]
    
    unique_classes = {k: v for k, v in sorted(dictOfClass.items(), key=lambda item: item[1])}

    class_weight = compute_class_weight("balanced",classes = list(unique_classes.keys()), y=y_train_r)
    print('Class weight = {}'.format(class_weight))

    if "FRP" in dir_img:
        print('Input data: FRP')
        dic_methods={
            'training_set_cl': Dataset_FPR_CL,
            'training_set': Dataset_FPR,
            'scale':0.8 #crop_factor
            }
        
    elif "txt" in dir_img:
        dic_methods={
            'training_set_cl': Dataset_Spec_txtmat_CL,
            'training_set': Dataset_Spec_txtmat,
            'scale':(0.2, 0.3) #occlusion scale
            } 
    else:
        print('Input data: *.png Spec')
        dic_methods={
            'training_set_cl': Dataset_Spec_img_CL,
            'training_set': Dataset_Spec_img,
            'scale':(0.2, 0.3) #occlusion scale
            } 
 
    if (scheme == 2) or (scheme== 4):

        training_set_cl = dic_methods['training_set_cl'](dir_img,X_train_r, y_train_r, dictOfClass, multi_instance=True, scale=dic_methods['scale'])
        training_set_ft = dic_methods['training_set'](dir_img,X_train_r, y_train_r, dictOfClass, multi_instance=True, scale=dic_methods['scale'])
        val_set = dic_methods['training_set'](dir_img,X_val_r, y_val_r, dictOfClass, train=False, multi_instance=True)
        test_set = dic_methods['training_set'](dir_img,X_test, y_test, dictOfClass,train=False, multi_instance=True)

        training_generator_cl = torch.utils.data.DataLoader(training_set_cl,  
                                                 batch_sampler=StratifiedBatchSampler(y_train_r, batch_size=params_dloader_training['batch_size']),
                                                 num_workers=0)
        training_generator_ft = torch.utils.data.DataLoader(training_set_ft,  
                                                 batch_sampler=StratifiedBatchSampler(y_train_r, batch_size=2*params_dloader_training['batch_size']),
                                                 num_workers=0)
        val_generator_cl = torch.utils.data.DataLoader(val_set, **params_dloader_training)
        val_generator_ft = torch.utils.data.DataLoader(val_set, **params_dloader_test)  
        test_generator = torch.utils.data.DataLoader(test_set, **params_dloader_test)

        dict_generators ={
            'training_generator_cl' : training_generator_cl,
            'training_generator_ft' : training_generator_ft,
            'val_generator_cl' : val_generator_cl,
            'val_generator_ft' : val_generator_ft,
            'test_generator': test_generator,
        }
    else:
        training_set_ft = dic_methods['training_set'](dir_img,X_train_r, y_train_r, dictOfClass, multi_instance=True, scale=dic_methods['scale'])
        val_set = dic_methods['training_set'](dir_img,X_val_r, y_val_r, dictOfClass, train=False, multi_instance=True)
        test_set = dic_methods['training_set'](dir_img,X_test, y_test, dictOfClass,train=False, multi_instance=True)

        training_generator_ft = torch.utils.data.DataLoader(training_set_ft,  
                                                 batch_sampler=StratifiedBatchSampler(y_train_r, batch_size=2*params_dloader_training['batch_size']),
                                                 num_workers=0)
        val_generator_ft = torch.utils.data.DataLoader(val_set, **params_dloader_test)  
        test_generator = torch.utils.data.DataLoader(test_set, **params_dloader_test)

        dict_generators ={
            'training_generator_ft' : training_generator_ft,
            'val_generator_ft' : val_generator_ft,
            'test_generator': test_generator,
        }


    return dict_generators, class_weight


def load_data(dir_img,sp_inc = [1],axis='x',batch=4,scheme=1,repeat=1):
    
    groups_df, dictOfClass = load_patients_sp(group_file='/home/julian/Documents/MATLAB/Oculografia/SP_Groups.csv')
    X_sp_ids, label, Participant = read_sp(dir_img, groups_df, sp_inc = sp_inc,axis=axis)
 
    X_train, X_test, y_train, y_test, p_train, p_test = data_partition(X_sp_ids,label,Participant, n_splits=13)

    dict_generators,  class_weight = Create_data_loaders( X_train, X_test, y_train, y_test,dictOfClass,dir_img,batch_size=batch, scheme=scheme, repeat = repeat)
    
    return dict_generators, class_weight, p_train, p_test


def cross_validation_experiment(args):

    project_name = args.model_name + args.project_name

    acc_t = []
    sensi_t = []
    especi_t = []
    preci_t = []
    f1_t = []
    Patients_t = []
    target_t = []
    prediction_t = []
    score_t = []
    std_prob_t = []
    for k in range(args.kfolds):
        #---------------- Training Monitor ---------------------------------------------
        wandb.init(project=project_name)
        wandb.config = {
            "learning_rate_cl": args.lr,
            "learning_rate_ft": args.lr,
            "epochs_cl": args.nTraining_epochs_cl+args.nCooldown_epochs_cl,
            "epochs_ft": args.nTraining_epochs_ft+args.nCooldown_epochs_ft,
            "batch_size": args.batch_size
            }
        #wandb = []
    #--------------------------------------------------------------------------------
        acc, sensi, especi, preci, f1, Patients, target, pre, score, std_prob = Fold_Process(args,wandb,project_name,k)
        
        acc_t.append(acc)
        sensi_t.append(sensi)
        especi_t.append(especi)
        preci_t.append(preci)
        f1_t.append(f1)
        Patients_t.append(Patients)
        target_t.append(target)
        prediction_t.append(pre)
        score_t.append(score)
        std_prob_t.append(std_prob)
    #---------------------- saveing results ----------------------------------------------
    dictResults = {
        'acc_t':acc_t,
        'sensi_t':sensi_t,
        'especi_t':especi_t,
        'preci_t':preci_t,
        'f1_t':f1_t,
        'Patients_t':Patients_t,
        'target_t':target_t,
        'prediction_t':prediction_t,
        'score_t':score_t,
        'std_prob_t':std_prob_t
    }

    filename = os.path.join(args.save, project_name + "_results.pkl")
    
    if os.path.exists(filename):
        os.remove(filename)
        
    with open(filename, 'ab') as f:
        pickle.dump(dictResults,f)
    #-------------------------------------------------------------------------------------
    print('End of experiment')
        

def Fold_Process(args, wandb, project_name, k=1):
    
    dict_generators, class_weight, _ , p_test = load_data(args.dataset,sp_inc = args.sp_inc,axis=args.axis,batch=args.batch_size,scheme=args.scheme,repeat=args.repeat_data)
    args.class_weight = class_weight
    model = Training_model(project_name, dict_generators,args,wandb,freeze=False,k=k)

    print(f'Fold {k}')
    one_run_validate_model_multinstance(model, dict_generators['test_generator'],p_test)

    acc, sensi, especi, preci, f1, Npatients, target, pre, score, std_prob  = multi_run_validate_model_multinstance(model, dict_generators['test_generator'], p_test, repeat=10)

    return acc, sensi, especi, preci, f1, Npatients, target, pre, score, std_prob