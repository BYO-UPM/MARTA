import argparse
import torch
import numpy as np
import wandb
from data_loader import load_patients_sp, Create_data_loaders
from pt_training import Training_model, Validate_model


def main():
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (args.cuda):
        torch.cuda.manual_seed(SEED)
    
    project_name = args.model_name + '_CL_CWT_Spec_from_matrix'

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
    #--------------------------------------------------------------------------------
        _ = Fold_Process(args,wandb,project_name,k)


def Fold_Process(args, wandb, project_name, k=1):
    X_train, X_test, y_train, y_test, dictOfClass = load_patients_sp(group_file='/home/julian/Documents/MATLAB/Oculografia/SP_Groups.csv')
    
    training_generator, val_generator, test_generator = Create_data_loaders( X_train.values, X_test.values, y_train.values, y_test.values,dictOfClass,args.dataset,batch_size=args.batch_size, repeat = 10)
    training_generator_ft, val_generator_ft, _ = Create_data_loaders( X_train.values, X_test.values, y_train.values, y_test.values,dictOfClass,args.dataset,batch_size=2*args.batch_size, repeat = 10)

    model = Training_model(project_name, training_generator,val_generator,training_generator_ft,val_generator_ft,wandb,args,freeze=False,k=k)

    print(f'Fold {k}')
    Validate_model(model, test_generator)

    return model

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/julian/DataDisk3/Datasets/SP_ICAs_CWT_Spec/',
                        help='path to dataset ')                #path to the datset (folder with different sp)
    parser.add_argument('--sp_inc', type=list, default=[1])     #list of sp experiments to include"
    parser.add_argument('--axis', type=str, default='x')        #sp axis
    parser.add_argument('--scheme', type=int, default=1)        #training scheme 1 to 4 (see pt_training.Training_model)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--class_weights', type=list, default=[1,1])
    parser.add_argument('--nTraining_epochs_cl', type=int, default=120)
    parser.add_argument('--nCooldown_epochs_cl', type=int, default=30)
    parser.add_argument('--nTraining_epochs_ft', type=int, default=40)
    parser.add_argument('--nCooldown_epochs_ft', type=int, default=10)
    parser.add_argument('--kfolds', type=int, default=5)
    parser.add_argument('--inChannels', type=int, default=3)
    parser.add_argument('--lr', default=0.00001, type=float,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--model_name', type=str, default='Vgg_11',
                        help='type of embedding')
    parser.add_argument('--save', type=str, default='model_checkpoints',
                        help='path to checkpoint ')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()