import argparse
import torch
import numpy as np
from .experiments.Parkinson_sp import cross_validation_experiment

def main():
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (args.cuda):
        torch.cuda.manual_seed(SEED)
        print('GPU available: {}'.format(torch.cuda.is_available()))
    
    sp_list = [int(item) for item in args.list]
    args.sp_inc = sp_list
    cross_validation_experiment(args)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/julian/DataDisk3/Datasets/SP_ICAs_CWT_Spec/',
                        help='path to dataset ')                #path to the datset (folder with different sp)
    parser.add_argument('--project_name', type=str, default='',
                        help='name for saving models and results')                #path to the datset (folder with different sp)
    parser.add_argument('-l', '--list', nargs='+', help='list of sps to include', type=str, default='1')
    parser.add_argument('--axis', type=str, default='x')        #sp axis
    parser.add_argument('--scheme', type=int, default=1)        #training scheme 1 to 4 (see pt_training.Training_model)
    parser.add_argument('--repeat_data', type=int, default=1)   #pass training data multiple times per epoch
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--class_weights', type=list, default=[1,1])# it is replaced by one estimated to balance the class weights
    parser.add_argument('--nTraining_epochs_cl', type=int, default=10)
    parser.add_argument('--nCooldown_epochs_cl', type=int, default=5)
    parser.add_argument('--nTraining_epochs_ft', type=int, default=40)
    parser.add_argument('--nCooldown_epochs_ft', type=int, default=10)
    parser.add_argument('--kfolds', type=int, default=10)
    parser.add_argument('--inChannels', type=int, default=3)
    parser.add_argument('--lr', default=0.00001, type=float,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--model_name', type=str, default='Vgg_11',
                        help='type of embedding')
    parser.add_argument('--save', type=str, default='model_checkpoints',
                        help='path to checkpoint ')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()