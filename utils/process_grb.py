import numpy as np
import pandas as pd
import torch
import os 


FOLDS_COUNT = 11
TASKS = ['G', 'R', 'B']


def track_removed_patients(patients_to_remove, fold, file):

    with open(file, 'a') as f_patients:
        f_patients.write(f'fold {fold}')
        for id_patient in patients_to_remove:
            f_patients.write(f',{id_patient}')
        f_patients.write('\n')


def is_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def convert_data_for_grb():

    zenodo = pd.read_pickle(os.path.join('local_results', 'grbas_zenodo.pkl'))
    zenodo = zenodo['data']

    # Substitute 3 ratings by 2 ratings.
    for task in TASKS:
        zenodo[task] = zenodo[task].replace(3, 2)

    patients_zenodo = zenodo['id_patient'].unique()

    for set in ['train', 'val']:
        for f in range(FOLDS_COUNT):

            # Load list of input data.
            data_loader_name = f'local_results/folds/{set}_loader_supervised_True_frame_size_0.4spec_winsize_0.03hopsize_0.5fold{f}.pt'
            data_loader = torch.load(data_loader_name)
            # Load dataframe to append GRB features.
            data_labels_name = f'local_results/folds/{set}_data_supervised_True_frame_size_0.4spec_winsize_0.03hopsize_0.5fold{f}.pt'
            data_labels = torch.load(data_labels_name)
            # data_labels['id_patient'] = data_labels['id_patient'].astype(int)
            data_labels = data_labels[data_labels['id_patient'].apply(is_integer)]
            data_labels['id_patient'] = data_labels['id_patient'].astype(int)

            # Remove albayzin data.
            data_labels = data_labels[data_labels['dataset'] == 'neurovoz']

            # Remove patients that are not rated with the GRB scale.
            patients_neurovoz = data_labels['id_patient'].unique()
            patients_to_remove = np.setdiff1d(patients_neurovoz, patients_zenodo)
            track_removed_patients(patients_to_remove, f, 'local_results/patients_removed.txt')
            if len(patients_to_remove) == 0:
                continue
            data_labels.reset_index(inplace=True)
            indices_to_remove = data_labels[data_labels['id_patient'].isin(patients_to_remove)].index
            data_labels.drop(indices_to_remove, inplace=True)
            data_loader = torch.utils.data.DataLoader(
                [row for i, row in enumerate(data_loader.dataset) if i not in indices_to_remove],
                drop_last=False, batch_size=128, shuffle=True)
            print(f'[DEBUG] patients with missing GRB labels: {len(patients_to_remove)}')
            
            # Add GRB labels.
            data_labels = data_labels.merge(zenodo, how='inner', on=['id_patient', 'text'])
            # columns_to_append = ['B', 'C']
            # if len(data_list) == len(df):
            #     for i in range(len(data_list)):
            #         for col in columns_to_append:
            #             data_list[i].append(df[col].iloc[i])
            print(f'[DEBUG] data_loader: {len(data_loader.dataset)} rows \tdata_labels: {len(data_labels)} rows')

            # Make 3s be 2s.
            for task in TASKS:
                data_labels[task] = data_labels[task].replace(2, 3)

            torch.save(data_loader, data_loader_name)
            torch.save(data_labels, data_labels_name)