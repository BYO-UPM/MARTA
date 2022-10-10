import numpy as np
from sklearn.model_selection import StratifiedKFold

def StratifiedGroupKFold_local(class_labels,groups,n_splits=2,shuffle=True):
    
    unique_groups = np.unique(groups)
    train_idx = []
    test_idx = []
    class_labels_g = []

    for i in range(len(unique_groups)):
        
        indx = np.argwhere(groups==unique_groups[i])
        if len(indx)>1:
            indx = indx[0]
        class_labels_g.append(class_labels[indx])
    class_labels_g = np.stack(class_labels_g).ravel()
    train_idx_p, _ = next(StratifiedKFold(n_splits=n_splits,shuffle=shuffle).split(np.zeros(len(class_labels_g)),class_labels_g))

    for i in range(len(class_labels_g)):
        indx = np.argwhere(groups==unique_groups[i])
        if i in train_idx_p:
            train_idx.append(indx)
        else:
            test_idx.append(indx)


    train_idx = np.concatenate(train_idx).ravel().tolist()
    test_idx = np.concatenate(test_idx).ravel().tolist()

    return train_idx, test_idx
        


