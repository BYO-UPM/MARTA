import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt

def one_run_validate_model(model, test_generator, multi_instance=False, p_test=[], fPrint = True, fReturn=False):

    model.to('cpu')
    model.eval()
    m = torch.nn.Softmax(dim=1)
    pred_label = []
    pred_prob = []
    target_c = []
    for batch in test_generator:
        sample, target = batch
        target_c.append(target.detach().numpy())
        output=m(model.forward(sample))
        pred_label.append(np.argmax(output.detach().numpy(),axis=1))
        pred_prob.append(output.detach().numpy()[:,1])
    pred = np.concatenate(pred)
    target_c = np.concatenate(target_c)

    if fPrint:
        if multi_instance:
            _,_,_,_,_,_,_,_,_ = eval_multinstance_biclass(target_c,pred_prob,p_test,plot_roc=False)
        else:
            print(metrics.classification_report(target_c, pred_label, target_names=['norm','pat']))
    
    if fReturn:
        return target_c,pred_prob

def one_run_validate_model_multinstance(model, test_generator, p_test):
    one_run_validate_model(model, test_generator, multi_instance=True, p_test=p_test)

def multi_run_validate_model_multinstance(model, test_generator, p_test, repeat=10):
    pred = []
    for _ in range(repeat):
        target_c, pred_prob = one_run_validate_model(model, test_generator, p_test=p_test, fPrint = False, fReturn=True)
        pred.append(pred_prob)
    pred = np.stack(pred)
    mean_prob = np.mean(pred,axis=0)
    std_prob = np.std(pred,axis=0)
    acc, sensi, especi, preci, f1, Npatients, target, pre, score = eval_multinstance_biclass(target_c,mean_prob,p_test,print_custom_report=False)
    return acc, sensi, especi, preci, f1, Npatients, target, pre, score, std_prob


def eval_multinstance_biclass(target,predict,patient,print_custom_report=True, plot_det=False,plot_roc=False,print_report=False, estimator_name='example estimator'):

    Npatients = np.unique(patient)
    target_p = []
    predict_p = []
    score_p = []
    for i in range(len(Npatients)):
        target_pat = target[patient==Npatients[i]]
        predict_pat =  predict[patient==Npatients[i]]
        tar, pre, score = joint_prob(target_pat,predict_pat)
        target_p.append(tar)
        predict_p.append(pre)
        score_p.append(score)

    target_p = np.stack(target_p)
    print(target_p)
    predict_p = np.stack(predict_p)
    score_p = np.stack(score_p)
    print(score_p)
    TP = np.sum(predict_p[target_p==1]==1)
    TN = np.sum(predict_p[target_p==0]==0)
    FP = np.sum(predict_p[target_p==0]==1)
    FN = np.sum(predict_p[target_p==1]==0)

    acc = (TP + TN)/(TP + TN + FP + FN)
    sensi = TP/(TP + FN)
    especi = TN/(TN + FP)
    preci = TP/(TP + FP)
    f1 = (2*preci*sensi)/(preci+sensi)

    if plot_roc:
        fpr, tpr, _ = metrics.roc_curve(target_p, score_p)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name=estimator_name)
        display.plot()
        plt.show()

    if plot_det:
        fpr, fnr, _ = metrics.det_curve(target_p, score_p)
        display = metrics.DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=estimator_name)
        display.plot()
        plt.show()

    if print_report:
        print(metrics.classification_report(target_p, predict_p, target_names=['norm','pat']))

    if print_custom_report:
        print("Accuracy={}".format(acc))
        print("Sensitivity={}".format(sensi))
        print("Especificity={}".format(especi))
        print("Precision={}".format(preci))
        print("F1={}".format(f1))

    return acc, sensi, especi, preci, f1, Npatients, target, pre, score

def joint_prob(target_pat,predict_pat):
    target = target_pat[0]
    predict_norm = 1 - predict_pat
    score = np.sum(np.log(predict_pat) - np.log(predict_norm))
    if score >= 0:
        pre = 1
    else:
        pre = 0

    return target, pre, score