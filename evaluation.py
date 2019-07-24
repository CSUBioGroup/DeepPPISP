#-*- encoding:utf8 -*-
#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from collections import deque
import pickle

from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_recall_curve,accuracy_score 


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_aupr(preds, labels):
    # Compute ROC curve and ROC area for each class
    p, r, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    aupr = auc(r, p)
    return aupr


def compute_mcc(preds, labels, threshold=0.5):
    preds = preds.astype(np.float64)
    labels = labels.astype(np.float64)
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def compute_performance(preds, labels):

    predictions_max = None
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        p = 0.0
        r = 0.0
        total = 0
        p_total = 0
        
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp

        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            p_total += 1
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall

        if total > 0 and p_total > 0:
            r /= total
            p /= p_total
            if p + r > 0:
                f = 2 * p * r / (p + r)
                if f_max < f:
                    f_max = f
                    p_max = p
                    r_max = r
                    t_max = threshold
                    predictions_max = predictions

    return f_max, p_max, r_max, t_max, predictions_max


def micro_score(output, label):
    N = len(output)
    total_P = np.sum(output)
    total_R = np.sum(label)
    TP = float(np.sum(output * label))
    MiP = TP / max(total_P, 1e-12)  
    MiR = TP / max(total_R, 1e-12)
    if TP==0:
        MiF = 0
    else:
        MiF = 2 * MiP * MiR / (MiP + MiR)
    return MiP, MiR, MiF, total_P / N, total_R / N
    
def acc_score(output,label):
    acc = accuracy_score(label.flatten(), output.flatten())

    return acc

if __name__ == '__main__':
    pass
