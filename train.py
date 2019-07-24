#-*- encoding:utf8 -*-

import os
import time


import pickle
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn
import torch.utils.data.sampler as sampler


from utils.config import DefaultConfig
from models.deep_ppi import DeepPPI
from generator import data_generator


from evaluation import compute_roc, compute_aupr, compute_mcc, micro_score,acc_score, compute_performance

configs = DefaultConfig()
THREADHOLD = 0.2

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        xavier_normal_(m.weight.data)
    elif isinstance(m,nn.Linear):
        xavier_normal_(m.weight.data)


def train_epoch(model, loader, optimizer, epoch, all_epochs, print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()

    global THREADHOLD
    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (seq_data, pssm_data, dssp_data, local_data, label) in enumerate(loader):
        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                seq_var = torch.autograd.Variable(seq_data.cuda(async=True).float())
                pssm_var = torch.autograd.Variable(pssm_data.cuda(async=True).float())
                dssp_var = torch.autograd.Variable(dssp_data.cuda(async=True).float())
                local_var = torch.autograd.Variable(local_data.cuda(async=True).float())
                target_var = torch.autograd.Variable(label.cuda(async=True).float())
            else:
                seq_var = torch.autograd.Variable(seq_data.float())
                pssm_var = torch.autograd.Variable(pssm_data.float())
                dssp_var = torch.autograd.Variable(dssp_data.float())
                local_var = torch.autograd.Variable(local_data.float())
                target_var = torch.autograd.Variable(label.float())

        # compute output
        output = model(seq_var, dssp_var, pssm_var, local_var)
        shapes = output.data.shape
        output = output.view(shapes[0]*shapes[1])
        loss = torch.nn.functional.binary_cross_entropy(output, target_var).cuda()

        # measure accuracy and record loss
        batch_size = label.size(0)
        pred_out = output.ge(THREADHOLD)
        MiP, MiR, MiF, PNum, RNum = micro_score(pred_out.data.cpu().numpy(),
                                                target_var.data.cpu().numpy())
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'f_max:%.6f' % (MiP),
                'p_max:%.6f' % (MiR),
                'r_max:%.6f' % (MiF),
                't_max:%.2f' % (PNum)])
            print(res)

    return batch_time.avg, losses.avg


def eval_epoch(model, loader, print_freq=10, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    global THREADHOLD
    # Model on eval mode
    model.eval()

    all_trues = []
    all_preds = []
    all_gos = []
    end = time.time()
    for batch_idx, (seq_data, pssm_data, dssp_data, local_data, label) in enumerate(loader):

        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                seq_var = torch.autograd.Variable(seq_data.cuda(async=True).float())
                pssm_var = torch.autograd.Variable(pssm_data.cuda(async=True).float())
                dssp_var = torch.autograd.Variable(dssp_data.cuda(async=True).float())
                local_var = torch.autograd.Variable(local_data.cuda(async=True).float())
                target_var = torch.autograd.Variable(label.cuda(async=True).float())
            else:
                seq_var = torch.autograd.Variable(seq_data.float())
                pssm_var = torch.autograd.Variable(pssm_data.float())
                dssp_var = torch.autograd.Variable(dssp_data.float())
                local_var = torch.autograd.Variable(local_data.float())
                target_var = torch.autograd.Variable(label.float())

        # compute output
        output =  model(seq_var, dssp_var, pssm_var, local_var)
        shapes = output.data.shape
        output = output.view(shapes[0]*shapes[1])

        loss = torch.nn.functional.binary_cross_entropy(output, target_var).cuda()

        # measure accuracy and record loss
        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            ])
            print(res)
        all_trues.append(label.numpy())
        all_preds.append(output.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    auc = compute_roc(all_preds, all_trues)
    aupr = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds,all_trues)
    acc_val = acc_score(predictions_max,all_trues)
    mcc = compute_mcc(predictions_max, all_trues)
    return batch_time.avg, losses.avg, acc_val, f_max, p_max, r_max, auc, aupr,t_max, mcc


def train(class_tag,model, train_data_set, save, n_epochs=3,
          batch_size=64, lr=0.001, wd=0.0001, momentum=0.9, seed=None, num=1,
          train_file=None):

    class_tag = "all_dset"
    if seed is not None:
        torch.manual_seed(seed)
    global THREADHOLD
    # # split data
    with open(train_file,"rb") as fp:
        train_list = pickle.load(fp)

    samples_num =len(train_list)
    split_num = int(configs.splite_rate * samples_num)
    data_index = train_list
    np.random.shuffle(data_index)
    train_index = data_index[:split_num]
    eval_index = data_index[split_num:]
    train_samples = sampler.SubsetRandomSampler(train_index)
    eval_samples = sampler.SubsetRandomSampler(eval_index)



    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                               sampler=train_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                              sampler=eval_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model

    # Optimizer
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.001)

    # Start log
    with open(os.path.join(save, 'DeepPPI_results.csv'), 'w') as f:
        f.write('epoch,loss,acc,F_value, precision,recall,auc,aupr,mcc,threadhold\n')

        # Train model
        best_F = 0
        threadhold = 0
        count = 0
        for epoch in range(n_epochs):
            _, train_loss = train_epoch(
                model=model_wrapper,
                loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                all_epochs=n_epochs,
            )
            _, valid_loss, acc, f_max, p_max, r_max, auc, aupr,t_max,mcc= eval_epoch(
                model=model_wrapper,
                loader=valid_loader,
                is_test=(not valid_loader)
            )
    
            print(
            'epoch:%03d,valid_loss:%0.5f\nacc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f\n' % (
                (epoch + 1), valid_loss, acc, f_max, p_max, r_max,auc, aupr,mcc,t_max))
            if f_max > best_F:
                count = 0
                best_F = f_max
                THREADHOLD = t_max
                print("new best F_value:{0}(threadhold:{1})".format(f_max, THREADHOLD))
                torch.save(model.state_dict(), os.path.join(save, 'DeepPPI_model.dat'))
            else:
                count += 1
                if count>=5:
                    return None
            # Log results
            f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f\n' % (
                (epoch + 1), valid_loss, acc, f_max, p_max, r_max, auc, aupr,mcc,t_max))

    
    
def demo(train_data,save=None, train_num = 1, 
    ratio=None,window_size=3,splite_rate = 0.1, efficient=True,
              epochs=10, seed=None,pretrained_result=None):

    train_sequences_file = ['data_cache/{0}_sequence_data.pkl'.format(key) for key in train_data]
    train_dssp_file = ['data_cache/{0}_dssp_data.pkl'.format(key) for key in train_data]
    train_pssm_file = ['data_cache/{0}_pssm_data.pkl'.format(key) for key in train_data]
    train_label_file = ['data_cache/{0}_label.pkl'.format(key) for key in train_data]
    all_list_file = 'data_cache/all_dset_list.pkl'
    train_list_file = 'data_cache/training_list.pkl'

    
    #parameters
    batch_size = configs.batch_size

    # Datasets
    train_dataSet = data_generator.dataSet(window_size, train_sequences_file, train_pssm_file, train_dssp_file, train_label_file,
                                             all_list_file)
    # Models

    class_nums = 1
    model = DeepPPI(class_nums,window_size,ratio)
    model.apply(weight_init)

    # Train the model
    train(train_data,model=model, train_data_set=train_dataSet, save=save,
          n_epochs=epochs, batch_size=batch_size, seed=seed,num=train_num,
          train_file=train_list_file)
    print('Done!')

if __name__ == '__main__':

    ratio_list = (2,1)  #glboal:local
    path_dir = "./checkpoints/deep_ppi_saved_models"
    train_data = ["dset186","dset164","dset72"]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    for ii in range(1,5):
        demo(train_data,path_dir,ii,ratio_list)
