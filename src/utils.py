import os
import torch
import random
import numpy as np

cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(taskcla, acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    tag_acc = []
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')

            # calculate average
            task_weight = np.array([ncla for _,ncla in taskcla[0:i+1]])
            task_weight = task_weight / task_weight.sum()

            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                avg_metric = 100 * (metric[i, :i + 1]*task_weight).sum()
                print('\tAvg.:{:5.1f}% '.format(avg_metric), end='')
                if name == 'TAg Acc':
                    tag_acc.append(avg_metric)
            print()
    print('*' * 108)
    avg_tag_acc = np.array(tag_acc).mean()
    print('Average Incremental Accuracy: ', avg_tag_acc)
