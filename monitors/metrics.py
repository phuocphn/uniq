import torch
import torch.nn as nn


def write_metrics(writer,  epoch, net,  wt_optimizer, train_loss, train_acc1, test_loss, test_acc1, prefix="Train"):

    writer.add_scalar('%s_Train/Loss' % (prefix), train_loss, epoch)
    writer.add_scalar('%s_Train/Acc1'% (prefix), train_acc1, epoch)
    writer.add_scalar('%s_Test/Loss' % (prefix), test_loss, epoch)
    writer.add_scalar('%s_Test/Acc1' % (prefix), test_acc1, epoch)    
    writer.add_scalar('%s_Train/LR' % (prefix), wt_optimizer.param_groups[0]['lr'], epoch)

    for n, param in net.named_parameters():
        if ".delta" in n:
            if param.ndim == 0:
                writer.add_scalar('{}_Train/delta_{}'.format(prefix, n), param, epoch)  
            else:
                writer.add_histogram('{}_Train/delta_{}'.format(prefix, n), param, epoch)  

    # Weight Histogram
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            writer.add_histogram('{}_Train/{}.weight'.format(prefix, n), m.weight, epoch)
            writer.add_histogram('{}_Train/{}.weight.grad'.format(prefix, n), m.weight.grad, epoch)

            if m.bias != None:
                writer.add_histogram('{}_Train/{}.bias'.format(prefix, n), m.bias, epoch)
                writer.add_histogram('{}_Train/{}.bias.grad'.format(prefix, n), m.bias.grad, epoch)