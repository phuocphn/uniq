'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def replace_all(model, replacement_dict={}):
    """
    Replace all layers in the original model with new layers corresponding to `replacement_dict`.
    E.g input example:
    replacement_dict={ nn.Conv2d : partial(NIPS2019_QConv2d, bit=args.bit) }
    """ 

    def __replace_module(model):
        for module_name in model._modules:
            m = model._modules[module_name]             

            if type(m) in replacement_dict.keys():
                if isinstance(m, nn.Conv2d):
                    new_module = replacement_dict[type(m)]
                    model._modules[module_name] = new_module(in_channels=m.in_channels, 
                            out_channels=m.out_channels, kernel_size=m.kernel_size, 
                            stride=m.stride, padding=m.padding, dilation=m.dilation, 
                            groups=m.groups, bias=(m.bias!=None))
                
                elif isinstance(m, nn.Linear):
                    new_module = replacement_dict[type(m)]
                    model._modules[module_name] = new_module(in_features=m.in_features, 
                            out_features=m.out_features,
                            bias=(m.bias!=None))

            elif len(model._modules[module_name]._modules) > 0:
                __replace_module(model._modules[module_name])

    __replace_module(model)
    return model


def replace_single_module(new_cls, current_module):
    m = current_module
    if isinstance(m, nn.Conv2d):
        return new_cls(in_channels=m.in_channels, 
                out_channels=m.out_channels, kernel_size=m.kernel_size, 
                stride=m.stride, padding=m.padding, dilation=m.dilation, 
                groups=m.groups, bias=(m.bias!=None))
    
    elif isinstance(m, nn.Linear):
        return new_cls(in_features=m.in_features, out_features=m.out_features, bias=(m.bias != None))        

    return None



def replace_module(model, replacement_dict={}, exception_dict={}, arch="presnet18"):
    """
    Replace all layers in the original model with new layers corresponding to `replacement_dict`.
    E.g input example:
    replacement_dict={ nn.Conv2d : partial(NIPS2019_QConv2d, bit=args.bit) }
    exception_dict={
        'conv1': partial(NIPS2019_QConv2d, bit=8)
        'fc': partial(NIPS2019_QLinear, bit=8)
    }
    """ 
    assert arch in ["presnet32", "presnet18", "glouncv-alexnet", "glouncv-alexnet-bn",  "postech-alexnet", "glouncv-presnet34", "glouncv-presnet50", "glouncv-mobilenetv2_w1"],\
            ("Not support this type of architecture !")

    model = replace_all(model, replacement_dict=replacement_dict)

    if arch == "presnet32":
        model.conv1 = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.conv1)
        model.fc = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.fc)

        if "__downsampling__" in exception_dict.keys():
            new_conv_cls = exception_dict['__downsampling__']
            model.layer2[0].downsample[0] = replace_single_module(new_cls=new_conv_cls, current_module=model.layer2[0].downsample[0] )
            model.layer3[0].downsample[0] = replace_single_module(new_cls=new_conv_cls, current_module=model.layer3[0].downsample[0] )

    if arch == "presnet18":
        model.conv1 = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.conv1)
        model.fc = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.fc)

        if "__downsampling__" in exception_dict.keys():
            new_conv_cls = exception_dict['__downsampling__']
            model.layer2[0].shortcut[0] = replace_single_module(new_cls=new_conv_cls, current_module=model.layer2[0].shortcut[0] )
            model.layer3[0].shortcut[0] = replace_single_module(new_cls=new_conv_cls, current_module=model.layer3[0].shortcut[0] )
            model.layer4[0].shortcut[0] = replace_single_module(new_cls=new_conv_cls, current_module=model.layer4[0].shortcut[0] )

    if arch == "glouncv-presnet34":
        model.features.init_block.conv = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block.conv)
        model.output = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.output)

        if "__downsampling__" in exception_dict.keys():
            new_conv_cls = exception_dict['__downsampling__']
            model.features.stage2.unit1.identity_conv = replace_single_module(new_cls=new_conv_cls, current_module=model.features.stage2.unit1.identity_conv )
            model.features.stage3.unit1.identity_conv = replace_single_module(new_cls=new_conv_cls, current_module=model.features.stage3.unit1.identity_conv )
            model.features.stage4.unit1.identity_conv = replace_single_module(new_cls=new_conv_cls, current_module=model.features.stage4.unit1.identity_conv )

    if arch == "glouncv-presnet50":
        model.features.init_block.conv = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block.conv)
        model.output = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.output)
        if "__downsampling__" in exception_dict.keys():
            new_conv_cls = exception_dict['__downsampling__']
            model.features.stage1.unit1.identity_conv = replace_single_module(new_cls=new_conv_cls, current_module=model.features.stage1.unit1.identity_conv )              
            model.features.stage2.unit1.identity_conv = replace_single_module(new_cls=new_conv_cls, current_module=model.features.stage2.unit1.identity_conv )              
            model.features.stage3.unit1.identity_conv = replace_single_module(new_cls=new_conv_cls, current_module=model.features.stage3.unit1.identity_conv )              
            model.features.stage4.unit1.identity_conv = replace_single_module(new_cls=new_conv_cls, current_module=model.features.stage4.unit1.identity_conv )  

    if arch in ["glouncv-alexnet", "glouncv-alexnet-bn"]:
        model.features.stage1.unit1.conv = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.stage1.unit1.conv)
        model.output.fc3 = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.output.fc3)

    if arch == "glouncv-mobilenetv2_w1":
        model.features.init_block.conv = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block.conv)
        model.output = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.output)
    return model




def get_dataloaders(dataset="cifar100", batch_size=128, data_root="~/data"):
    if dataset in ("imagenet", "imagenette"):
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))


        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)

        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)


    elif dataset == "cifar100":

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),  #ResNet20, #ResNet32 does not have enough capacity for this transformation.
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])


        trainloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(
                                                    root=data_root, train=True, download=True,
                                                    transform=transform_train), 
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=4)

        testloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(
                                                    root=data_root, train=False, download=True,
                                                    transform=transform_test), 
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=4)

    else: 
        raise NotImplementedError('Not support this type of dataset: ' + dataset)

    return trainloader, testloader


def save_checkpoint(net, lr_scheduler, optimizer, acc, epoch, filename='ckpt_best.pth'):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None \
                        else None,
        'optimizer': optimizer.state_dict() if optimizer is not None \
                        else None,
    }
    torch.save(state,  filename)

def add_weight_decay(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                print ("Skip weight decay for: ", name)
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]
