import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import math
import argparse
import warnings
import numpy as np

from functools import partial 
from torch.utils.tensorboard import SummaryWriter
from monitors.metrics import write_metrics

import lr_scheduler 
import utils

from models.imagenet_presnet import PreActResNet18
from models.glouncv.alexnet import alexnet
from models.glouncv.preresnet import preresnet34
from models.glouncv.mobilenetv2 import mobilenetv2_w1
from models.cifar100_presnet import preact_resnet32_cifar


parser = argparse.ArgumentParser(description='PyTorch ImageNet/CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='Main learning rate')
parser.add_argument('--warmup_lr', default=0.001, type=float, help='Warmup learning rate')

parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--bit', default=4, type=int, help='bit-width for UniQ quantizer')

parser.add_argument('--dataset', default='imagenette', type=str,
                    help='dataset name for training')
parser.add_argument('--data_root', default = '/soc_local/data/pytorch/imagenet/', type=str,
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--arch', default='resnet18', type=str, 
					choices=['presnet18', 'presnet32', 'glouncv-presnet34', 'glouncv-mobilenetv2_w1'],
                    help='network architecture')

parser.add_argument('--init_from', type=str,
                    help='init weights from from checkpoint')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--epochs', default=120, type=int, help='number of training epochs')

parser.add_argument('--train_id', type=str, default= 'train-01',
                    help='training id, is used for collect experiment results')

parser.add_argument('--train_scheme', type=str, default= 'fp32', choices=['fp32', 'uniq'],
                    help='Training scheme')

parser.add_argument('--optimizer', type=str, default= 'sgd', choices=['sgd', 'adam'],
                    help='Optimizer selection.')

parser.add_argument('--output_dir', type=str, default= 'outputs',
                    help='output directory')

parser.add_argument('--print_freq', default=10, type=int, help='log print frequency.')


parser.add_argument('--quant_mode', type=str, default= 'layer_wise', choices=['layer_wise', 'kernel_wise'],
                    help='Quantization mode')

parser.add_argument('--num_calibration_batches', default=100, type=int, help='number of calibration training batches')

parser.add_argument('--enable_warmup', dest='enable_warmup', action='store_true',
                    help='Enable warm-up learning rate.')

parser.add_argument('--warmup_epochs', default=5, type=int, help='number of epochs for warm-up')

parser.add_argument('--dropout_ratio', default=0.1, type=float, help='dropout ratio for AlexNet.')


args = parser.parse_args()
print ("Script arguments:\n", args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  
working_dir = os.path.join(args.output_dir, args.train_id)
os.makedirs(working_dir, exist_ok=True)
writer = SummaryWriter(working_dir)


# Setup data.
print('==> Preparing data..')
trainloader, testloader = utils.get_dataloaders(dataset=args.dataset, batch_size=args.batch_size, data_root=args.data_root)

# Setup model
# ----------------------------------------
print('==> Building model..')
if args.dataset == "imagenet":
    models = {
        'presnet18': PreActResNet18,
        'glouncv-alexnet': alexnet,
        'glouncv-presnet34': preresnet34,
        'glouncv-mobilenetv2_w1': mobilenetv2_w1
    }
    net = models.get(args.arch, None)()

elif args.dataset == "cifar100":
    assert args.arch == "presnet32"
    net = preact_resnet32_cifar(num_classes=100)

assert net != None



# Module replacement
# ---------------------------------
if args.train_scheme.startswith("uniq"):
    from quantizer.uniq import UniQConv2d, UniQInputConv2d, UniQLinear
    if args.bit > 1:
        replacement_dict = {
                        nn.Conv2d : partial(UniQConv2d, bit=args.bit, quant_mode=args.quant_mode), 
                        nn.Linear: partial(UniQLinear, bit=args.bit, quant_mode=args.quant_mode)}
        exception_dict = {
            '__first__': partial(UniQInputConv2d, bit=8),
            '__last__': partial(UniQLinear, bit=8),
        }        

        if args.arch == "glouncv-mobilenetv2_w1":
            exception_dict['__last__'] = partial(UniQConv2d, bit=8)
        net = utils.replace_module(net, replacement_dict=replacement_dict, exception_dict=exception_dict, arch=args.arch)  

    else:
        # All settings for binary neural networks.
        assert args.wd == 0
        replacement_dict = {nn.Conv2d : partial(UniQConv2d, bit=1, quant_mode=args.quant_mode), 
                            nn.Linear: partial(UniQLinear, bit=1, quant_mode=args.quant_mode) }
        exception_dict = {
            '__first__': partial(UniQInputConv2d, bit=32),
            '__last__': partial(UniQLinear, bit=32),
            '__downsampling__': partial(UniQConv2d, bit=32, quant_mode=args.quant_mode)
        }

        if args.arch == "glouncv-mobilenetv2_w1":
            exception_dict['__last__'] = partial(UniQConv2d, bit=32)
        net = utils.replace_module(net, replacement_dict=replacement_dict, exception_dict=exception_dict, arch=args.arch) 

    # The following part is used for dropout ratio modification.
    if args.arch.startswith("glouncv-alexnet"):
        net.output.fc1.dropout = nn.Dropout(p=args.dropout_ratio, inplace=False)
        net.output.fc2.dropout = nn.Dropout(p=args.dropout_ratio, inplace=False)



net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print (net)
print ("Number of learnable parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6, "M")
time.sleep(5)



# Loading checkpoint
# -----------------------------
if args.init_from and os.path.isfile(args.init_from):
    print('==> Initializing from checkpoint: ', args.init_from)
    checkpoint = torch.load(args.init_from)
    loaded_params = {}
    for k,v in checkpoint['net'].items():
        if not k.startswith("module."):
            loaded_params["module." + k] = v
        else:
            loaded_params[k] = v

    net_state_dict = net.state_dict()
    net_state_dict.update(loaded_params)
    net.load_state_dict(net_state_dict)
else:
    warnings.warn("No checkpoint file is provided !!!")



params = utils.add_weight_decay(net, weight_decay=args.wd, skip_keys=['delta', 'alpha'])
criterion = nn.CrossEntropyLoss()

# Setup optimizer
# ----------------------------
if args.optimizer == 'sgd':
    print ("==> Use SGD optimizer")
    optimizer = optim.SGD(params, lr=args.lr,
                          momentum=0.9, weight_decay=args.wd)
elif args.optimizer == 'adam':
    print ("==> Use Adam optimizer")
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)


# Setup LR scheduler
# ----------------------------
if args.enable_warmup:
    lr_scheduler = lr_scheduler.ConstantWarmupScheduler(optimizer=optimizer, min_lr=args.warmup_lr,  total_epoch=args.warmup_epochs, after_lr=args.lr, 
                                after_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs))
else:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs )



def train(epoch, ):
    global args

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.print_freq == 0:
            print ("[Train] Epoch=", epoch,  " BatchID=", batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  \
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return (train_loss/batch_idx, correct/total)

def test(epoch):
    global best_acc, args

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % args.print_freq == 0:
                print ("[Test] Epoch=", epoch, " BatchID=", batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        utils.save_checkpoint(net, lr_scheduler, optimizer, acc, epoch, 
            filename=os.path.join(working_dir, 'ckpt_best.pth'))
        print('Saving..')
        print ('Best accuracy: ', best_acc)

    return (test_loss/batch_idx, correct/total)


def simple_initialization(num_batches=100):
    net.train()
    from quantizer.uniq import STATUS, UniQConv2d, UniQInputConv2d, UniQLinear
    for n, m in net.named_modules():
        if isinstance(m, UniQConv2d) or isinstance(m, UniQInputConv2d) or isinstance(m, UniQLinear):
            assert getattr(m, 'quan_a', None) != None
            assert getattr(m, 'quan_w', None) != None
            m.quan_a.set_init_state(STATUS.INIT_READY)
            m.quan_w.set_init_state(STATUS.INIT_READY)


    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs)
        if batch_idx + 1 == num_batches: break

    for n, m in net.named_modules():
        if isinstance(m, UniQConv2d) or isinstance(m, UniQInputConv2d) or isinstance(m, UniQLinear):
            assert getattr(m, 'quan_a', None) != None
            assert getattr(m, 'quan_w', None) != None
            m.quan_a.set_init_state(STATUS.INIT_DONE)
            m.quan_w.set_init_state(STATUS.INIT_DONE)



if args.evaluate:
    print ("==> Start evaluating ...")
    test(-1)
    exit()



# Main training
# -----------------------------------------------
# Reset to 'warmup_lr' if we are using warmup strategy.
if args.enable_warmup:
    assert args.bit == 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.warmup_lr

# Initialization 
# ------------------------------------------------
if args.bit != 32 and args.train_scheme in ["uniq", ]:
    simple_initialization(num_batches=args.num_calibration_batches)

# Training
# -----------------------------------------------
for epoch in range(start_epoch, args.epochs):
    train_loss, train_acc1 = train(epoch)
    test_loss, test_acc1 = test(epoch)

    if lr_scheduler is not None:
        lr_scheduler.step()

    write_metrics(writer, epoch, net,  \
                optimizer, train_loss, train_acc1, test_loss, test_acc1, prefix="Standard_Training")
