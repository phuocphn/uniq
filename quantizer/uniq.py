import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class STATUS(object):
    INIT_READY = 0
    INIT_DONE = 1
    NOT_READY = -1



class UniQQuantizer(t.nn.Module):
    def __init__(self, bit, is_activation=False,  **kwargs):
        super(UniQQuantizer,self).__init__()

        self.bit = bit
        self.is_activation = is_activation
        self.delta_normal = {1: 1.595769121605729, 2: 0.9956866859435065, 3: 0.5860194414434872, 4: 0.33520061219993685, 5: 0.18813879027991698, 6: 0.10406300944201481, 7: 0.05686767238235839, 8: 0.03076238758025524, 9: 0.016498958773102656}
        self.delta_positive_normal = {1: 1.22399153, 2: 0.65076985, 3: 0.35340955, 4: 0.19324868, 5: 0.10548752, 6: 0.0572659, 7: 0.03087133, 8: 0.01652923, 9: 0.00879047}
        self.quant_mode = kwargs.get('quant_mode', 'layer_wise')
        self.layer_type = kwargs.get('layer_type', 'conv')

        if self.quant_mode == 'layer_wise': 
            self.delta = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        elif self.quant_mode == 'kernel_wise':
            assert kwargs['num_channels'] > 1
            if self.layer_type == 'conv':
                shape = [1, kwargs['num_channels'], 1, 1] if self.is_activation else [kwargs['num_channels'], 1, 1, 1]
                self.delta = nn.Parameter(torch.Tensor(*shape), requires_grad=True)
            else:
                shape = [1, kwargs['num_channels']] if self.is_activation else [kwargs['num_channels'], 1]
                self.delta = nn.Parameter(torch.Tensor(*shape), requires_grad=True)

        self.kwargs = kwargs
        self.register_buffer('init_state', torch.tensor(STATUS.NOT_READY)) 
        self.register_buffer('min_val', torch.tensor(0.0, dtype=torch.float))
        self.register_buffer('max_val', torch.tensor(2**(self.bit) - 1, dtype=torch.float)) 


    def set_init_state(self, value):
        self.init_state.fill_(value)

    def initialization(self, x):
        if self.is_activation:
            if self.quant_mode == 'kernel_wise':
                if self.layer_type == 'conv':
                    _meanx = (x.detach()**2).view(x.shape[0], -1, x.shape[2] * x.shape[3]).mean(2, True).mean(0, True).view(1, -1, 1, 1)

                elif self.layer_type == 'linear':
                    _meanx = (x.detach()**2).mean(1, True).mean(0, True).view(1, 1)

                _meanx[_meanx==0] = _meanx[_meanx!=0].min()
                pre_relu_std = ((2*_meanx))**0.5                        
            else:
                pre_relu_std = (2*((x.detach()**2).mean()))**0.5
            self.delta.data.copy_(torch.max(self.delta.data, pre_relu_std * self.delta_positive_normal[self.bit]))

        else:

            if self.quant_mode == 'kernel_wise':
                if self.layer_type == 'conv':
                    std = x.detach().view(-1, x.shape[1] * x.shape[2] * x.shape[3]).std(1, True).view(-1, 1, 1, 1)
                if self.layer_type == 'linear':
                    std = x.detach().view(-1, x.shape[1]).std(1, True).view(-1, 1)
            else:    
                std = x.detach().std()
            self.delta.data.copy_( std * self.delta_normal[self.bit])

    def forward(self, x):
        if self.training and self.init_state == STATUS.INIT_READY:
            self.initialization(x)

        # Quantization
        if self.is_activation:
            if self.quant_mode == 'kernel_wise':
                g = 1.0 / math.sqrt((x.numel() / x.shape[1]) * (2**self.bit -1))
            else:
                g = 1.0 / math.sqrt(x.numel() * (2**self.bit -1))

            step_size = grad_scale(self.delta, g)
            x = x / step_size
            x = round_pass(torch.min(torch.max(x, self.min_val), self.max_val)) * step_size
        else:

            if self.quant_mode== 'kernel_wise':
                g = 1.0 / math.sqrt((x.numel() / x.shape[0]) * max((2**(self.bit-1) -1),1))
            else:
                g = 1.0 / math.sqrt(x.numel() * max((2**(self.bit-1) -1),1))

            step_size = grad_scale(self.delta, g)
            alpha = step_size * self.max_val * 0.5
            x = (x + alpha) / step_size
            x = round_pass(torch.min(torch.max(x, self.min_val), self.max_val)) * step_size - alpha

        return x

    def extra_repr(self):
        return "bit=%s, is_activation=%s, quant_mode=%s" % \
            (self.bit, self.is_activation, self.kwargs.get('quant_mode', 'layer_wise'))



class UniQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=4, quant_mode='layer_wise'):


        super(UniQConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        # use per-channel quantization (optinal) for weights only.
        self.quan_w = UniQQuantizer(bit=bit, is_activation=False, quant_mode=quant_mode, num_channels=out_channels)
        self.quan_a = UniQQuantizer(bit=bit, is_activation=True, quant_mode='layer_wise', num_channels=in_channels)
        self.bit = bit

    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(self.quan_a(x), self.quan_w(self.weight), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class UniQInputConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=4, quant_mode='layer_wise'):


        super(UniQInputConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        #always use `layer_wise` for the first layer
        self.quan_w = UniQQuantizer(bit=bit, is_activation=False, quant_mode=quant_mode, num_channels=out_channels) 
        self.quan_a = UniQQuantizer(bit=bit, is_activation=False, quant_mode='layer_wise', num_channels=in_channels)
        self.bit = bit

    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(self.quan_a(x), self.quan_w(self.weight), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class UniQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=4, quant_mode='layer_wise'):

        super(UniQLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        #always use `layer_wise` for the last layer
        self.quan_w = UniQQuantizer(bit=bit, is_activation=False, quant_mode=quant_mode, num_channels=out_features, layer_type='linear') 
        self.quan_a = UniQQuantizer(bit=bit, is_activation=True, quant_mode='layer_wise', num_channels=in_features, layer_type='linear')
        self.bit = bit

    def forward(self, x):
        if self.bit == 32:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(self.quan_a(x), self.quan_w(self.weight), self.bias)
