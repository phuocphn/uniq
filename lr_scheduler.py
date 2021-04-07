class ConstantWarmupScheduler(object):

    def __init__(self, optimizer, min_lr=0.001, total_epoch=5, after_lr=0.01, after_scheduler=None):
        self.optimizer = optimizer
        self.total_epoch = total_epoch
        self.min_lr = min_lr
        self.after_lr = after_lr
        self.after_scheduler = after_scheduler
        self._current_epoch = 0
        super(ConstantWarmupScheduler, self).__init__()

    def step(self):
        if self._current_epoch < self.total_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.min_lr
        else:
            if self._current_epoch  == self.total_epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.after_lr
                    
            self.after_scheduler.step()
        self._current_epoch += 1


    def state_dict(self):
        self.after_scheduler.state_dict() \
                if self._current_epoch >= self.total_epoch else None