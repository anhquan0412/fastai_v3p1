from exp.nb_09b import *
import time
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
    
    def begin_fit(self):
        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
        names = ['epoch'] + [f'train_{n}' for n in met_names] + [
            f'valid_{n}' for n in met_names] + ['time']
        self.logger(names)
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time() # start timer for this epoch
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        stats = [str(self.epoch)] # self.epoch is like i, i.e. for i in range(epochs)
        for o in [self.train_stats, self.valid_stats]:           
            stats += [f'{v:.6f}' for v in o.avg_stats] 
        stats += [format_time(time.time() - self.start_time)]
        # [0,train_loss,train_metric1,train_metric2,val_loss,val_metric1,val_metric2, time for completing 1 epoch]
        # will be added to master bar (below)
        self.logger(stats)

class ProgressCallback(Callback):
    _order=-1 # to be used from the very beginning, so we can get epoch duration correctly
    def begin_fit(self):
        # master_bar handles the count over the epochs
        # create one at the beginning
        self.mbar = master_bar(range(self.epochs)) 
        self.mbar.on_iter_begin()
        
        # changing the logger of the Learner to the write function of the master bar
        # so everything will be written in the master bar (as HTML)
        self.run.logger = partial(self.mbar.write, table=True)
        
    def begin_epoch(self): self.set_pb() # create progress bar
        
    def after_batch(self): self.pb.update(self.iter) # update progress bar after an iteration
    
    def begin_validate(self): self.set_pb() # create new progress bar for validation
    
    def after_fit(self): self.mbar.on_iter_end()
        
    def set_pb(self):
        # master_bar's child, progress_bar, is looping over all the batches, and disappear when done
        self.pb = progress_bar(self.dl, parent=self.mbar, auto_update=False)
        self.mbar.update(self.epoch)