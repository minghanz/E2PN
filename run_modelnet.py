import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'vgtk') )

from SPConvNets.trainer_modelnet import Trainer
from SPConvNets.options import opt

if __name__ == '__main__':
    opt.model.model = "cls_so3net_pn"

    if opt.mode == 'train':
        # overriding training parameters here
        opt.batch_size = 12
        opt.train_lr.decay_rate = 0.5
        opt.train_lr.decay_step = 20000
        opt.train_loss.attention_loss_type = 'default'
        opt.num_iterations = 80000
    elif opt.mode == 'eval':
        opt.batch_size = 24

    trainer = Trainer(opt)
    if opt.mode == 'train':
        trainer.train()
    elif opt.mode == 'eval':
        trainer.eval() 
