import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'vgtk') )

from SPConvNets.trainer_modelnetRotation import Trainer 
from SPConvNets.options import opt

if __name__ == '__main__':
    opt.model.model = 'reg_so3net'

    if opt.mode == 'train':
        # overriding training parameters here
        opt.batch_size = 8
        opt.decay_rate = 0.97
        opt.decay_step = 3000
        opt.dropout_rate = 0.0
        opt.num_iterations = 80000

    trainer = Trainer(opt)

    if opt.mode == 'train':
        trainer.train()
    elif opt.mode == 'eval':
        trainer.eval() 
