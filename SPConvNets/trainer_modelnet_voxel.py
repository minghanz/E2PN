from importlib import import_module
from SPConvNets import Dataloader_ModelNet40_VoxelSmooth
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from SPConvNets.datasets.evaluation.retrieval import modelnet_retrieval_mAP
from SPConvNets.models.cls_so3net_vx import ClsSO3VoxConvModel
import time
import matplotlib.pyplot as plt
from thop import profile
from fvcore.nn import FlopCountAnalysis

class DatasetInitializer:
    def __init__(self, opt):
        self.opt = opt
        self._setup_datasets()
        log_path = 'log_init_voxel_data.txt'
        self.logger = vgtk.Logger(log_file=log_path)
        self.logger.log('DatasetInitializer', f'Logger created! Hello World!')

    def _setup_datasets(self):
        if self.opt.mode == 'train':
            dataset = Dataloader_ModelNet40_VoxelSmooth(self.opt)
            self.dataset = torch.utils.data.DataLoader(dataset, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)

        # dataset_test = Dataloader_ModelNet40_VoxelSmooth(self.opt, 'testR')
        # self.dataset_test = torch.utils.data.DataLoader(dataset_test, \
        #                                                 batch_size=self.opt.batch_size, \
        #                                                 shuffle=False, \
        #                                                 num_workers=self.opt.num_thread)

    def __call__(self):
        t = time.time()
        added = 0
        for i, batch in enumerate(self.dataset):
            # if 'flag' not in batch:
            #     added += 1
            #     self.logger.log('DatasetInitializer', f"added")
            if i % 10 == 0:
                t_new = time.time()
                self.logger.log('DatasetInitializer', f"{i} files processed for {t_new - t:.2f}s")

                # data = batch['occ'][0,0].detach().cpu().numpy()

                # data_occ = data > 0.5
                # colors  = np.empty(data_occ.shape, dtype=object)
                # colors[data_occ] = 'red'

                # # and plot everything
                # ax = plt.figure().add_subplot(projection='3d')
                # ax.voxels(data_occ, facecolors=colors, edgecolor='k')

                # plt.savefig("%04d.png"%i)
                # self.logger.log('DatasetInitializer', "plotted")


        # self.logger.log('DatasetInitializer', f"{added} files added")
        t_new = time.time()
        self.logger.log('DatasetInitializer', f"Finished train set: {i} files processed for {t_new - t:.2f}s")

        # t = time.time()
        # for i, batch in enumerate(self.dataset_test):
        #     if i % 10 == 0:
        #         t_new = time.time()
        #         self.logger.log('DatasetInitializer', f"{i} files processed for {t_new - t:.2f}s")
        # t_new = time.time()
        # self.logger.log('DatasetInitializer', f"Finished testR set: {i} files processed for {t_new - t:.2f}s")


def val(dataset_test, model, metric, best_acc, test_accs, device, logger, info):

    accs = []
    all_labels = []
    all_feats = []

    for it, data in enumerate(tqdm(dataset_test, miniters=100,maxinterval=600)):
        in_tensors = data['occ'].to(device)
        bdim = in_tensors.shape[0]
        in_label = data['label'].to(device).reshape(-1)
        
        pred, x_feat = model(in_tensors)
        cls_loss, acc = metric(pred, in_label)
        loss = cls_loss

        all_labels.append(in_label.cpu().numpy())
        all_feats.append(x_feat.cpu().numpy())  # feat
        accs.append(acc.cpu())
        # logger.log("Testing", "Accuracy: %.1f, Loss: %.2f!"%(100*acc.item(), loss.item()))

    accs = np.array(accs, dtype=np.float32)
    mean_acc = 100*accs.mean()
    logger.log('Testing', 'Average accuracy is %.2f!!!!'%(mean_acc))
    test_accs.append(mean_acc)
    
    new_best = best_acc is None or mean_acc > best_acc
    if new_best:
        best_acc = mean_acc
    info_print = info if info == '' else info+': '
    logger.log('Testing', info_print+'Best accuracy so far is %.2f!!!!'%(best_acc))

    mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
    torch.cuda.reset_peak_memory_stats()
    logger.log('Testing', f'Mem: {mem_used_max_GB:.3f}GB')
    
    n = 1
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    logger.log("Testing", "all_feats.shape, {}, all_labels.shape, {}".format(all_feats.shape, all_labels.shape))
    mAP = modelnet_retrieval_mAP(all_feats,all_labels,n)
    logger.log('Testing', 'Mean average precision at %d is %f!!!!'%(n, mAP))

    return mean_acc, best_acc, new_best

class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        """Trainer for modelnet40 classification. """
        super(Trainer, self).__init__(opt)

        self.summary.register(['Loss', 'Acc'])
        self.epoch_counter = 0
        self.iter_counter = 0
        
        if self.opt.group_test:
            self.best_accs_ori = {None: 0, 'z': 0, 'so3': 0}
            self.test_accs_ori = {None: [], 'z': [], 'so3': []}
        else:
            self.test_accs = []
            self.best_acc = None

    def _setup_datasets(self):
        if self.opt.mode == 'train':
            dataset = Dataloader_ModelNet40_VoxelSmooth(self.opt, rot=self.opt.train_rot)
            self.dataset = torch.utils.data.DataLoader(dataset, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)

        if self.opt.mode =='train':
            test_batch_size = self.opt.test_batch_size
        else:
            test_batch_size = self.opt.batch_size

        if self.opt.group_test:
            self.datasets_test = dict()
            for rot_mode in [None, 'z', 'so3']:
                dataset_test = Dataloader_ModelNet40_VoxelSmooth(self.opt, 'testR', rot=rot_mode)
                self.datasets_test[rot_mode] = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=test_batch_size, \
                                                        shuffle=False, \
                                                        num_workers=self.opt.num_thread)
        else:
            dataset_test = Dataloader_ModelNet40_VoxelSmooth(self.opt, 'testR')
            dataset_test = torch.utils.data.DataLoader(dataset_test, \
                                                            batch_size=test_batch_size, \
                                                            shuffle=False, \
                                                            num_workers=self.opt.num_thread)


    def _setup_model(self):
        # if self.opt.mode == 'train':
        #     param_outfile = os.path.join(self.root_dir, "params.json")
        # else:
        #     param_outfile = None

        # module = import_module('SPConvNets.models')
        # self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

        self.model = ClsSO3VoxConvModel(self.opt).to(self.opt.device)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log("Training", "Total number of parameters: {}".format(pytorch_total_params))

        # input = torch.randn(1, 33, 33, 33).to(self.opt.device)
        # macs, params = profile(self.model, inputs=(input, ))
        # print(
        #     "Batch size: 1 | params(M): %.2f | FLOPs(G) %.5f" % (params / (1000 ** 2), macs / (1000 ** 3))
        # )
        # input = torch.randn(12, 33, 33, 33).to(self.opt.device)
        # macs, params = profile(self.model, inputs=(input, ))
        # print(
        #     "Batch size: 12 | params(M): %.2f | FLOPs(G) %.5f" % (params / (1000 ** 2), macs / (1000 ** 3))
        # )
        self.profiled = 0

    def _setup_metric(self):
        self.metric = vgtk.CrossEntropyLoss()
        # use cross entropy loss, but not sure about the dimension yet

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset)):
            self._optimize(data)
    
    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
            if data['label'].shape[0] < self.opt.batch_size:
                raise StopIteration
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)

        self._optimize(data)
        self.iter_counter += 1
        
    def _optimize(self, data):
        in_tensors = data['occ'].to(self.opt.device)
        bdim = in_tensors.shape[0]
        in_label = data['label'].to(self.opt.device).reshape(-1)
        
        if self.profiled < 1:
            self.profiled +=1
            self.logger.log('Profile', f'in_tensors: {in_tensors.shape}')   #b,1,33,33,33
            flops = FlopCountAnalysis(self.model, (in_tensors, ))
            self.logger.log('Profile', f'flops: {flops.total()/ (1000**3)}')
            self.logger.log('Profile', f'flops.by_module(): {flops.by_module()}')

        pred, _ = self.model(in_tensors)

        self.optimizer.zero_grad()

        cls_loss, acc = self.metric(pred, in_label)
        self.loss = cls_loss

        self.loss.backward()
        self.optimizer.step()

        log_info = {
            'Loss': cls_loss.item(),
            'Acc': 100 * acc.item(),
        }

        self.summary.update(log_info)

    def _print_running_stats(self, step):
        stats = self.summary.get()
        
        mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
        torch.cuda.reset_peak_memory_stats()
        mem_str = f', Mem: {mem_used_max_GB:.3f}GB'

        self.logger.log('Training', f'{step}: {stats}'+mem_str)
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])
        
    def test(self, dataset=None, best_acc=None, test_accs=None, info=''):
        new_best, best_acc = self.eval(dataset, best_acc, test_accs, info)
        return new_best, best_acc

    def eval(self, dataset=None, best_acc=None, test_accs=None, info=''):
        self.logger.log('Testing','Evaluating test set!'+info)
        self.model.eval()
        self.metric.eval()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            
            if dataset is None:
                dataset = self.dataset_test
            if best_acc is None:
                best_acc = self.best_acc
            if test_accs is None:
                test_accs = self.test_accs

            mean_acc, best_acc, new_best = val(dataset, self.model, self.metric, best_acc, test_accs, self.opt.device, self.logger, info)
            
        self.model.train()
        self.metric.train()

        return new_best, best_acc
        
    def train_iter(self):
        for i in range(self.opt.num_iterations+1):
            # if i == 5:
            #     break
            self.timer.set_point('train_iter')
            self.lr_schedule.step()
            self.step()
            # print({'Time': self.timer.reset_point('train_iter')})
            self.summary.update({'Time': self.timer.reset_point('train_iter')})

            if i % self.opt.log_freq == 0:
                if hasattr(self, 'epoch_counter'):
                    step = f'Epoch {self.epoch_counter}, Iter {i}'
                else:
                    step = f'Iter {i}'
                self._print_running_stats(step)

            if i > 0 and i % self.opt.eval_freq == 0:
                if self.opt.group_test:
                    for key, dataset_test in self.datasets_test.items():
                        info = 'ori_' + str(key)
                        new_best, self.best_accs_ori[key] = self.test(
                            dataset_test, self.best_accs_ori[key], self.test_accs_ori[key], info)
                        if new_best:
                            self.logger.log('Testing', 'New best! Saving this model. '+info)
                            self._save_network('best_'+info)
                else:
                    new_best, self.best_acc = self.test()
                    if new_best:
                        self.logger.log('Testing', 'New best! Saving this model. ')
                        self._save_network('best')

            if i > 0 and i % self.opt.save_freq == 0:
                self._save_network(f'Iter{i}')