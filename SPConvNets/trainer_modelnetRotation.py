from importlib import import_module
from telnetlib import NEW_ENVIRON
from SPConvNets import Dataloader_ModelNet40, Dataloader_ModelNet40Alignment
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import torch.nn.functional as F
from sklearn.neighbors import KDTree

class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        """Trainer for modelnet40 rotation registration. """
        super(Trainer, self).__init__(opt)
        self.summary.register(['Loss', 'Reg_Loss','Mean_Err', 'R_Acc'])
        self.epoch_counter = 0
        self.iter_counter = 0
        self.test_accs = []
        self.best_acc = None

    def _setup_datasets(self):
        dataloader = Dataloader_ModelNet40Alignment # Dataloader_ModelNet40

        if self.opt.mode == 'train':
            dataset = dataloader(self.opt)
            self.dataset = torch.utils.data.DataLoader(dataset, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)

        dataset_test = dataloader(self.opt, 'testR')
        self.dataset_test = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)


    def _setup_model(self):
        if self.opt.resume_path is not None:
            splits = os.path.basename(self.opt.resume_path).split('_net_')
            self.exp_name = splits[0] + splits[1][:-4]
            print("[trainer] setting experiment id to be %s!"%self.exp_name)
        else:
            self.exp_name = None

        if self.opt.mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log("Training", "Total number of parameters: {}".format(pytorch_total_params))

    def _setup_metric(self):
        # regressor + classifier
        import vgtk.so3conv.functional as L
        # anchors = torch.from_numpy(L.get_anchors(self.opt.model.kanchor)).to(self.opt.device)
        if self.opt.model.kanchor == 12:
            anchors = L.get_anchorsV()
        else:
            anchors = L.get_anchors(self.opt.model.kanchor)
        anchors = torch.tensor(anchors).to(self.opt.device)

        s2_mode = self.opt.model.flag == 'permutation'

        if self.opt.model.representation == 'quat':
            out_channel = 4
        elif self.opt.model.representation == 'ortho6d':
            out_channel = 6
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%self.opt.model.representation)

        r_cls_loss = self.opt.train_loss.reg_r_cls_loss
        self.metric = vgtk.MultiTaskDetectionLoss(anchors, nr=out_channel, s2_mode=s2_mode, r_cls_loss=r_cls_loss)

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset)):
            self._optimize(data)

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
            if data['R_label'].shape[0] < self.opt.batch_size:
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
        in_tensors = data['pc'].to(self.opt.device)
        nb, _, npoint, _ = in_tensors.shape
        # in_tensors = torch.cat([in_tensors[:,0], in_tensors[:,1]],dim=0)
        in_rot_label = data['R_label'].to(self.opt.device).view(nb,-1)
        in_alignment = data['T'].to(self.opt.device).float()
        in_R = data['R'].to(self.opt.device).float()
        if 'anchor_label' in data:
            in_anchor_label = data['anchor_label'].to(self.opt.device)
        else:
            in_anchor_label = None

        #########################################
        # pc1 = pctk.cent(in_tensors[0]).cpu().numpy()
        # pc2 = pctk.cent(in_tensors[8]).cpu().numpy()
        # alignment = data['T'][0].numpy()
        # pc2_T = pc2 @ alignment.T
        # pctk.save_ply('vis/pc1.ply', pc1)
        # pctk.save_ply('vis/pc2.ply', pc2_T, c='r')
        # import ipdb; ipdb.set_trace()
        ###########################################

        preds, y = self.model(in_tensors)
        self.optimizer.zero_grad()

        # TODO
        self.loss, cls_loss, l2_loss, acc, error = self.metric(preds, in_rot_label, y, in_R, in_alignment, in_anchor_label)
        self.loss.backward()
        self.optimizer.step()

        # Log training stats
        log_info = {
            'Loss': self.loss.item(),
            'Reg_Loss': l2_loss.item(),
            'Mean_Err': error.mean().item(),
            'R_Acc': 100 * acc.item(),
        }

        self.summary.update(log_info)


    def _print_running_stats(self, step):
        stats = self.summary.get()
        
        mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
        torch.cuda.reset_peak_memory_stats()
        mem_str = f', Mem: {mem_used_max_GB:.3f}GB'
        
        self.logger.log('Training', f'{step}: {stats}'+mem_str)
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self):
        new_best = self.eval()
        return new_best

    def eval(self):
        self.logger.log('Testing','Evaluating test set!')
        self.model.eval()
        self.metric.eval()
        torch.cuda.reset_peak_memory_stats()

        all_error = []
        all_acc = []

        with torch.no_grad():
            for it, data in enumerate(self.dataset_test):
                in_tensors = data['pc'].to(self.opt.device)
                nb, _, npoint, _ = in_tensors.shape
                # in_tensors = torch.cat([in_tensors[:,0], in_tensors[:,1]],dim=0)
                in_rot_label = data['R_label'].to(self.opt.device).view(nb,-1)
                in_alignment = data['T'].to(self.opt.device).float()
                in_R = data['R'].to(self.opt.device).float()
                if 'anchor_label' in data:
                    in_anchor_label = data['anchor_label'].to(self.opt.device)
                else:
                    in_anchor_label = None

                preds, y = self.model(in_tensors)
                # TODO
                self.loss, cls_loss, l2_loss, acc, error = self.metric(preds, in_rot_label, y, in_R, in_alignment, in_anchor_label)
                # all_labels.append(in_label.cpu().numpy())
                # all_feats.append(feat.cpu().numpy())
                all_acc.append(acc.cpu().numpy())
                all_error.append(error.cpu().numpy())
                self.logger.log("Testing", "Accuracy: %.1f, error: %.2f!"%(100*acc.item(), error.mean().item()))

            # import ipdb; ipdb.set_trace()
            all_error = np.concatenate(all_error, 0)
            all_acc = np.array(all_acc, dtype=np.float32)

            mean_error = np.mean(all_error) * 180 / np.pi
            mean_acc = 100 * all_acc.mean()
            self.test_accs.append(mean_error)
            new_best = self.best_acc is None or mean_error < self.best_acc
            if new_best:
                self.best_acc = mean_error

            self.logger.log('Testing', 'Average classifier acc is %.2f!!!!'%(mean_acc))
            self.logger.log('Testing', 'Median angular error is %.2f degree!!!!'%(np.median(all_error) * 180 / np.pi))
            self.logger.log('Testing', 'Mean angular error is %.2f degree!!!!'%(mean_error))
            self.logger.log('Testing', 'Max angular error is %.2f degree!!!!'%(np.amax(all_error) * 180 / np.pi))
            
            self.logger.log('Testing', 'Best mean angular error so far is %.2f degree!!!!'%(self.best_acc))
            
            mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
            torch.cuda.reset_peak_memory_stats()
            self.logger.log('Testing', f'Mem: {mem_used_max_GB:.3f}GB')
            
            if self.exp_name is not None:
                save_path = os.path.join(self.root_dir, 'data','alignment_errors', f'{self.exp_name}_error.txt')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savetxt(save_path,all_error)
            # import ipdb; ipdb.set_trace()

            # self.logger.log('Testing', 'Best accuracy so far is %.2f!!!!'%(best_acc))

        self.model.train()
        self.metric.train()
        return new_best