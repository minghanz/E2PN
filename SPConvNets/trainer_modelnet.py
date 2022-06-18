from importlib import import_module
from SPConvNets import Dataloader_ModelNet40
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from SPConvNets.datasets.evaluation.retrieval import modelnet_retrieval_mAP

class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        """Trainer for modelnet40 classification. """
        self.attention_model = opt.model.flag.startswith('attention') and opt.debug_mode != 'knownatt'
        self.attention_loss = self.attention_model and opt.train_loss.cls_attention_loss
        self.att_permute_loss = opt.model.flag == 'permutation'
        super(Trainer, self).__init__(opt)

        if self.attention_loss or self.att_permute_loss:
            self.summary.register(['Loss', 'Acc', 'R_Loss', 'R_Acc'])
        else:
            self.summary.register(['Loss', 'Acc'])
        self.epoch_counter = 0
        self.iter_counter = 0
        self.test_accs = []
        self.best_acc = None

    def _setup_datasets(self):
        if self.opt.mode == 'train':
            dataset = Dataloader_ModelNet40(self.opt)
            self.dataset = torch.utils.data.DataLoader(dataset, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)

        dataset_test = Dataloader_ModelNet40(self.opt, 'testR')
        self.dataset_test = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=False, \
                                                        num_workers=self.opt.num_thread)


    def _setup_model(self):
        if self.opt.mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log("Training", "Total number of parameters: {}".format(pytorch_total_params))

    def _setup_metric(self):
        if self.attention_loss:
            ### loss on category and rotation classification
            self.metric = vgtk.AttentionCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin)
            # self.r_metric = AnchorMatchingLoss()
        elif self.att_permute_loss:
            ### loss on category classification and anchor alignment
            self.metric = vgtk.AttPermuteCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin, self.opt.device)
        else:
            ### loss on category classification only
            self.metric = vgtk.CrossEntropyLoss()

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
        in_tensors = data['pc'].to(self.opt.device)
        bdim = in_tensors.shape[0]
        in_label = data['label'].to(self.opt.device).reshape(-1)
        in_Rlabel = data['R_label'].to(self.opt.device) if self.opt.debug_mode == 'knownatt' else None
        # import ipdb; ipdb.set_trace()

        ###################### ----------- debug only ---------------------
        # in_tensorsR = data['pcR'].to(self.opt.device)
        # import ipdb; ipdb.set_trace()
        ##################### --------------------------------------------

        pred, feat, x_feat = self.model(in_tensors, in_Rlabel)

        ##############################################
        # predR, featR = self.model(in_tensorsR, in_Rlabel)
        # print(torch.sort(featR[0,0])[0])
        # print(torch.sort(feat[0,0])[0])
        # import ipdb; ipdb.set_trace()
        ##############################################

        self.optimizer.zero_grad()

        if self.attention_loss:
            in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
            self.loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000)
        elif self.att_permute_loss:
            in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
            in_anchor_label = data['anchor_label'].to(self.opt.device)
            self.loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000, in_anchor_label)
        else:
            cls_loss, acc = self.metric(pred, in_label)
            self.loss = cls_loss

        self.loss.backward()
        self.optimizer.step()

        # Log training stats
        if self.attention_loss or self.att_permute_loss:
            log_info = {
                'Loss': cls_loss.item(),
                'Acc': 100 * acc.item(),
                'R_Loss': r_loss.item(),
                'R_Acc': 100 * r_acc.item(),
            }
        else:
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

    def test(self):
        new_best = self.eval()
        return new_best

    def eval(self):
        self.logger.log('Testing','Evaluating test set!')
        self.model.eval()
        self.metric.eval()
        torch.cuda.reset_peak_memory_stats()

        ################## DEBUG ###############################
        # for module in self.model.modules():
        #     if isinstance(module, torch.nn.modules.BatchNorm1d):
        #         module.train()
        #     if isinstance(module, torch.nn.modules.BatchNorm2d):
        #         module.train()
        #     if isinstance(module, torch.nn.modules.BatchNorm3d):
        #         module.train()
            # if isinstance(module, torch.nn.Dropout):
            #     module.train()
        #####################################################

        with torch.no_grad():
            accs = []
            # lmc = np.zeros([40,60], dtype=np.int32)

            all_labels = []
            all_feats = []

            for it, data in enumerate(self.dataset_test):
                in_tensors = data['pc'].to(self.opt.device)
                bdim = in_tensors.shape[0]
                in_label = data['label'].to(self.opt.device).reshape(-1)
                in_Rlabel = data['R_label'].to(self.opt.device) if self.opt.debug_mode == 'knownatt' else None

                pred, feat, x_feat = self.model(in_tensors, in_Rlabel)

                if self.attention_loss:
                    in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
                    loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000)
                    attention = F.softmax(feat,1)

                    if self.opt.train_loss.attention_loss_type == 'no_cls':
                        acc = r_acc
                        loss = r_loss

                    # max_id = attention.max(-1)[1].detach().cpu().numpy()
                    # labels = data['label'].cpu().numpy().reshape(-1)
                    # for i in range(max_id.shape[0]):
                    #     lmc[labels[i], max_id[i]] += 1
                elif self.att_permute_loss:
                    in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
                    in_anchor_label = data['anchor_label'].to(self.opt.device)
                    loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000, in_anchor_label)
                else:
                    cls_loss, acc = self.metric(pred, in_label)
                    loss = cls_loss

                all_labels.append(in_label.cpu().numpy())
                all_feats.append(x_feat.cpu().numpy())  # feat

                accs.append(acc.cpu())
                self.logger.log("Testing", "Accuracy: %.1f, Loss: %.2f!"%(100*acc.item(), loss.item()))
                if self.attention_loss or self.att_permute_loss:
                    self.logger.log("Testing", "Rot Acc: %.1f, Rot Loss: %.2f!"%(100*r_acc.item(), r_loss.item()))

            accs = np.array(accs, dtype=np.float32)
            mean_acc = 100*accs.mean()
            self.logger.log('Testing', 'Average accuracy is %.2f!!!!'%(mean_acc))
            self.test_accs.append(mean_acc)

            new_best = self.best_acc is None or mean_acc > self.best_acc
            if new_best:
                self.best_acc = mean_acc
            self.logger.log('Testing', 'Best accuracy so far is %.2f!!!!'%(self.best_acc))

            mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
            torch.cuda.reset_peak_memory_stats()
            self.logger.log('Testing', f'Mem: {mem_used_max_GB:.3f}GB')

            # self.logger.log("Testing", 'Here to peek at the lmc')
            # self.logger.log("Testing", str(lmc))
            # import ipdb; ipdb.set_trace()
            n = 1
            all_feats = np.concatenate(all_feats, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            self.logger.log("Testing", "all_feats.shape, {}, all_labels.shape, {}".format(all_feats.shape, all_labels.shape))
            mAP = modelnet_retrieval_mAP(all_feats,all_labels,n)
            self.logger.log('Testing', 'Mean average precision at %d is %f!!!!'%(n, mAP))

        self.model.train()
        self.metric.train()

        return new_best