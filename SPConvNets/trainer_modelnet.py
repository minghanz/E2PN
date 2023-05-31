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
import vgtk.so3conv.functional as L
from thop import profile
from fvcore.nn import FlopCountAnalysis

def val(dataset_test, model, metric, best_acc, test_accs, device, logger, info,
        debug_mode, attention_loss, attention_loss_type, att_permute_loss):
    accs = []
    # lmc = np.zeros([40,60], dtype=np.int32)

    all_labels = []
    all_feats = []
    dataset_test.dataset.set_seed(0)
    for it, data in enumerate(tqdm(dataset_test, miniters=100, maxinterval=600)):
        in_tensors = data['pc'].to(device)
        bdim = in_tensors.shape[0]
        in_label = data['label'].to(device).reshape(-1)
        in_Rlabel = data['R_label'].to(device) if debug_mode == 'knownatt' else None

        pred, feat, x_feat = model(in_tensors, in_Rlabel)

        if attention_loss:
            in_rot_label = data['R_label'].to(device).reshape(bdim)
            loss, cls_loss, r_loss, acc, r_acc = metric(pred, in_label, feat, in_rot_label, 2000)
            attention = F.softmax(feat,1)

            if attention_loss_type == 'no_cls':
                acc = r_acc
                loss = r_loss

            # max_id = attention.max(-1)[1].detach().cpu().numpy()
            # labels = data['label'].cpu().numpy().reshape(-1)
            # for i in range(max_id.shape[0]):
            #     lmc[labels[i], max_id[i]] += 1
        elif att_permute_loss:
            in_rot_label = data['R_label'].to(device).reshape(bdim)
            in_anchor_label = data['anchor_label'].to(device)
            loss, cls_loss, r_loss, acc, r_acc = metric(pred, in_label, feat, in_rot_label, 2000, in_anchor_label)
        else:
            cls_loss, acc = metric(pred, in_label)
            loss = cls_loss

        all_labels.append(in_label.cpu().numpy())
        all_feats.append(x_feat.cpu().numpy())  # feat

        accs.append(acc.cpu())
        # ### comment out if do not need per batch print
        # logger.log("Testing", "Accuracy: %.1f, Loss: %.2f!"%(100*acc.item(), loss.item()))
        # if attention_loss or att_permute_loss:
        #     logger.log("Testing", "Rot Acc: %.1f, Rot Loss: %.2f!"%(100*r_acc.item(), r_loss.item()))

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

    # self.logger.log("Testing", 'Here to peek at the lmc')
    # self.logger.log("Testing", str(lmc))
    # import ipdb; ipdb.set_trace()
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
        self.attention_model = opt.model.flag.startswith('attention') and opt.debug_mode != 'knownatt'
        self.attention_loss = self.attention_model and opt.train_loss.cls_attention_loss
        self.att_permute_loss = opt.model.flag == 'permutation'
        if opt.group_test:
            self.rot_set = [None, 'so3'] # 'ico' 'z', 
            if opt.train_rot is None:
                ### Test the icosahedral equivariance when not using rotation augmentation in training
                self.rot_set.append('ico')
        super(Trainer, self).__init__(opt)

        if self.attention_loss or self.att_permute_loss:
            self.summary.register(['Loss', 'Acc', 'R_Loss', 'R_Acc'])
        else:
            self.summary.register(['Loss', 'Acc'])
        self.epoch_counter = 0
        self.iter_counter = 0
        if self.opt.group_test:
            # self.best_accs_ori = {None: 0, 'z': 0, 'so3': 0}
            # self.best_accs_aug = {None: 0, 'z': 0, 'so3': 0}
            # self.test_accs_ori = {None: [], 'z': [], 'so3': []}
            # self.test_accs_aug = {None: [], 'z': [], 'so3': []}
            self.best_accs_ori = dict()
            self.best_accs_aug = dict()
            self.test_accs_ori = dict()
            self.test_accs_aug = dict()
            for rot in self.rot_set:
                self.best_accs_ori[rot] = 0
                self.best_accs_aug[rot] = 0
                self.test_accs_ori[rot] = []
                self.test_accs_aug[rot] = []
        else:
            self.test_accs = []
            self.best_acc = None

        
        if self.opt.model.kanchor == 12:
            self.anchors = L.get_anchorsV()
            self.trace_idx_ori, self.trace_idx_rot = L.get_relativeV_index()
            self.trace_idx_ori = torch.tensor(self.trace_idx_ori).to(self.opt.device)
            self.trace_idx_rot = torch.tensor(self.trace_idx_rot).to(self.opt.device)
        else:
            self.anchors = L.get_anchors(self.opt.model.kanchor)

    def _setup_datasets(self):
        if self.opt.mode == 'train':
            dataset = Dataloader_ModelNet40(self.opt, rot=self.opt.train_rot)
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
            self.datasets_test_ori = dict()
            # for rot_mode in [None, 'z', 'so3']:
            for rot_mode in self.rot_set:
                dataset_test = Dataloader_ModelNet40(self.opt, 'testR', test_aug=False, rot=rot_mode)
                self.datasets_test_ori[rot_mode] = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=test_batch_size, \
                                                        shuffle=False, \
                                                        num_workers=self.opt.num_thread)
            self.datasets_test_aug = dict()
            # for rot_mode in [None, 'z', 'so3']:
            for rot_mode in self.rot_set:
                dataset_test = Dataloader_ModelNet40(self.opt, 'testR', test_aug=True, rot=rot_mode)
                self.datasets_test_aug[rot_mode] = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=test_batch_size, \
                                                        shuffle=False, \
                                                        num_workers=self.opt.num_thread)
        else:
            dataset_test = Dataloader_ModelNet40(self.opt, 'testR')
            self.dataset_test = torch.utils.data.DataLoader(dataset_test, \
                                                            batch_size=test_batch_size, \
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

            
        input = torch.randn(1, 3, 1024).to(self.opt.device)
        macs, params = profile(self.model, inputs=(input, ))
        print(
            "Batch size: 1 | params(M): %.2f | FLOPs(G) %.5f" % (params / (1000 ** 2), macs / (1000 ** 3))
        )
        input = torch.randn(12, 3, 1024).to(self.opt.device)
        macs, params = profile(self.model, inputs=(input, ))
        print(
            "Batch size: 12 | params(M): %.2f | FLOPs(G) %.5f" % (params / (1000 ** 2), macs / (1000 ** 3))
        )
        self.profiled = 1 # 0

    def _setup_metric(self):
        if self.attention_loss:
            ### loss on category and rotation classification
            self.metric = vgtk.AttentionCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin)
            # self.r_metric = AnchorMatchingLoss()
        elif self.att_permute_loss:
            ### loss on category classification and anchor alignment
            self.metric = vgtk.AttPermuteCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin, self.opt.device, \
                                                            self.opt.train_loss.anchor_ab_loss, self.opt.train_loss.cross_ab, self.opt.train_loss.cross_ab_T)
        else:
            ### loss on category classification only
            self.metric = vgtk.CrossEntropyLoss()

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset)):
            if self.opt.debug_mode == 'check_equiv':
                self._check_equivariance(data)
            else:
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

        if self.opt.debug_mode == 'check_equiv':
            self._check_equivariance(data)
        else:
            self._optimize(data)
        self.iter_counter += 1

    def cos_sim(self, f1, f2):
        ### both bc(p)a
        f1_norm = torch.norm(f1, dim=1)
        f2_norm = torch.norm(f2, dim=1)
        cos_similarity = (f1 * f2).sum(1) / (f1_norm * f2_norm)
        return cos_similarity

    def _check_equivariance(self, data):
        self.model.eval()
        in_tensors = data['pc'].to(self.opt.device)
        in_label = data['label'].to(self.opt.device).reshape(-1)
        in_Rlabel = data['R_label'].to(self.opt.device) #if self.opt.debug_mode == 'knownatt' else None #!!!!
        in_R = data['R'].to(self.opt.device)

        feat_conv, x = self.model(in_tensors, in_Rlabel)
        pred, feat, x_feat = x
        n_anchors = feat.shape[-1]
        x_feat = x_feat.reshape(x_feat.shape[0], -1, n_anchors)

        in_tensors_ori = torch.matmul(in_tensors, in_R) # B*n*3, B*3*3
        feat_conv_ori, x_ori = self.model(in_tensors_ori, in_Rlabel)  # bn, bra, b[ca]
        pred_ori, feat_ori, x_feat_ori = x_ori
        n_anchors = feat_ori.shape[-1]
        x_feat_ori = x_feat_ori.reshape(x_feat_ori.shape[0], -1, n_anchors)

        trace_idx_ori = self.trace_idx_ori[in_Rlabel.flatten()] # ba
        trace_idx_ori_p = trace_idx_ori[:,None,None].expand_as(feat_conv_ori) #bcpa
        feat_conv_align = torch.gather(feat_conv, -1, trace_idx_ori_p)

        trace_idx_ori_global = trace_idx_ori[:,None].expand_as(x_feat_ori) #bca
        x_feat_align = torch.gather(x_feat, -1, trace_idx_ori_global)

        # self.logger.log('TestEquiv', f'feat_ori: {feat_ori.shape}, x_feat_ori: {x_feat_ori.shape}')
        # self.logger.log('TestEquiv', f'x_feat: {x_feat.shape}, x_feat_from_ori: {x_feat_from_ori.shape}')
        # self.logger.log('TestEquiv', f'in_Rlabel: {in_Rlabel}, in_R: {in_R}')

        cos_sim_before = self.cos_sim(feat_conv, feat_conv_ori)
        cos_sim_after = self.cos_sim(feat_conv_align, feat_conv_ori)

        self.logger.log('TestEquiv', f'per point cos before: {cos_sim_before}, after: {cos_sim_after}')

        cos_sim_before = self.cos_sim(x_feat, x_feat_ori)
        cos_sim_after = self.cos_sim(x_feat_align, x_feat_ori)
        self.logger.log('TestEquiv', f'global cos before: {cos_sim_before}, after: {cos_sim_after}')

    def _optimize(self, data):
        in_tensors = data['pc'].to(self.opt.device)
        bdim = in_tensors.shape[0]
        in_label = data['label'].to(self.opt.device).reshape(-1)
        in_Rlabel = data['R_label'].to(self.opt.device) #if self.opt.debug_mode == 'knownatt' else None
        # import ipdb; ipdb.set_trace()

        ###################### ----------- debug only ---------------------
        # in_tensorsR = data['pcR'].to(self.opt.device)
        # import ipdb; ipdb.set_trace()
        ##################### --------------------------------------------
        if self.profiled < 1:
            self.logger.log('Profile', f'in_tensors: {in_tensors.shape}, in_Rlabel: {in_Rlabel.shape}')
            flops = FlopCountAnalysis(self.model, (in_tensors, in_Rlabel))
            self.logger.log('Profile', f'flops: {flops.total()/ (1000**3)}')
            self.logger.log('Profile', f'flops.by_module(): {flops.by_module()}')
            self.profiled +=1

        pred, feat, x_feat = self.model(in_tensors, in_Rlabel)
        # x_feat not used in training, but used in eval() for retrieval mAP

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

    def test(self, dataset=None, best_acc=None, test_accs=None, info=''):
        new_best, best_acc = self.eval(dataset, best_acc, test_accs, info)
        return new_best, best_acc

    def eval(self, dataset=None, best_acc=None, test_accs=None, info=''):
        self.logger.log('Testing','Evaluating test set!'+info)
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
            if dataset is None:
                dataset = self.dataset_test
            if best_acc is None:
                best_acc = self.best_acc
            if test_accs is None:
                test_accs = self.test_accs

            mean_acc, best_acc, new_best = val(dataset, self.model, self.metric, 
                best_acc, test_accs, self.opt.device, self.logger, info,
                self.opt.debug_mode, self.attention_loss, self.opt.train_loss.attention_loss_type, 
                self.att_permute_loss)

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
                    for key, dataset_test in self.datasets_test_ori.items():
                        info = 'ori_' + str(key)
                        new_best, self.best_accs_ori[key] = self.test(
                            dataset_test, self.best_accs_ori[key], self.test_accs_ori[key], info)
                        if new_best:
                            self.logger.log('Testing', 'New best! Saving this model. '+info)
                            self._save_network('best_'+info)
                    for key, dataset_test in self.datasets_test_aug.items():
                        info = 'aug_' + str(key)
                        new_best, self.best_accs_aug[key] = self.test(
                            dataset_test, self.best_accs_aug[key], self.test_accs_aug[key], info)
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
                