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
import vgtk.so3conv.functional as L

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False, 
                         points2=None, info=None, 
                         c1=None, c2=None, cm1='viridis', cm2='viridis', s1=5, s2=5):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    if c1 is not None:
        cmap1 = plt.get_cmap(cm1)   # viridis, magma
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s1, c=c1, cmap=cmap1)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    if points2 is not None:
        if c2 is not None:
            cmap2 = plt.get_cmap(cm2)
            ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], s=s2, c=c2, cmap=cmap2, marker='^')
        else:
            ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], 'r')
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.8, color='gray', linewidth=0.8
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=30, azim=45)
    if info is not None:
        plt.title("{}".format(info))

    if out_file is not None:
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

class DatasetInitializerRot:
    def __init__(self, opt):
        self.opt = opt
        self.opt.device = torch.device('cuda')
        log_path = 'log_inspect_equiv.txt'
        self.logger = vgtk.Logger(log_file=log_path)
        self.logger.log('DatasetInitializerRot', f'Logger created! Hello World!')
        self._setup_datasets()
        self._setup_model()
        
        assert opt.rot_ref_tgt # since we use f1_permute, f2

        if self.opt.model.kanchor == 12:
            self.anchors = L.get_anchorsV()
            self.anchors = torch.tensor(self.anchors).to(self.opt.device)
            self.trace_idx_ori, self.trace_idx_rot = L.get_relativeV_index()
            self.trace_idx_ori = torch.tensor(self.trace_idx_ori).to(self.opt.device)
            self.trace_idx_rot = torch.tensor(self.trace_idx_rot).to(self.opt.device)
        else:
            self.anchors = L.get_anchors(self.opt.model.kanchor)
            self.anchors = torch.tensor(self.anchors).to(self.opt.device)

    def _setup_datasets(self):
        dataloader = Dataloader_ModelNet40Alignment # Dataloader_ModelNet40

        if self.opt.mode == 'train':
            dataset = dataloader(self.opt)
            self.dataset = torch.utils.data.DataLoader(dataset, \
                                                        batch_size=1, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)
            
    def _setup_model(self):
        # if self.opt.resume_path is not None:
        #     splits = os.path.basename(self.opt.resume_path).split('_net_')
        #     self.exp_name = splits[0] + splits[1][:-4]
        #     print("[trainer] setting experiment id to be %s!"%self.exp_name)
        # else:
        #     self.exp_name = None

        # if self.opt.mode == 'train':
        #     param_outfile = os.path.join(self.root_dir, "params.json")
        # else:
        #     param_outfile = None
        param_outfile = None

        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log("Training", "Total number of parameters: {}".format(pytorch_total_params))

    def cos_sim(self, f1, f2):
        ### both bca
        f1_norm = torch.norm(f1, dim=1)
        f2_norm = torch.norm(f2, dim=1)
        cos_similarity = (f1 * f2).sum(1) / (f1_norm * f2_norm)
        return cos_similarity

    def __call__(self):
        added = 0
        for i, batch in enumerate(self.dataset):
            in_tensors = batch['pc'].to(self.opt.device)
            # confidence, quats, f1, f2_permute = self.model(in_tensors)  # f_diff b,c,r,a
            confidence, quats, f1_permute, f1, f2 = self.model(in_tensors)  # f_diff b,c,r,a
            R_label0 = batch['R_label'][0]
            delta_R = batch['R'][0]
            delta_angle = torch.acos((torch.einsum("ii", delta_R) - 1)/2) / np.pi * 180

            batch_size = in_tensors.shape[0]
            f1_align = f1_permute[torch.arange(batch_size), :, batch['R_label'].flatten()]

            cos_before = self.cos_sim(f1, f2)
            cos_after = self.cos_sim(f1_align, f2)
            self.logger.log('DatasetInitializerRot', f'cos_sim before: {cos_before}, \nafter: {cos_after}')

            # pcl = batch['pc'][0,1].detach().cpu().numpy()
            # visualize_pointcloud(pcl, c1=pcl[:,2], out_file='tmp/vis/train_%03d.png'%i)
            if i % 10 == 0:
                self.logger.log('DatasetInitializerRot', f"{i} files processed.")

                # data = batch['occ'][0,0].detach().cpu().numpy()

                # data_occ = data > 0.5
                # colors  = np.empty(data_occ.shape, dtype=object)
                # colors[data_occ] = 'red'

                # # and plot everything
                # ax = plt.figure().add_subplot(projection='3d')
                # ax.voxels(data_occ, facecolors=colors, edgecolor='k')

                # plt.savefig("%04d.png"%i)
                # self.logger.log('DatasetInitializer', "plotted")
            if i == 10:
                break   # tmp!!!

        # self.logger.log('DatasetInitializer', f"{added} files added")
        self.logger.log('DatasetInitializerRot', f"Finished train set: {i} files processed.")

        # for i, batch in enumerate(self.dataset_test):
        #     pcl = batch['pc'][0,1].detach().cpu().numpy()
        #     visualize_pointcloud(pcl, c1=pcl[:,2], out_file='tmp/vis/test_%03d.png'%i)
        #     if i % 10 == 0:
        #         self.logger.log('DatasetInitializerRot', f"{i} files processed.")
        # self.logger.log('DatasetInitializerRot', f"Finished testR set: {i} files processed.")


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
        # if self.opt.resume_path is not None:
        #     splits = os.path.basename(self.opt.resume_path).split('_net_')
        #     self.exp_name = splits[0] + splits[1][:-4]
        #     print("[trainer] setting experiment id to be %s!"%self.exp_name)
        # else:
        #     self.exp_name = None
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
        self.metric = vgtk.MultiTaskDetectionLoss(anchors, rot_ref_tgt=self.opt.rot_ref_tgt, nr=out_channel, s2_mode=s2_mode, r_cls_loss=r_cls_loss, topk=self.opt.topk, logger=self.logger)#, writer=self.writer) #, w=50)    # tmp!!!!

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
            for it, data in enumerate(tqdm(self.dataset_test)):
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
                # self.logger.log("Testing", "Accuracy: %.1f, error: %.2f!"%(100*acc.item(), error.mean().item()))

            # import ipdb; ipdb.set_trace()
            all_error = np.concatenate(all_error, 0)
            all_acc = np.array(all_acc, dtype=np.float32)

            var_error = np.var(all_error)
            std_error = np.std(all_error)
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
            self.logger.log('Testing', 'Error variance is %.3f degree^2!!!!'%(var_error))
            self.logger.log('Testing', 'Error std is %.3f degree!!!!'%(std_error))
            
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