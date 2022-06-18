from importlib import import_module
from SPConvNets import FragmentLoader, PointCloudPairSampler, Dataloader_3dmatch_eval
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import os.path as osp

class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        """Trainer for 3dmatch SE(3)-invariant descriptor learning. """
        super(Trainer, self).__init__(opt)

        if self.opt.train_loss.equi_alpha > 0:
            self.summary.register(['Loss', 'InvLoss', 'Pos', 'Neg', 'Acc', \
                'EquiLoss', 'EquiPos', 'EquiNeg', 'EquiAcc' ])
        elif self.opt.train_loss.equi_beta > 0:
            self.summary.register(['Loss', 'InvLoss', 'Pos', 'Neg', 'Acc', \
                'EquiLoss', 'EquiAcc' ])
        else:
            self.summary.register(['Loss', 'Pos', 'Neg', 'Acc'])

        self.epoch_counter = 0
        self.iter_counter = 0

    def _setup_datasets(self):

        if self.opt.mode == 'train':
            dataset = FragmentLoader(self.opt, self.opt.model.search_radius, kptname=self.opt.dataset, \
                                     use_normals=self.opt.model.normals, npt=self.opt.npt)

            sampler = PointCloudPairSampler(len(dataset))

            self.dataset_train = torch.utils.data.DataLoader(dataset, \
                                             batch_size=self.opt.batch_size, \
                                             shuffle=False, \
                                             sampler=sampler,
                                             num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset_train)


        if self.opt.mode == 'eval':
            self.dataset_train = None


    def _setup_eval_datasets(self, scene):
        dataset_eval = Dataloader_3dmatch_eval(self.opt, scene)
        self.dataset_eval = torch.utils.data.DataLoader(dataset_eval, \
                            batch_size=1, \
                            shuffle=False, \
                            num_workers=1)

    def _setup_model(self):
        param_outfile = osp.join(self.root_dir, "params.json")
        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)


    def _setup_metric(self):
        # self.anchors = self.model.get_anchor().to(self.opt.device)
        self.metric = vgtk.loss.TripletBatchLoss(self.opt,\
                                                 anchors = None, #self.anchors,
                                                 alpha = self.opt.train_loss.equi_alpha, 
                                                 beta = self.opt.train_loss.equi_beta,
                                                )

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset_train)):
            self._optimize(data)

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset_train)
            data = next(self.dataset_iter)
        self._optimize(data)


    def _prepare_input(self, data):
        in_tensor_src = data['src'].to(self.opt.device)
        in_tensor_tgt = data['tgt'].to(self.opt.device)
        nchannel = in_tensor_src.shape[-1]
        in_tensor_src = in_tensor_src.view(-1, self.opt.model.input_num, nchannel)
        in_tensor_tgt = in_tensor_tgt.view(-1, self.opt.model.input_num, nchannel)

        return in_tensor_src, in_tensor_tgt


    def _optimize(self, data):

        gt_T = data['T'].to(self.opt.device)

        in_tensor_src, in_tensor_tgt = self._prepare_input(data)

        y_src, yw_src = self.model(in_tensor_src)
        y_tgt, yw_tgt = self.model(in_tensor_tgt)
        
        self.optimizer.zero_grad()

        if self.opt.train_loss.equi_alpha > 0:
            self.loss, inv_info, equi_info = self.metric(y_src, y_tgt, gt_T, yw_src, yw_tgt)
            invloss, pos_loss, neg_loss, accuracy = inv_info
            equiloss, equi_accuracy, equi_pos_loss, equi_neg_loss = equi_info
        elif self.opt.train_loss.equi_beta > 0:
            gt_T_label = data['T_label'].to(self.opt.device).reshape(-1)
            self.loss, inv_info, equi_info = self.metric(y_src, y_tgt, gt_T_label, yw_src, yw_tgt)
            invloss, accuracy, pos_loss, neg_loss = inv_info
            equiloss, equi_accuracy = equi_info
        else:
            self.loss, accuracy, pos_loss, neg_loss = self.metric(y_src, y_tgt, gt_T)
            
        self.loss.backward()
        self.optimizer.step()

        # Log training stats
        if self.opt.train_loss.equi_alpha > 0:
            log_info = {
                'Loss': self.loss.item(),
                'InvLoss': invloss.item(),
                'Pos': pos_loss.item(),
                'Neg': neg_loss.item(),
                'Acc': 100 * accuracy.item(),
                'EquiLoss': equiloss.item(),
                'EquiPos': equi_pos_loss.item(),
                'EquiNeg': equi_neg_loss.item(),
                'EquiAcc': 100 * equi_accuracy.item(),
            }
        elif self.opt.train_loss.equi_beta > 0:
            log_info = {
                'Loss': self.loss.item(),
                'InvLoss': invloss.item(),
                'Pos': pos_loss.item(),
                'Neg': neg_loss.item(),
                'Acc': 100 * accuracy.item(),
                'EquiLoss': equiloss.item(),
                'EquiAcc': 100 * equi_accuracy.item(),
            }
        else:
            log_info = {
                'Loss': self.loss.item(),
                'Pos': pos_loss.item(),
                'Neg': neg_loss.item(),
                'Acc': 100 * accuracy.item(),
            }
        self.summary.update(log_info)
        self.iter_counter += 1


    def _print_running_stats(self, step):
        stats = self.summary.get()
        
        mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
        torch.cuda.reset_peak_memory_stats()
        mem_str = f', Mem: {mem_used_max_GB:.3f}GB'

        self.logger.log('Training', f'{step}: {stats}'+mem_str)
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self):
        pass

    def eval(self, select):
        '''
            3D Match evaluation.
        '''
        from SPConvNets.datasets import evaluation_3dmatch as eval3dmatch

        # set up where to store the output feature
        all_results = dict()
        for scene in select:
            assert osp.isdir(osp.join(self.opt.dataset_path, scene))
            self.logger.log('Evaluation', f"Working on scene {scene}...")
            # target_folder = osp.join('data/evaluate/3DMatch/', self.opt.experiment_id, scene, f'{self.opt.model.output_num}_dim')
            target_folder = osp.join(self.root_dir, 'data/evaluate/3DMatch/', scene, f'{self.opt.model.output_num}_dim')
            self._setup_eval_datasets(scene)
            self._generate(target_folder)
            # recalls: [tau, ratio]
            results, total_recall = eval3dmatch.evaluate_scene(self.opt.dataset_path, target_folder, scene)
            self.logger.log('Evaluation', f"Total recall: {total_recall:.4f}!")
            all_results[scene] = results        
        self._write_csv(all_results)
        self.logger.log('Evaluation', "Done!")


    def _generate(self, target_folder):
        with torch.no_grad():
            self.model.eval()
            bs = self.opt.batch_size

            self.logger.log('Generating', "---------- Evaluating the network! ------------------")

            ################### EVAL LOADER ###############################3
            from tqdm import tqdm
            for it, data in enumerate(self.dataset_eval):
                torch.cuda.reset_peak_memory_stats()

                sid = data['sid'].item()
                # scene = data['scene']

                checknan = lambda tensor: torch.sum(torch.isnan(tensor))
                
                self.logger.log('Generating', "Working on fragment id {}".format(sid))
                n_keypoints = data['clouds'].shape[0]
                # 5000 x N x 3
                clouds = data['clouds'].to(self.opt.device).squeeze()
                npt = clouds.shape[0]

                feature_buffer = []
                for bi in tqdm(range(0, npt, bs)):
                    in_tensor_test = clouds[bi : min(npt,bi+bs)]
                    feature, _ = self.model(in_tensor_test)
                    feature_np = feature.detach().cpu().numpy()
                    if checknan(feature).item() > 0:
                        feature_np = np.nan_to_num(feature_np)
                    feature_buffer.append(feature_np)
                    # print("Batch counter at %d/%d"%(bi, npt), end='\r')

                mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
                torch.cuda.reset_peak_memory_stats()
                self.logger.log('Generating', f'Mem: {mem_used_max_GB:.3f}GB')

                # target_folder = osp.join('data/evaluate/3DMatch/', self.opt.experiment_id, scene, f'{self.opt.model.output_num}_dim')
                os.makedirs(target_folder, exist_ok=True)
                feature_out = np.vstack(feature_buffer)
                out_path = osp.join(target_folder, "feature%d.npy"%sid)
                self.logger.log('Generating', f"Saving features to {out_path}")
                np.save(out_path, feature_out)
            ######################################################################


    def _write_csv(self, results):
        import csv
        from SPConvNets.datasets import evaluation_3dmatch as eval3dmatch
        # csvpath_root = osp.join('trained_models/evaluate/3DMatch/', self.opt.experiment_id)
        csvpath_root = osp.join(self.root_dir, 'data/evaluate/3DMatch/')
        os.makedirs(csvpath_root, exist_ok=True)
        csvpath = osp.join( csvpath_root, 'recall.csv')
        with open(csvpath, 'w', newline='') as csvfile:
            fieldnames = ['Scene'] + ['tau_%.2f'%tau for tau in eval3dmatch.TAU_RANGE]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for scene in results.keys():
                recalls = results[scene]
                row = dict()
                row['Scene'] = scene
                for tup in recalls:
                    tau, ratio = tup
                    row['tau_%.2f'%tau] = "%.2f"%ratio
                writer.writerow(row)

        overallpath = osp.join( csvpath_root, 'recall_overall.txt')
        with open(overallpath, 'w') as f:
            ### print out the stats
            all_recall = []
            for scene in results.keys():
                tau, ratio = results[scene][0]
                self.logger.log('Evaluation', " %s recall is %.2f at tau %.2f"%(scene, ratio, tau))
                f.write("%s recall is %.2f at tau %.2f\n"%(scene, ratio, tau))
                all_recall.append(ratio)

            avg = np.array(all_recall).mean()
            self.logger.log('Evaluation', "Average recall is %.2f !" % avg)
            f.write("Average recall is %.2f !\n" % avg)
                
