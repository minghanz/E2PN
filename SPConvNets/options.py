
import vgtk


parser = vgtk.HierarchyArgmentParser()

######### Experiment arguments
exp_args = parser.add_parser("experiment")
exp_args.add_argument('--experiment-id', type=str, default='playground',
                      help='experiment id (subpath after model_dir)')
exp_args.add_argument('-d', '--dataset-path', type=str, required=True,
                      help='path to datasets')
exp_args.add_argument('--dataset', type=str, default='kpts',
                      help='name of the datasets')
exp_args.add_argument('--model-dir', type=str, default='trained_models/models',
                      help='path to models (the root of all outputs)')
exp_args.add_argument('-s', '--seed', type=int, default=2913,
                      help='random seed')
exp_args.add_argument('--run-mode', type=str, default='train',
                      help='train | eval | test')

######### Network arguments
net_args = parser.add_parser("model")
net_args.add_argument('-m', '--model', type=str, default='inv_so3net_pn',
                      help='type of model to use')
net_args.add_argument('--input-num', type=int, default=1024,
                      help='the number of the input points')
net_args.add_argument('--output-num', type=int, default=32,
                      help='the number of the input points')
net_args.add_argument('--search-radius', type=float, default=0.4)
net_args.add_argument('--normalize-input', action='store_true',
                      help='normalize the input points')
net_args.add_argument('--dropout-rate', type=float, default=0.,
                      help='dropout rate, no dropout if set to 0')
net_args.add_argument('--init-method', type=str, default="xavier",
                      help='method for weight initialization')
net_args.add_argument('-k','--kpconv', action='store_true', help='If set, use a kpconv structure instead')
net_args.add_argument('--kanchor', type=int, default=60, help='# of anchors used: {1,20,40,60}')
net_args.add_argument('--normals', action='store_true', help='If set, add normals to the input (default setting is false)')
net_args.add_argument('-u', '--flag', type=str, default='attention',
                      help='pooling method: max | mean | attention | permutation')
net_args.add_argument('--representation', type=str, default='quat',
                      help='how to represent rotation: quaternion | ortho6d ')

### modelnet40 classification specific options
net_args.add_argument('--feat-all-anchors', action='store_true', help='If set, use features from all anchors to do retrieval')

### 3DMatch specific options
net_args.add_argument('--p-pool-first', action='store_true', help='If set, do spatial pooling before anchor pooling in InvOutBlockMVD (3DMatch)')
# net_args.add_argument('--permute', action='store_true', help='If set, do permutation in InvOutBlockMVD (3DMatch)')
net_args.add_argument('--permute-nl', action='store_true', help='If set, do permutation in InvOutBlockMVD (3DMatch)')


######### Training arguments
train_args = parser.add_parser("train")
train_args.add_argument('-e', '--num-epochs', type=int, default=None,
                        help='maximum number of training epochs')
train_args.add_argument('-i', '--num-iterations', type=int, default=1000000,
                        help='maximum number of training iterations')
train_args.add_argument('-b', '--batch-size', type=int, default=8,
                        help='batch size to train')
train_args.add_argument('--npt', type=int, default=24,
                        help='number of point per fragment')
train_args.add_argument('-t', '--num-thread', default=8, type=int,
                        help='number of threads for loading data')
train_args.add_argument('--no-augmentation', action="store_true",
                        help='no data augmentation if set true')
train_args.add_argument('-r','--resume-path', type=str, default=None,
                        help='Training using the pre-trained model')
train_args.add_argument('--save-freq', type=int, default=20000,
                        help='the frequency of saving the checkpoint (iters)')
train_args.add_argument('-lf','--log-freq', type=int, default=100,
                        help='the frequency of logging training info (iters)')
train_args.add_argument('--eval-freq', type=int, default=5000,
                        help='frequency of evaluation (iters)')
train_args.add_argument('--debug-mode', type=str, default=None,
                        help='if specified, train with a certain debug procedure')


######### Learning rate arguments
lr_args = parser.add_parser("train_lr")
lr_args.add_argument('-lr', '--init-lr', type=float, default=1e-3,
                     help='the initial learning rate')
lr_args.add_argument('-lrt', '--lr-type', type=str, default='exp_decay',
                     help='learning rate schedule type: exp_decay | constant')
lr_args.add_argument('--decay-rate', type=float, default=0.5,
                     help='the rate of exponential learning rate decaying')
lr_args.add_argument('--decay-step', type=int, default=10000,
                     help='the frequency of exponential learning rate decaying')


######### Loss funtion arguments
loss_args = parser.add_parser("train_loss")
loss_args.add_argument('--temperature', type=float, default=3,
                       help='temperature in attention') # appears in modelnet models

### modelnet40 classification specific options
loss_args.add_argument('--attention-loss-type', type=str, default='no_reg',
                       help='composition of attention loss function')
loss_args.add_argument('--attention-margin', type=float, default=1.0,
                       help='weight of rotational attention loss wrt category classification loss')
loss_args.add_argument('--cls-attention-loss', action='store_true', 
                       help='if not set, only calculate category classification loss \
                             and leave rotational attention unsupervised, given model.flag=="attention"')

### modelnet40 rotation registration specific options
loss_args.add_argument('--reg-r-cls-loss', action='store_true', 
                       help='if set, also use cross entropy loss on rotation classification in registration task \
                             besides binary classification on anchors, given model.flag=="permutation"')

### 3DMatch specific options
loss_args.add_argument('--loss-type', type=str, default='soft',
                       help='type of loss function')
loss_args.add_argument('--margin', type=float, default=1.0,
                       help='margin of hard batch loss')
loss_args.add_argument('--equi-alpha', type=float, default=0.0,
                       help='weight for equivariance loss')
loss_args.add_argument('--equi-beta', type=float, default=0.0,
                       help='weight for equivariance loss, given model.flag=="permutation"')

# loss_args.add_argument('--attention-pretrain-step', type=int, default=3000,
#                        help='step for scheduled pretrain (only used in attention model)')
######### Eval arguments
eval_args = parser.add_parser("eval")

######### Test arguments
test_args = parser.add_parser("test")

opt = parser.parse_args()


opt.mode = opt.run_mode