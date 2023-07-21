import tensorflow as tf
import numpy as np
import sys, os, time, argparse, shutil, scipy, h5py, glob

parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',  type=str, default="1", help='list of visiable GPUs')
parser.add_argument('-mdl', type=str, default='241train-scaffolds-sparse2noise-L1/tomogan-it02400.h5', help='Experiment name')
parser.add_argument('-dsfn',type=str, default='../Dataset/241train-scaffolds-sparse2noise.h5', help='h5 dataset file')
parser.add_argument('-depth', type=int, default=3, help='input depth (use for 3D CT image only)')
parser.add_argument('-dnfn',type=str, default='../InferredData/241train-scaffolds-noise2inverse-120LD-1500P-S4_dn_02400.h5')
args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

mdl = tf.keras.models.load_model(args.mdl, )

with h5py.File(args.dsfn, 'r') as h5fd:
    ns_img_test = h5fd['train_ns'][:].astype(np.float32)
    # gt_img_test = h5fd['test_gt'][:]

idx = [s_idx for s_idx in range(ns_img_test.shape[0] - args.depth)]
dn_img=[]
for s_idx in idx:
    X = np.array(np.transpose(ns_img_test[s_idx : (s_idx+args.depth)], (1, 2, 0)))
    X = X[np.newaxis, :, :, :]
    Y = mdl.predict(X[:1])
    dn = Y[0,:,:,0]
    dn_img.append(dn)
# ns_img_test = np.expand_dims([ns_img_test[s_idx+args.depth//2] for s_idx in idx], 3)


if os.path.exists(args.dnfn):
    with h5py.File(args.dnfn, 'a') as h5fd:
        # h5fd.create_dataset("ns", data=ns_img_test, dtype=np.float32)
        # h5fd.create_dataset("gt", data=gt_img_test, dtype=np.float32)
        h5fd.create_dataset("dn", data=dn_img, dtype=np.float32)
        print('dn')
else:
    with h5py.File(args.dnfn,'w') as h5fd:
        h5fd.create_dataset("dn", data=dn_img, dtype=np.float32)

