# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import pickle
from open3d import visualization as o3dv
import random
import argparse
import numpy as np
import time
import network.util as util
import network.geometric_eval as geometric_eval
import pprint
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics
import trimesh
import os
import open3d as o3d

SAVE_OBJ_FOLDER = 'eval/saveobj'


def vis_sample(gt_ho, in_ho, out_ho, mje_in=None, mje_out=None):
    hand_gt, obj_gt = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_in, obj_in = in_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_in.translate((0.0, 0.2, 0.0))
    obj_in.translate((0.0, 0.2, 0.0))

    if not args.split == 'honn':
        out_ho.hand_contact = in_ho.hand_contact
        out_ho.obj_contact = in_ho.obj_contact

    hand_out, obj_out = out_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_out.translate((0.0, 0.4, 0.0))
    obj_out.translate((0.0, 0.4, 0.0))

    geom_list = [hand_gt, obj_gt, hand_out, obj_out, hand_in, obj_in]
    geom_list.append(util.text_3d('In', pos=[-0.4, 0.2, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('Refined', pos=[-0.4, 0.4, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('GT', pos=[-0.4, 0.0, 0], font_size=40, density=2))

    if mje_in is not None:
        geom_list.append(util.text_3d('MJE in {:.2f}cm out {:.2f}cm'.format(mje_in * 100, mje_out * 100), pos=[-0.4, -0.2, 0], font_size=40, density=2))

    o3dv.draw_geometries(geom_list)

def save_sample(gt_ho, in_ho, out_ho,idx, mje_in=None, mje_out=None):
    hand_gt, obj_gt = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_in, obj_in = in_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_in.translate((0.0, 0.2, 0.0))
    obj_in.translate((0.0, 0.2, 0.0))
    ho_in = hand_in + obj_in

    if not args.split == 'honn':
        out_ho.hand_contact = in_ho.hand_contact
        out_ho.obj_contact = in_ho.obj_contact

    hand_out, obj_out = out_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_out.translate((0.0, 0.4, 0.0))
    obj_out.translate((0.0, 0.4, 0.0))
    ho_out = hand_out + obj_out
    o3d.io.write_triangle_mesh(os.path.join('/home/zzq/Zhongqun/HO/ContactOpt-main/vis/', 'ho_out_{}.ply').format(idx), ho_out)
    o3d.io.write_triangle_mesh(os.path.join('/home/zzq/Zhongqun/HO/ContactOpt-main/vis/', 'ho_in_{}.ply').format(idx), ho_in)

def calc_mean_dicts(all_dicts, phase=''):
    keys = all_dicts[0].keys()
    mean_dict = dict()
    stds = ['pen_vol']

    for k in keys:
        l = list()
        for d in all_dicts:
            l.append(d[k])
        mean_dict[k] = np.array(l).mean()

        if k in stds:
            mean_dict[k + '_std'] = np.array(l).std()

    return mean_dict


def calc_sample(ho_test, ho_gt, idx, phase='nophase'):
    stats = geometric_eval.geometric_eval(ho_test, ho_gt)

    return stats


def process_sample(sample, idx):
    gt_ho, in_ho, out_ho = sample['gt_ho'], sample['in_ho'], sample['out_ho']
    in_stats = calc_sample(in_ho, gt_ho, idx, 'before ContactOpt')
    out_stats = calc_sample(out_ho, gt_ho, idx, 'after ContactOpt')

    return in_stats, out_stats


def run_eval(args):
    in_file = 'data/optimized_{}.pkl'.format(args.split+'_'+args.model)
    runs = pickle.load(open(in_file, 'rb'))
    print('Loaded {} len {}'.format(in_file, len(runs)))

    # if args.vis or args.physics:
    #     print('Shuffling!!!')
    #     random.shuffle(runs)

    if args.partial > 0:
        runs = runs[:args.partial]

    do_parallel = not args.vis
    if do_parallel:
        all_data = Parallel(n_jobs=mp.cpu_count() - 2)(delayed(process_sample)(s, idx) for idx, s in enumerate(tqdm(runs)))
        in_all = [item[0] for item in all_data]
        out_all = [item[1] for item in all_data]
    else:
        all_data = []   # Do non-parallel
        for idx, s in enumerate(tqdm(runs)):
            all_data.append(process_sample(s, idx))

            # if args.vis:
            #     print('In vs GT\n', pprint.pformat(all_data[-1][0]))
            #     print('Out vs GT\n', pprint.pformat(all_data[-1][1]))
            #     if args.split == 'im_pred_trans':
            #         vis_sample(s['gt_ho'], s['in_ho'], s['out_ho'], mje_in=all_data[-1][0]['objalign_hand_joints'], mje_out=all_data[-1][1]['objalign_hand_joints'])
            #     else:
            #         vis_sample(s['gt_ho'], s['in_ho'], s['out_ho'], mje_in=all_data[-1][0]['unalign_hand_joints'], mje_out=all_data[-1][1]['unalign_hand_joints'])

            if args.vis:
                print('In vs GT\n', pprint.pformat(all_data[-1][0]))
                print('Out vs GT\n', pprint.pformat(all_data[-1][1]))
                if args.split == 'im_pred_trans':
                    save_sample(s['gt_ho'], s['in_ho'], s['out_ho'], mje_in=all_data[-1][0]['objalign_hand_joints'], mje_out=all_data[-1][1]['objalign_hand_joints'])
                else:
                    save_sample(s['gt_ho'], s['in_ho'], s['out_ho'], idx, mje_in=all_data[-1][0]['unalign_hand_joints'], mje_out=all_data[-1][1]['unalign_hand_joints'])

        in_all = [item[0] for item in all_data]
        out_all = [item[1] for item in all_data]

    mean_in = calc_mean_dicts(in_all, 'In vs GT')
    mean_out = calc_mean_dicts(out_all, 'Out vs GT')
    print('In vs GT\n', pprint.pformat(mean_in))
    print('Out vs GT\n', pprint.pformat(mean_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval on fitted pkl')
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--contact_f1', action='store_true')
    parser.add_argument('--pen', action='store_true')
    parser.add_argument('--saveobj', action='store_true')
    parser.add_argument('--partial', default=-1, type=int, help='Only run for n samples')
    parser.add_argument('--model', default='pointnet', type=str)
    # parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    start_time = time.time()
    run_eval(args)
    print('Eval time', time.time() - start_time)
