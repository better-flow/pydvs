#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, shutil, signal, glob, time
import matplotlib.colors as colors
import pydvs, cv2


global_scale_pn = 50
global_scale_pp = 50
global_shape = (260, 346)
slice_width = 1


def clear_dir(f):
    if os.path.exists(f):
        print ("Removed directory: " + f)
        shutil.rmtree(f)
    os.makedirs(f)
    print ("Created directory: " + f)


def dvs_img(cloud, shape, K, D):
    cmb = pydvs.dvs_img(cloud, shape, K=K, D=D)

    cmb[:,:,0] *= global_scale_pp
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= global_scale_pn

    return cmb
    return cmb.astype(np.uint8)


def mask_to_color(mask):
    colors = [[255,255,0], [0,255,255], [255,0,255],
              [0,255,0],   [0,0,255],   [255,0,0]]

    cmb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    m_ = np.max(mask) + 500
    m_ = max(m_, 3500)
    i = 0
    while (m_ > 0):
        cmb[mask < m_] = np.array(colors[i % len(colors)])
        i += 1
        m_ -= 1000

    cmb[mask < 500] = np.array([0,0,0])
    return cmb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    parser.add_argument('--width',
                        type=float,
                        required=False,
                        default=0.05)
    parser.add_argument('--fps',
                        type=float,
                        required=False,
                        default=-1)
    parser.add_argument('--no_undist',
                        action='store_true',
                        required=False,
                        default=False)
    parser.add_argument('--mode',
                        type=int,
                        required=False,
                        default=0)

    args = parser.parse_args()

    print ("Opening", args.base_dir)

    sl_npz = np.load(args.base_dir + '/recording.npz')
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    slice_width = args.width

    first_ts = cloud[0][0]
    last_ts = cloud[-1][0]

    with_depth = True
    try:
        depth_gt = sl_npz['depth']
    except:
        with_depth = False

    with_mask = True
    try:
        mask_gt = sl_npz['mask']
    except:
        with_mask = False

    with_gt_ts = True
    try:
        gt_ts = sl_npz['gt_ts']
    except:
        with_gt_ts = False

    if (not with_gt_ts):
        if (args.fps <= 0.0):
            print ("No ground truth timestamps available; please specify framerate from cli!")
            sys.exit(0)
        gt_ts = np.arange(first_ts, last_ts, 1.0 / args.fps)
        with_gt_ts = True

    K = None
    D = None

    if (not args.no_undist):
        K = sl_npz['K']
        D = sl_npz['D']
        print ("K and D:")
        print (K)
        print (D)
        print ("")
    else:
        print (pydvs.wrn("Undistortion disabled"))

    slice_dir = os.path.join(args.base_dir, 'slices')
    vis_dir   = os.path.join(args.base_dir, 'vis')

    pydvs.replace_dir(slice_dir)
    pydvs.replace_dir(vis_dir)

    print ("The recording range:", first_ts, "-", last_ts)
    print ("The gt range:", gt_ts[0], "-", gt_ts[-1])
    print ("Discretization resolution:", discretization)

    for i, time in enumerate(gt_ts):
        if (time > last_ts or time < first_ts):
            continue

        if (with_depth):
            depth = pydvs.undistort_img(depth_gt[i], K, D)
            cv2.imwrite(os.path.join(slice_dir, 'depth_' + str(i).rjust(10, '0') + '.png'), depth.astype(np.uint16))

        if (with_mask):
            mask  = pydvs.undistort_img(mask_gt[i], K, D)
            cv2.imwrite(os.path.join(slice_dir, 'mask_'  + str(i).rjust(10, '0') + '.png'), mask.astype(np.uint16))

        sl, _ = pydvs.get_slice(cloud, idx, time, args.width, args.mode, discretization)

        eimg = dvs_img(sl, global_shape, K, D)
        cv2.imwrite(os.path.join(slice_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)

        cimg = eimg[:,:,0] + eimg[:,:,2]

        depth = cimg
        if (with_depth):
            nmin = np.nanmin(depth)
            nmax = np.nanmax(depth)
            depth = (depth - nmin) / (nmax - nmin) * 255
            depth = np.dstack((depth, depth * 0, cimg))

        eimg = depth
        if (with_mask):
            col_mask = mask_to_color(mask)
            eimg = np.hstack((depth, col_mask))

        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)
