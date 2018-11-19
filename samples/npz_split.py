#!/usr/bin/python

import argparse
import numpy as np
import os, sys, signal, glob, time
import pydvs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice_in',
                        type=str,
                        required=True)
    parser.add_argument('--slice_out',
                        type=str,
                        required=True)
    parser.add_argument('--t1',
                        type=float,
                        required=True)
    parser.add_argument('--t2',
                        type=float,
                        required=True)

    args = parser.parse_args()

    print "Opening", args.slice_in

    sl_npz = np.load(args.slice_in)
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    K              = sl_npz['K']
    D              = sl_npz['D']

    first_ts = cloud[0][0]
    last_ts = cloud[-1][0]

    print "The recording range:", first_ts, "-", last_ts
    print "The gt range:", gt_ts[0], "-", gt_ts[-1]
    print "gt frame count:", len(gt_ts)
    print "Discretization resolution:", discretization
    if (args.t1 < first_ts or args.t2 > last_ts):
        print "The time boundaries have to be within range"
        exit(0)

    width = args.t2 - args.t1
    sl, idx_, t0 = pydvs.get_slice(cloud, idx, args.t1, width, 0, discretization)
    t1 = t0 + sl[-1][0] - sl[0][0] # The t1 - t2 ragne can be shifted due to discretization

    idx_lo = 0
    for i, t in enumerate(gt_ts):
        if t > t0:
           idx_lo = i
           break
    idx_hi = 0
    for i, t in enumerate(gt_ts):
        if t > t1:
           idx_hi = i
           break

    print "Saving", depth_gt_.shape[0], "gt slices"

    np.savez_compressed(args.slice_out, events=sl, index=idx_, 
                        discretization=discretization, K=K, D=D)
