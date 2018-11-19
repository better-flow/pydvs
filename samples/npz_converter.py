#!/usr/bin/python3

import argparse
import numpy as np
import os, sys
import pydvs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_file',
                        type=str,
                        required=True)
    parser.add_argument('--output_file',
                        type=str,
                        required=True)
    parser.add_argument('--discretization',
                        type=float,
                        required=False,
                        default=0.01)
    parser.add_argument('--calib',
                        type=str,
                        required=False,
                        default='')


    args = parser.parse_args()

    print ("Opening event file:", pydvs.okg(args.event_file))

    cloud, idx = pydvs.read_event_file_txt(args.event_file, args.discretization)

    K = None
    D = None
    if ('txt' in args.calib):
        K, D = pydvs.read_calib_txt(args.calib)
    elif ('yaml' in args.calib):
        K, D = pydvs.read_calib_yaml(args.calib)
    elif (args.calib != ''):
        print (pydvs.bld(pydvs.err("Error: Unknown calibration file format!")), args.calib)

    print ("Saving...")
    np.savez_compressed(args.output_file, events=cloud, index=idx, 
        discretization=args.discretization, K=K, D=D)

    print ("Done.")
