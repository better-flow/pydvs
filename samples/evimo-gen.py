#!/usr/bin/python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, signal, glob
import cv2
import pydvs


def mask_to_color(mask):
    colors = [[84, 71, 140],   [44, 105, 154],  [4, 139, 168],
              [13, 179, 158],  [22, 219, 147],  [131, 227, 119],
              [185, 231, 105], [239, 234, 90],  [241, 196, 83],
              [242, 158, 76],  [239, 71, 111],  [255, 209, 102],
              [6, 214, 160],   [17, 138, 178],  [7, 59, 76],
              [6, 123, 194],   [132, 188, 218], [236, 195, 11],
              [243, 119, 72],  [213, 96, 98]]

    cmb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    m_ = np.max(mask) + 500
    m_ = max(m_, 3500)

    maxoid = int(m_ / 1000)
    for i in range(maxoid):
        cutoff_lo = 1000.0 * (i + 1.0) - 5
        cutoff_hi = 1000.0 * (i + 1.0) + 5
        cmb[np.where(np.logical_and(mask>=cutoff_lo, mask<=cutoff_hi))] = np.array(colors[i % len(colors)])
    return cmb


def gen_text_stub(shape_y, meta, step=15, font_scale=0.8):
    strings = {}
    for key in sorted(meta.keys()):
        try:
            pos = meta[key]['pos']
        except:
            continue
        strings[key] = {}
        strings[key]['pt'] = ("{0:.2f}".format(pos['t']['x']) + " "
                              "{0:.2f}".format(pos['t']['y']) + " "
                              "{0:.2f}".format(pos['t']['z']))
        strings[key]['pr'] = ("{0:.2f}".format(pos['rpy']['r']) + " "
                              "{0:.2f}".format(pos['rpy']['p']) + " "
                              "{0:.2f}".format(pos['rpy']['y']))
    shape_x = int(len(strings) * step + step / 2)
    cmb = np.zeros((shape_x, shape_y, 3), dtype=np.float32)

    offst = {}
    offst['pt'] = step * 4
    offst['pr'] = offst['pt'] + step * 10

    for i, key in enumerate(sorted(strings.keys())):
        cv2.putText(cmb, key + ': ', (step // 2, step + i * step),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(cmb, '| ' + strings[key]['pt'], (offst['pt'], step + i * step),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(cmb, '| ' + strings[key]['pr'], (offst['pr'], step + i * step),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, (255,255,255), 1, cv2.LINE_AA)
    return cmb


def dvs_img(cloud, shape, K, D, slice_width, mode=0):
    cmb = pydvs.dvs_img(cloud, shape, K=K, D=D)

    ncnt = cmb[:,:,0]
    time = cmb[:,:,1]
    pcnt = cmb[:,:,2]
    cnt = pcnt + ncnt

    # Scale up to be able to save as uint8
    # For visualization only. May cause overflow if slice_width is large
    cmb[:,:,0] *= 50
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= 50

    if (mode == 1):
        cmb = np.dstack((time, pcnt, ncnt))
    return cmb


def aos2soa(aos, keys=[]):
    # Convert an array of structures (expressed as Python dict)
    # to a structure of arrays
    if (len(aos) == 0):
        return aos
    end = aos[0]
    for key in keys:
        end = end[key]
    if (type(end) == type(dict())):
        ret = {}
        for key in end:
            ret[key] = aos2soa(aos, keys + [key])
        return ret
    ret = []
    for s in aos:
        end = s
        for key in keys:
            end = end[key]
        ret.append(end)
    return ret


def angle_to_absolute(angles):
    return np.array(angles)
    ret = [angles[0]]
    offset = 0.0
    for i in range(1, len(angles)):
        if (angles[i - 1] - angles[i] > 5.5):
            offset += 2 * math.pi
        if (angles[i] - angles[i - 1] > 5.5):
            offset -= 2 * math.pi
        ret.append(angles[i] + offset)
    return np.array(ret)


def save_plot(frames_meta, oids, file_name, tp='pos'):
    plottable_meta = aos2soa(frames_meta)
    plt.rcParams['lines.linewidth'] = 0.8
    fig, axs = plt.subplots(2 * (len(oids) + 1), 1)

    oid = 'cam'
    plottable_meta[oid]['pos']['rpy']['r'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['r'])
    plottable_meta[oid]['pos']['rpy']['p'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['p'])
    plottable_meta[oid]['pos']['rpy']['y'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['y'])

    axs[0].plot(plottable_meta['ts'], plottable_meta['cam'][tp]['t']['x'], label='X axis')
    axs[0].plot(plottable_meta['ts'], plottable_meta['cam'][tp]['t']['y'], label='Y axis')
    axs[0].plot(plottable_meta['ts'], plottable_meta['cam'][tp]['t']['z'], label='Z axis')
    axs[0].set_ylabel('camera translation (m)')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(plottable_meta['ts'], plottable_meta['cam'][tp]['rpy']['r'], label='R')
    axs[1].plot(plottable_meta['ts'], plottable_meta['cam'][tp]['rpy']['p'], label='P')
    axs[1].plot(plottable_meta['ts'], plottable_meta['cam'][tp]['rpy']['y'], label='Y')
    axs[1].set_xlabel('frame')
    axs[1].set_ylabel('camera rotation (rad)')
    axs[1].grid()
    axs[1].legend()

    for k, id_ in enumerate(oids):
        plottable_meta[oid]['pos']['rpy']['r'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['r'])
        plottable_meta[oid]['pos']['rpy']['p'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['p'])
        plottable_meta[oid]['pos']['rpy']['y'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['y'])

        axs[2 * k + 2].plot(plottable_meta['ts'], plottable_meta[id_][tp]['t']['x'], label='X axis')
        axs[2 * k + 2].plot(plottable_meta['ts'], plottable_meta[id_][tp]['t']['y'], label='Y axis')
        axs[2 * k + 2].plot(plottable_meta['ts'], plottable_meta[id_][tp]['t']['z'], label='Z axis')
        axs[2 * k + 2].set_ylabel('object_' + str(id_) + ' translation (m)')
        axs[2 * k + 2].grid()
        axs[2 * k + 2].legend()
        axs[2 * k + 3].plot(plottable_meta['ts'], plottable_meta[id_][tp]['rpy']['r'], label='R')
        axs[2 * k + 3].plot(plottable_meta['ts'], plottable_meta[id_][tp]['rpy']['p'], label='P')
        axs[2 * k + 3].plot(plottable_meta['ts'], plottable_meta[id_][tp]['rpy']['y'], label='Y')
        axs[2 * k + 3].set_xlabel('frame')
        axs[2 * k + 3].set_ylabel('object_' + str(id_) + ' rotation (rad)')
        axs[2 * k + 3].grid()
        axs[2 * k + 3].legend()

    fig.set_size_inches(0.03 * 50 * (plottable_meta['ts'][-1] - plottable_meta['ts'][0]), 8 * (1 + len(oids)))
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    parser.add_argument('--discretization',
                        type=float,
                        required=False,
                        default=0.01)
    parser.add_argument('--slice_width',
                        type=float,
                        required=False,
                        default=0.05)

    args = parser.parse_args()
    print (pydvs.okb("Opening"), args.base_dir)

    dataset_txt = eval(open(os.path.join(args.base_dir, 'meta.txt')).read())

    K = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0])

    K[0][0] = dataset_txt['meta']['fx']
    K[1][1] = dataset_txt['meta']['fy']
    K[0][2] = dataset_txt['meta']['cx']
    K[1][2] = dataset_txt['meta']['cy']
    D[0] = dataset_txt['meta']['k1']
    D[1] = dataset_txt['meta']['k2']
    D[2] = dataset_txt['meta']['k3']
    D[3] = dataset_txt['meta']['k4']
    RES_X = dataset_txt['meta']['res_x']
    RES_Y = dataset_txt['meta']['res_y']
    NUM_FRAMES = len(dataset_txt['frames'])
    frames_meta = dataset_txt['frames']

    oids = []
    for key in frames_meta[0]:
        if (key == 'cam'): continue
        if (type(frames_meta[0][key]) == type(dict()) and 'pos' in frames_meta[0][key]):
            oids.append(key)

    print (pydvs.okb("Resolution:"), RES_X, 'x', RES_Y)
    print (pydvs.okb("Frames:"), NUM_FRAMES)
    print (pydvs.okb("Object ids:"), oids)
    print (pydvs.okb("Calibration:"))
    print (K)
    print (D)

    # Create a plot
    #save_plot(frames_meta, oids, os.path.join(args.base_dir, 'position_plots.pdf'), tp='pos')
    save_plot(dataset_txt['full_trajectory'], oids, os.path.join(args.base_dir, 'position_plots.pdf'), tp='pos')

    # Read depth / masks
    print (pydvs.bld("Reading the depth and masks:"))
    depths    = np.zeros((NUM_FRAMES,) + (RES_Y, RES_X))
    masks     = np.zeros((NUM_FRAMES,) + (RES_Y, RES_X))
    classical = np.zeros((NUM_FRAMES,) + (RES_Y, RES_X))
    classical_read = 0
    for i, frame in enumerate(frames_meta):
        print ("frame\t", i + 1, "/", NUM_FRAMES, "\t", end='\r')

        gt_frame_name = os.path.join(args.base_dir, frame['gt_frame'])
        gt_img = cv2.imread(gt_frame_name, cv2.IMREAD_UNCHANGED).astype(dtype=np.float32)

        depth = gt_img[:,:,0]
        mask = gt_img[:,:,2]

        depth[depth <= 10] = np.nan
        depths[i,:,:] = depth
        masks[i,:,:]  = mask

        if ('classical_frame' in frame.keys()):
            classical_frame_name = os.path.join(args.base_dir, frame['classical_frame'])
            classical[i,:,:] = cv2.imread(classical_frame_name, cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            classical_read += 1
    print ("\n")

    if (classical_read > 0):
        print (pydvs.okb("Read "), classical_read, "/", NUM_FRAMES, pydvs.okb(" classical frames"))
    else:
        classical = None

    # Read event cloud
    cloud, idx = pydvs.read_event_file_txt(os.path.join(args.base_dir, 'events.txt'), args.discretization)
    tmin = frames_meta[0]['ts']
    tmax = frames_meta[-1]['ts']
    if (cloud.shape[0] > 0):
        tmin = cloud[0][0]
        tmax = cloud[-1][0]
    print (pydvs.okb("The recording range:"), tmin, "-", tmax)
    print (pydvs.okb("The gt range:"), frames_meta[0]['ts'], "-", frames_meta[-1]['ts'])
    print (pydvs.okb("Discretization resolution:"), args.discretization)

    # Save .npz file
    print (pydvs.bld("Saving..."))
    np.savez_compressed(os.path.join(args.base_dir, 'dataset.npz'), events=cloud, index=idx, classical=classical,
        discretization=args.discretization, K=K, D=D, depth=depths, mask=masks, meta=dataset_txt)
    print ("\n")

    # Generate images:
    slice_dir = os.path.join(args.base_dir, 'slices')
    vis_dir   = os.path.join(args.base_dir, 'vis')

    pydvs.replace_dir(slice_dir)
    pydvs.replace_dir(vis_dir)
    for i, frame in enumerate(frames_meta):
        print ("Saving sanity check frames\t", i + 1, "/", NUM_FRAMES, "\t", end='\r')
        time = frame['ts']
        if (time > tmax or time < tmin):
            continue

        cv2.imwrite(os.path.join(slice_dir, 'depth_' + str(i).rjust(10, '0') + '.png'), depths[i].astype(np.uint16))
        cv2.imwrite(os.path.join(slice_dir, 'mask_'  + str(i).rjust(10, '0') + '.png'), masks[i].astype(np.uint16))

        if (cloud.shape[0] > 0):
            sl, _ = pydvs.get_slice(cloud, idx, time, args.slice_width, 1, args.discretization)
            eimg = dvs_img(sl, (RES_Y, RES_X), None, None, args.slice_width, mode=0)
            cv2.imwrite(os.path.join(slice_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)

        depth = depths[i].astype(np.float)
        mask  = masks[i].astype(np.float)
        col_mask = mask_to_color(mask)

        # normalize for visualization
        mask = (255 * (mask.astype(np.float) - np.nanmin(mask)) / (np.nanmax(mask) - np.nanmin(mask))).astype(np.uint8)
        depth = (255 * (depth.astype(np.float) - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth))).astype(np.uint8)

        if ((classical_read > 0) and (classical is not None)):
            rgb_img = np.dstack((classical[i], classical[i], classical[i]))
            rgb_img[mask > 0] = rgb_img[mask > 0] * 0.2 + col_mask[mask > 0] * 0.8
            rgb_img = np.rot90(rgb_img, k=2)
            depth = np.rot90(depth, k=2)
            eimg = np.hstack((rgb_img.astype(np.uint8), np.dstack((depth,depth,depth))))
        else:
            eimg = dvs_img(sl, (RES_Y, RES_X), None, None, args.slice_width, mode=0)
            eimg[mask > 0] = eimg[mask > 0] * 0.5 + col_mask[mask > 0] * 0.5
            eimg = np.hstack((eimg.astype(np.uint8), np.dstack((depth,depth,depth))))

        #footer = gen_text_stub(eimg.shape[1], frame)
        #eimg = np.vstack((eimg, footer))

        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)
    print (pydvs.okg("\nDone.\n"))
