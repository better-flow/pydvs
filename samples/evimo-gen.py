#!/usr/bin/python3

import argparse
import multiprocessing
from multiprocessing import Pool
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, signal, glob
import cv2
import pydvs
from tqdm import tqdm
import shutil

# https://stackoverflow.com/a/57364423
# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap


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


# aos2soa does not suffice because there are dropped poses
def frames_meta_to_arrays(all_objects_pose_list):
    objects_arrays = {}

    for objects_pose in all_objects_pose_list:
        for obj_id in objects_pose:
            if obj_id == 'ts':
                if 'ts' not in objects_arrays:
                    objects_arrays['ts'] = []
                objects_arrays['ts'].append(objects_pose['ts'])

            if obj_id == 'cam' or obj_id.isnumeric():
                if obj_id not in objects_arrays:
                    objects_arrays[obj_id] = {}
                    objects_arrays[obj_id]['ts'] = []
                    objects_arrays[obj_id]['pos'] = {}
                    objects_arrays[obj_id]['pos']['t'] = {}
                    objects_arrays[obj_id]['pos']['t']['x'] = []
                    objects_arrays[obj_id]['pos']['t']['y'] = []
                    objects_arrays[obj_id]['pos']['t']['z'] = []
                    objects_arrays[obj_id]['pos']['rpy'] = {}
                    objects_arrays[obj_id]['pos']['rpy']['r'] = []
                    objects_arrays[obj_id]['pos']['rpy']['p'] = []
                    objects_arrays[obj_id]['pos']['rpy']['y'] = []
                    objects_arrays[obj_id]['pos']['q'] = {}
                    objects_arrays[obj_id]['pos']['q']['w'] = []
                    objects_arrays[obj_id]['pos']['q']['x'] = []
                    objects_arrays[obj_id]['pos']['q']['y'] = []
                    objects_arrays[obj_id]['pos']['q']['z'] = []

                objects_arrays[obj_id]['ts'].append(objects_pose[obj_id]['ts'])
                objects_arrays[obj_id]['pos']['t']['x'].append(objects_pose[obj_id]['pos']['t']['x'])
                objects_arrays[obj_id]['pos']['t']['y'].append(objects_pose[obj_id]['pos']['t']['y'])
                objects_arrays[obj_id]['pos']['t']['z'].append(objects_pose[obj_id]['pos']['t']['z'])
                objects_arrays[obj_id]['pos']['rpy']['r'].append(objects_pose[obj_id]['pos']['rpy']['r'])
                objects_arrays[obj_id]['pos']['rpy']['p'].append(objects_pose[obj_id]['pos']['rpy']['p'])
                objects_arrays[obj_id]['pos']['rpy']['y'].append(objects_pose[obj_id]['pos']['rpy']['y'])
                objects_arrays[obj_id]['pos']['q']['w'].append(objects_pose[obj_id]['pos']['q']['w'])
                objects_arrays[obj_id]['pos']['q']['x'].append(objects_pose[obj_id]['pos']['q']['x'])
                objects_arrays[obj_id]['pos']['q']['y'].append(objects_pose[obj_id]['pos']['q']['y'])
                objects_arrays[obj_id]['pos']['q']['z'].append(objects_pose[obj_id]['pos']['q']['z'])
    return objects_arrays

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
    plottable_meta = frames_meta_to_arrays(frames_meta)
    plt.rcParams['lines.linewidth'] = 0.8
    fig, axs = plt.subplots(2 * (len(oids) + 1), 1)

    oid = 'cam'
    plottable_meta[oid]['pos']['rpy']['r'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['r'])
    plottable_meta[oid]['pos']['rpy']['p'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['p'])
    plottable_meta[oid]['pos']['rpy']['y'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['y'])

    axs[0].plot(plottable_meta['cam']['ts'], plottable_meta['cam'][tp]['t']['x'], label='X axis')
    axs[0].plot(plottable_meta['cam']['ts'], plottable_meta['cam'][tp]['t']['y'], label='Y axis')
    axs[0].plot(plottable_meta['cam']['ts'], plottable_meta['cam'][tp]['t']['z'], label='Z axis')
    axs[0].set_ylabel('camera translation (m)')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(plottable_meta['cam']['ts'], plottable_meta['cam'][tp]['rpy']['r'], label='R')
    axs[1].plot(plottable_meta['cam']['ts'], plottable_meta['cam'][tp]['rpy']['p'], label='P')
    axs[1].plot(plottable_meta['cam']['ts'], plottable_meta['cam'][tp]['rpy']['y'], label='Y')
    axs[1].set_xlabel('frame')
    axs[1].set_ylabel('camera rotation (rad)')
    axs[1].grid()
    axs[1].legend()

    for k, id_ in enumerate(oids):
        plottable_meta[oid]['pos']['rpy']['r'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['r'])
        plottable_meta[oid]['pos']['rpy']['p'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['p'])
        plottable_meta[oid]['pos']['rpy']['y'] = angle_to_absolute(plottable_meta[oid]['pos']['rpy']['y'])

        axs[2 * k + 2].plot(plottable_meta[id_]['ts'], plottable_meta[id_][tp]['t']['x'], label='X axis')
        axs[2 * k + 2].plot(plottable_meta[id_]['ts'], plottable_meta[id_][tp]['t']['y'], label='Y axis')
        axs[2 * k + 2].plot(plottable_meta[id_]['ts'], plottable_meta[id_][tp]['t']['z'], label='Z axis')
        axs[2 * k + 2].set_ylabel('object_' + str(id_) + ' translation (m)')
        axs[2 * k + 2].grid()
        axs[2 * k + 2].legend()
        axs[2 * k + 3].plot(plottable_meta[id_]['ts'], plottable_meta[id_][tp]['rpy']['r'], label='R')
        axs[2 * k + 3].plot(plottable_meta[id_]['ts'], plottable_meta[id_][tp]['rpy']['p'], label='P')
        axs[2 * k + 3].plot(plottable_meta[id_]['ts'], plottable_meta[id_][tp]['rpy']['y'], label='Y')
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
    parser.add_argument('--skip_slice_vis', action='store_true')
    parser.add_argument('--evimo2_npz', action='store_true')
    parser.add_argument('--evimo2_no_compress', action='store_true')
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

    oids = {}
    for frame in frames_meta:
        for key in frame:
            if (key == 'cam'): continue
            if (type(frame[key]) == type(dict()) and 'pos' in frame[key]):
                if key not in oids:
                    oids[key] = None
    oids = list(oids.keys())

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


    if args.evimo2_npz:
        pydvs.replace_dir(os.path.join(args.base_dir, 'depth_npy'))
        pydvs.replace_dir(os.path.join(args.base_dir, 'mask_npy'))
        pydvs.replace_dir(os.path.join(args.base_dir, 'classical_npy'))
    else:
        # For original EVIMO npz format, there is no easy way around these big arrays
        # because the entire array needs to be passed to savez_compressed
        # however, we can memory map them, so at least we do not run out of RAM
        depths    = np.memmap(os.path.join(args.base_dir, 'dataset_depth.mm'), mode='w+', shape=(NUM_FRAMES,) + (RES_Y, RES_X), dtype=np.uint16)
        masks     = np.memmap(os.path.join(args.base_dir, 'dataset_masks.mm'), mode='w+', shape=(NUM_FRAMES,) + (RES_Y, RES_X), dtype=np.uint16)
        classical = np.memmap(os.path.join(args.base_dir, 'dataset_classical.mm'), mode='w+', shape=(NUM_FRAMES,) + (RES_Y, RES_X, 3), dtype=np.uint8)

    def load_frame(i, frame):
        if 'gt_frame' in frame.keys():
            gt_frame_name = os.path.join(args.base_dir, frame['gt_frame'])
            gt_img = cv2.imread(gt_frame_name, cv2.IMREAD_UNCHANGED)
            if not args.evimo2_npz:
                if (gt_img.dtype != depths.dtype or gt_img.dtype != masks.dtype):
                    print ("\tType mismatch! Expected", depths.dtype, " but have", gt_img.dtype)
                    sys.exit(-1)

            if args.evimo2_npz:
                depth_name = os.path.join(args.base_dir, 'depth_npy', 'depth_' + str(i).rjust(10, '0') + '.npy')
                mask_name  = os.path.join(args.base_dir, 'mask_npy', 'mask_' + str(i).rjust(10, '0') + '.npy')
                np.save(depth_name, gt_img[:, :, 0], allow_pickle=False)
                np.save(mask_name,  gt_img[:, :, 2], allow_pickle=False)
            else:
                depths[i,:,:] = gt_img[:,:,0] # depth is in mm
                masks[i,:,:]  = gt_img[:,:,2] # mask is object ids * 1000

        if ('classical_frame' in frame.keys()):
            classical_frame_name = os.path.join(args.base_dir, frame['classical_frame'])
            classical_img = cv2.imread(classical_frame_name, cv2.IMREAD_UNCHANGED)
            if args.evimo2_npz:
                classical_name = os.path.join(args.base_dir, 'classical_npy', 'classical_' + str(i).rjust(10, '0') + '.npy')
                np.save(classical_name, classical_img, allow_pickle=False)
            else:
                classical[i,:,:,:] = classical_img

            if not args.evimo2_npz:
                if (gt_img.dtype != depths.dtype or gt_img.dtype != masks.dtype):
                    print ("\tType mismatch! Expected", classical.dtype, " but have", classical_img.dtype)
                    sys.exit(-1)
            return 1
        return 0

    num_cpu = multiprocessing.cpu_count()
    print('Using {} processes'.format(num_cpu))
    with Pool(num_cpu) as p:
        classical_read_list = list(tqdm(p.istarmap(load_frame, enumerate(frames_meta)), total=len(frames_meta)))

    classical_read = np.sum(classical_read_list)

    print ("\n")

    if (classical_read > 0):
        print (pydvs.okb("Read "), classical_read, "/", NUM_FRAMES, pydvs.okb(" classical frames"))
    else:
        if not args.evimo2_npz:
            print (pydvs.wrn("Removing mmap file: ") + pydvs.okb(os.path.join(args.base_dir, 'dataset_classical.mm')))
            classical._mmap.close()
            os.remove(os.path.join(args.base_dir, 'dataset_classical.mm'))

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
    # Save in EVIMO2 format
    if args.evimo2_npz:
        np.savez(os.path.join(args.base_dir, 'dataset_info.npz'),
                 index=idx, discretization=args.discretization, K=K, D=D, meta=dataset_txt)
        np.save(os.path.join(args.base_dir, 'dataset_events_t.npy'), cloud[:, 0])
        np.save(os.path.join(args.base_dir, 'dataset_events_xy.npy'), cloud[:, 1:3].astype(np.uint16))
        np.save(os.path.join(args.base_dir, 'dataset_events_p.npy'), cloud[:, 0].astype(np.uint8))

        # For compatibility when there are no classical frames
        if classical_read == 0:
            np.save(os.path.join(args.base_dir, 'classical_npy', 'empty.npy'), None)

        # If not compressed, the npy files can be compressed later in batches
        # which will greatly increase throughput when processing the entire dataset
        if not args.evimo2_no_compress:
            npz_pairs = [(os.path.join(args.base_dir, 'dataset_depth.npz'),     os.path.join(args.base_dir, 'depth_npy')),
                         (os.path.join(args.base_dir, 'dataset_mask.npz'),      os.path.join(args.base_dir, 'mask_npy')),
                         (os.path.join(args.base_dir, 'dataset_classical.npz'), os.path.join(args.base_dir, 'classical_npy'))]

            print (pydvs.bld("Compressing .npy into npz:"))
            print('Using {} processes'.format(len(npz_pairs)))
            compress_processes = [subprocess.Popen(['zip', '-rjq', filename, folder], cwd=args.base_dir)
                                      for filename, folder in npz_pairs]
            [p.wait() for p in compress_processes]

    # Save in original EVIMO format (takes a long time)
    else:
        np.savez_compressed(os.path.join(args.base_dir, 'dataset.npz'), events=cloud, index=idx, classical=classical,
            discretization=args.discretization, K=K, D=D, depth=depths, mask=masks, meta=dataset_txt)
    print ("\n")

    # Generate images:
    slice_dir = os.path.join(args.base_dir, 'slices')
    vis_dir   = os.path.join(args.base_dir, 'vis')

    pydvs.replace_dir(slice_dir)
    pydvs.replace_dir(vis_dir)

    def save_visualization(i, frame):
        time = frame['ts']
        if (time > tmax or time < tmin):
            return

        if args.evimo2_npz:
            depth_name = os.path.join(args.base_dir, 'depth_npy', 'depth_' + str(i).rjust(10, '0') + '.npy')
            mask_name  = os.path.join(args.base_dir, 'mask_npy', 'mask_' + str(i).rjust(10, '0') + '.npy')

            if os.path.exists(depth_name):
                depth = np.load(depth_name)
            else:
                depth = None

            if os.path.exists(mask_name):
                mask  = np.load(mask_name)
            else:
                mask = None
        else:
            depth = depths[i]
            mask  = masks[i]

        if not args.skip_slice_vis:
            if depth is not None:
                cv2.imwrite(os.path.join(slice_dir, 'depth_' + str(i).rjust(10, '0') + '.png'), depth.astype(np.uint16))
            if mask is not None:
                cv2.imwrite(os.path.join(slice_dir, 'mask_'  + str(i).rjust(10, '0') + '.png'), mask.astype(np.uint16))

        if (cloud.shape[0] > 0):
            sl, _ = pydvs.get_slice(cloud, idx, time, args.slice_width, 1, args.discretization)
            if not args.skip_slice_vis:
                eimg = dvs_img(sl, (RES_Y, RES_X), None, None, args.slice_width, mode=0)
                cv2.imwrite(os.path.join(slice_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)

        # normalize for visualization
        if depth is not None:
            depth = depth.astype(np.float32)
            depth = (255 * (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth))).astype(np.uint8)

        if mask is not None:
            mask  = mask.astype(np.float32)
            col_mask = mask_to_color(mask)
            mask = (255 * (mask - np.nanmin(mask)) / (np.nanmax(mask) - np.nanmin(mask))).astype(np.uint8)

        if classical_read > 0:
            if args.evimo2_npz:
                classical_name = os.path.join(args.base_dir, 'classical_npy', 'classical_' + str(i).rjust(10, '0') + '.npy')
                rgb_img = np.load(classical_name).astype(np.float32)
            else:
                rgb_img = classical[i].astype(np.float32)

            if mask is not None:
                mask_more_than_0 = mask > 0
                rgb_img[mask_more_than_0] = rgb_img[mask_more_than_0] * 0.2 + col_mask[mask_more_than_0] * 0.8
                
            rgb_img = np.rot90(rgb_img, k=2)
            if depth is not None:
                depth = np.rot90(depth, k=2)
            else:
                depth = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
            eimg = np.hstack((rgb_img.astype(np.uint8), np.dstack((depth,depth,depth))))
        else:
            eimg = dvs_img(sl, (RES_Y, RES_X), None, None, args.slice_width, mode=0)
            if mask is not None:
                eimg[mask > 0] = eimg[mask > 0] * 0.5 + col_mask[mask > 0] * 0.5
            if depth is None:
                depth = np.zeros((eimg.shape[0], eimg.shape[1]), dtype=np.uint8)
            eimg = np.hstack((eimg.astype(np.uint8), np.dstack((depth,depth,depth))))
        cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), eimg)

    print (pydvs.bld("Saving sanity and visualization frames:"))
    num_cpu = multiprocessing.cpu_count()
    print('Using {} processes'.format(num_cpu))
    with Pool(num_cpu) as p:
        list(tqdm(p.istarmap(save_visualization, enumerate(frames_meta)), total=len(frames_meta)))

    print (pydvs.bld("Cleaning up intermediate files:"))
    if not args.evimo2_npz:
        print (pydvs.wrn("Removed mmap file: ") + pydvs.okb(os.path.join(args.base_dir, 'dataset_depth.mm')))
        depths._mmap.close()
        os.remove(os.path.join(args.base_dir, 'dataset_depth.mm'))

        print (pydvs.wrn("Removed mmap file: ") + pydvs.okb(os.path.join(args.base_dir, 'dataset_masks.mm')))
        masks._mmap.close()
        os.remove(os.path.join(args.base_dir, 'dataset_masks.mm'))

        if (classical_read > 0):
            print (pydvs.wrn("Removed mmap file: ") + pydvs.okb(os.path.join(args.base_dir, 'dataset_classical.mm')))
            classical._mmap.close()
            os.remove(os.path.join(args.base_dir, 'dataset_classical.mm'))

    elif args.evimo2_npz and not args.evimo2_no_compress:
        print (pydvs.wrn("Removed directory: ") + pydvs.okb(os.path.join(args.base_dir, 'depth_npy')))
        shutil.rmtree(os.path.join(args.base_dir, 'depth_npy'))

        print (pydvs.wrn("Removed directory: ") + pydvs.okb(os.path.join(args.base_dir, 'mask_npy')))
        shutil.rmtree(os.path.join(args.base_dir, 'mask_npy'))

        print (pydvs.wrn("Removed directory: ") + pydvs.okb(os.path.join(args.base_dir, 'classical_npy')))
        shutil.rmtree(os.path.join(args.base_dir, 'classical_npy'))

    print (pydvs.okg("\nDone.\n"))
