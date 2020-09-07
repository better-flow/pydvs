import sys, os, shutil
import yaml
import cv2
import numpy as np
from math import fabs, sqrt

with_rosbag = True
try:
    import rosbag
except:
    with_rosbag = False

# The dvs-related functionality implemented in C.
import cpydvs


class bcolors:
    HEADER = '\033[95m'
    PLAIN = '\033[37m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def offset(str_, p_offset):
    for i in range(p_offset):
        str_ = '...' + str_
    return str_

def hdr(str_, p_offset=0):
    return offset(bcolors.HEADER + str_ + bcolors.ENDC, p_offset)

def wht(str_, p_offset=0):
    return offset(bcolors.PLAIN + str_ + bcolors.ENDC, p_offset)

def okb(str_, p_offset=0):
    return offset(bcolors.OKBLUE + str_ + bcolors.ENDC, p_offset)

def okg(str_, p_offset=0):
    return offset(bcolors.OKGREEN + str_ + bcolors.ENDC, p_offset)

def wrn(str_, p_offset=0):
    return offset(bcolors.WARNING + str_ + bcolors.ENDC, p_offset)

def err(str_, p_offset=0):
    return offset(bcolors.FAIL + str_ + bcolors.ENDC, p_offset)

def bld(str_, p_offset=0):
    return offset(bcolors.BOLD + str_ + bcolors.ENDC, p_offset)


def ensure_dir(f):
    if not os.path.exists(f):
        print (okg("Created directory: ") + okb(f))
        os.makedirs(f)

def replace_dir(f):
    if os.path.exists(f):
        print (wrn("Removed directory: ") + okb(f))
        shutil.rmtree(f)
    os.makedirs(f)
    print (okg("Created directory: ") + okb(f))


def read_calib_yaml(fname):
    K = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0])

    cam_file = open(fname)
    cam_data = yaml.safe_load(cam_file)
    cam_file.close()

    K[0][0] = cam_data['cam_fx']
    K[1][1] = cam_data['cam_fy']
    K[0][2] = cam_data['cam_cx']
    K[1][2] = cam_data['cam_cy']

    return K, D


def read_calib_txt(fname):
    K = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0])

    lines = []
    with open(fname) as calib:
        lines = calib.readlines()

    # A single line: fx, fy, xc, cy, k1...k4
    if (len(lines) == 1):
        calib = lines[0].split(' ')
        K[0][0] = calib[0]
        K[1][1] = calib[1]
        K[0][2] = calib[2]
        K[1][2] = calib[3]
        D[0] = calib[4]
        D[1] = calib[5]
        D[2] = calib[6]
        D[3] = calib[7]
        return K, D

    K_txt = lines[0:3]
    D_txt = lines[4]

    for i, line in enumerate(K_txt):
        for j, num_txt in enumerate(line.split(' ')[0:3]):
            K[i][j] = float(num_txt)

    for j, num_txt in enumerate(D_txt.split(' ')[0:4]):
        D[j] = float(num_txt)

    return K, D


def get_index(cloud, index_w):
    print (okb("Indexing..."))

    idx = [0]
    if (cloud.shape[0] < 2):
        return np.array(idx, dtype=np.uint32)

    last_ts = cloud[0][0]
    for i, e in enumerate(cloud):
        sys.stdout.write("\r" + str(i + 1) + ' / ' +str(len(cloud)) + '\t\t')
        while (e[0] - last_ts > index_w):
            if (e[0] - last_ts > 1.0):
                print (wrn("\nGap in the events:"), e[0] - last_ts, 'sec.')
            idx.append(i)
            last_ts += index_w

    print ()
    idx.append(cloud.shape[0] - 1)
    return np.array(idx, dtype=np.uint32)


def read_event_file_txt(fname, discretization, sort=False):
    print (okb("Reading the event file as a text file..."))
    cloud = np.loadtxt(fname, dtype=np.float)
    if (sort):
        cloud = cloud[cloud[:,0].argsort()]

    if (cloud.shape[0] == 0):
        print (wrn("Read 0 events from " + fname + "!"))
    else:
        t0 = cloud[0][0]
        if (cloud[0][0] > 1e5):
            cloud[:,0] -= t0
            print (wrn("Adjusting initial timestamp to 0!"))

    print (okg("Read"), cloud.shape[0], okg("events:"), cloud[0][0], "-", cloud[-1][0], "sec.")

    idx = get_index(cloud, discretization)
    return cloud.astype(np.float32), idx


def read_event_file_bag(fname, discretization, event_topic):
    if (not with_rosbag):
        print (wrn("rosbag not found!"))
        return None, None

    print (okb("Reading events from a bag file..."), "topic:", event_topic)
    with rosbag.Bag(fname, 'r') as bag:
        if (event_topic not in bag.get_type_and_topic_info()[1].keys()):
            print (wrn("topic '" + event_topic + "' is not found in bag " + fname))
            print ("Available topics:", bag.get_type_and_topic_info()[1].keys())
            return None, None

    ecount = 0
    msg_cnt = 0
    first_event_ts = None
    with rosbag.Bag(fname, 'r') as bag:
        msg_cnt = bag.get_message_count(topic_filters = [event_topic])

        for i, (topic, msg, t) in enumerate(bag.read_messages(topics = [event_topic])):
            if topic == event_topic:
                if (ecount == 0 and len(msg.events) > 0):
                    first_event_ts = msg.events[0].ts
                ecount += len(msg.events)
                sys.stdout.write("read message " + str(i) + " / " + str(msg_cnt) + "\t\t\r")

        print ("\nFound", ecount, "events")

    cloud = np.zeros((ecount, 4), dtype=np.float32)
    eid = 0

    with rosbag.Bag(fname, 'r') as bag:
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics = [event_topic])):
            if topic == event_topic:
                for e in msg.events:
                    cloud[eid][0] = (e.ts - first_event_ts).to_sec()
                    cloud[eid][1] = e.x
                    cloud[eid][2] = e.y
                    if (e.polarity):
                        cloud[eid][3] = 1
                    else:
                        cloud[eid][3] = 0
                    eid += 1
            if (i % 10 == 0):
                sys.stdout.write("convert to npz " + str(i) + " / " + str(msg_cnt) + "\t\t\r")

    print ()
    cloud = cloud[cloud[:,0].argsort()]

    if (cloud.shape[0] == 0):
        print (wrn("Read 0 events from " + fname + "!"))
    else:
        t0 = cloud[0][0]
        if (cloud[0][0] > 1e5):
            cloud[:,0] -= t0
            print (wrn("Adjusting initial timestamp to 0!"))

    print (okb("Indexing..."))
    idx = get_index(cloud, discretization)
    return cloud.astype(np.float32), idx


def undistort_img(img, K, D):
    if (K is None):
        return img
    if (D is None):
        D = np.array([0, 0, 0, 0])

    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted


def dvs_img(cloud, shape, model=None, scale=None, K=None, D=None):
    fcloud = cloud.astype(np.float32) # Important!

    if (model is None):
        model = [0, 0, 0, 0]

    if (scale is None):
        scale = 1

    cmb = np.zeros((shape[0] * scale, shape[1] * scale, 3), dtype=np.float32)
    cpydvs.dvs_img(fcloud, cmb, model, scale)

    cmb = undistort_img(cmb, K, D)

    cnt_img = cmb[:,:,0] + cmb[:,:,2] + 1e-8
    timg = cmb[:,:,1]

    timg /= cnt_img

    # Undistortion may affect the event counts
    timg[cnt_img < 0.9] = 0
    return cmb


def dvs_err(tc_img, G_img):
    return cpydvs.dvs_error(tc_img, G_img)


def dvs_flow_err(tc_img, G_img):
    return cpydvs.dvs_flow_error(tc_img, G_img)


def get_slice(cloud, idx, ts, width, mode=0, idx_step=0.01):
    if (cloud.shape[0] == 0):
        return cloud, np.array([0])

    ts_lo = ts
    ts_hi = ts + width
    if (mode == 1):
        ts_lo = ts - width / 2.0
        ts_hi = ts + width / 2.0
    if (mode == 2):
        ts_lo = ts - width
        ts_hi = ts
    if (mode > 2 or mode < 0):
        print (wrn("get_slice: Wrong mode! Reverting to default..."))
    if (ts_lo < 0): ts_lo = 0

    t0 = cloud[0][0]

    idx_lo = int((ts_lo - t0) / idx_step)
    idx_hi = int((ts_hi - t0) / idx_step)
    if (idx_lo >= len(idx)): idx_lo = -1
    if (idx_hi >= len(idx)): idx_hi = -1

    sl = np.copy(cloud[idx[idx_lo]:idx[idx_hi]].astype(np.float32))
    idx_ = np.copy(idx[idx_lo:idx_hi])

    if (idx_lo == idx_hi):
        return sl, np.array([0])

    if (len(idx_) > 0):
        idx_0 = idx_[0]
        idx_ -= idx_0

    if (sl.shape[0] > 0):
        t0 = sl[0][0]
        sl[:,0] -= t0

    return sl, idx_
