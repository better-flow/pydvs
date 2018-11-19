#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, signal, math, time
import matplotlib.colors as colors

import pydvs, cv2


def colorize_image(flow_x, flow_y):
    hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1], 3))
    hsv_buffer[:,:,1] = 1.0
    hsv_buffer[:,:,0] = (np.arctan2(flow_y, flow_x) + np.pi)/(2.0*np.pi)
    hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x,flow_y), axis=0), axis=0 )
    hsv_buffer[:,:,2] = np.log(1. + hsv_buffer[:,:,2])

    flat = hsv_buffer[:,:,2].reshape((-1))
    m = 1
    try:
        m = np.nanmax(flat[np.isfinite(flat)])
    except:
        m = 1
    if not np.isclose(m, 0.0):
        hsv_buffer[:,:,2] /= m

    return colors.hsv_to_rgb(hsv_buffer)


class AlignmentErrorTool:
    def __init__(self, cloud, shape, K, D):
        self.cloud  = np.copy(cloud).astype(np.float32)
        if (self.cloud.shape[0] > 0):
            t0 = self.cloud[0][0]
            self.cloud[:,0] -= t0

        self.scale = 3

        self.K = K
        self.D = D

        self.shape = shape

        self.width = 0
        if (self.cloud.shape[0] > 0):
            self.width = self.cloud[-1][0] - self.cloud[0][0]

        cv2.namedWindow('GUI')
        cv2.createTrackbar('X',  'GUI', 127, 255, self.manual_update)
        cv2.createTrackbar('Y',  'GUI', 127, 255, self.manual_update)
        cv2.createTrackbar('Z',  'GUI', 127, 255, self.manual_update)
        cv2.createTrackbar('Yaw','GUI', 127, 255, self.manual_update)

        circle = np.zeros((self.shape[0] * self.scale, self.shape[1] * self.scale, 2))
        for x in range(circle.shape[0]):
            for y in range(circle.shape[1]):
                circle[x, y] = (np.array([circle.shape[0]/2, circle.shape[1]/2]) - np.array([x, y])) / 100

        self.hsv_circle = colorize_image(circle[:,:,0], circle[:,:,1])

        self.x_err   = 0 
        self.y_err   = 0
        self.z_err   = 0 
        self.yaw_err = 0 
        self.e_count = 0
        self.nz_avg  = 0

        self.manual_update(None)
        while (True):
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k == 99:
                self.minimize_timg()

        cv2.destroyAllWindows()

    def iteration_step(self):

        # Compute images according to the model
        dvs_img = pydvs.dvs_img(self.cloud, self.shape, model=[self.x, self.y, self.z, self.yaw], 
                                scale=self.scale, K=self.K, D=self.D)

        # Compute errors on the images
        dgrad = np.zeros((self.shape[0] * self.scale, self.shape[1] * self.scale, 2), dtype=np.float32)
        self.x_err, self.y_err, self.yaw_err, self.z_err, self.e_count, self.nz_avg = \
            pydvs.dvs_err(dvs_img, dgrad)

        print ("-------------")
        print (self.x, self.y, self.yaw, self.z)
        print (self.x_err, self.y_err, self.yaw_err, self.z_err, self.e_count, self.nz_avg)

        # Visualization
        c_img = dvs_img[:,:,0] + dvs_img[:,:,2]
        c_img = np.dstack((c_img, c_img, c_img)) * 0.5 / (self.nz_avg + 1e-3)

        dvs_img[:,:,1] *= 1.0 / self.width
        t_img = np.dstack((dvs_img[:,:,1], dvs_img[:,:,1], dvs_img[:,:,1]))

        G_img = colorize_image(dgrad[:,:,0], dgrad[:,:,1])
        cv2.imshow('GUI', np.hstack((c_img, t_img, G_img, self.hsv_circle)))

    def manual_update(self, x):
        self.x   = float(cv2.getTrackbarPos('X',  'GUI') - 127) * 10
        self.y   = float(cv2.getTrackbarPos('Y',  'GUI') - 127) * 10
        self.z   = float(cv2.getTrackbarPos('Z',  'GUI') - 127) / 10
        self.yaw = float(cv2.getTrackbarPos('Yaw','GUI') - 127) / 10

        self.iteration_step()

    def minimize_timg(self):
        # Initial errors
        self.iteration_step()
        divs = [0.001, 0.001, 10.0, 10.0]
        
        old_x_err = self.x_err
        old_y_err = self.y_err
        old_yaw_err = self.yaw_err
        old_z_err = self.z_err
        
        enabled = False
        
        while (True):
            if (self.x_err * old_x_err < 0):
                divs[0] *= 2
            if (self.y_err * old_y_err < 0):
                divs[1] *= 2
            if (self.yaw_err * old_yaw_err < 0):
                divs[2] *= 2
            if (self.z_err * old_z_err < 0):
                divs[3] *= 2

            ex_step = self.x_err / divs[0]
            ey_step = self.y_err / divs[1]
            eyaw_step = self.yaw_err / divs[2]
            ez_step = self.z_err / divs[3]

            if (abs(ex_step) < 1e-3 and abs(ey_step) < 1e-3 and 
                abs(eyaw_step) < 1e-3 and abs(ez_step) < 1e-3):
                break
           
            if (abs(self.x_err) < 1e-2 and abs(self.y_err) < 1e-2 and 
                abs(self.yaw_err) < 1e-2 and abs(self.z_err) < 1e-2):
                break

            self.x += ex_step
            self.y += ey_step
 
            if (abs(self.x_err) < 1e-2 and abs(self.y_err)):
                enabled = True

            if (enabled):
                self.yaw += eyaw_step
                self.z += ez_step

            old_x_err = self.x_err
            old_y_err = self.y_err
            old_yaw_err = self.yaw_err
            old_z_err = self.z_err

            self.iteration_step()

            #cv2.waitKey(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice',
                        type=str,
                        required=True)
    parser.add_argument('--bounds',
                        nargs='+',
                        type=int,
                        default=[0, -1],
                        required=False)
    parser.add_argument('--info', 
                        action='store_true', 
                        required=False)

    args = parser.parse_args()

    print ("Opening", args.slice)

    sl_npz = np.load(args.slice)
    cloud = sl_npz['events']
    idx   = sl_npz['index']
    K     = None #sl_npz['K']
    D     = None #sl_npz['D'] / 10

    if (len(args.bounds) != 2 or (args.bounds[0] > args.bounds[1] and args.bounds[1] != -1) 
        or (args.bounds[0] < 0) or (args.bounds[1] < -1)):
        print ("Invalid bounds: ", args.bounds)
        print ("Bounds have to specify two points in the index array, possible values are 0 -", len(idx) - 1)
        exit(0)

    idx = np.append(idx, [cloud.shape[0]])
    sl = cloud[idx[args.bounds[0]]:idx[args.bounds[1]]]

    if (args.info):
        width = cloud[-1][0] - cloud[0][0]
        print ("Input cloud:")
        print ("\tWidth: ", width, "seconds and", len(cloud), "events.")
        print ("\tIndex size: ", len(idx), "points, step = ", width / float(len(idx) + 1), "seconds.")
        print ("")
        width = sl[-1][0] - sl[0][0]
        print ("Chosen slice:")
        print ("\tWidth: ", width, "seconds and", len(sl), "events.")
        print ("")


    a = AlignmentErrorTool(sl, (180, 240), K, D)
