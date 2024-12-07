import os
import cv2
import numpy as np
import torch
import time
from timeit import default_timer as timer
from scipy.spatial.transform import Rotation as R

import dsacstar


def pose_to_vector(pose):
    r, t = R.from_matrix(pose[:3, :3]), pose[:3, 3]
    rv = r.as_rotvec().reshape(-1)
    return np.hstack((rv, t))

def pose_to_inverse_vertor(pose):
    R12, t12 = pose[:3, :3], pose[:3, 3]
    R21 = R12.T
    t21 = -R21 @ t12

    r, t = R.from_matrix(R21), t21
    rv = r.as_rotvec().reshape(-1)
    return np.hstack((rv, t))

def vector_to_pose(v):
    rv, t = v[:3], v[3:]
    r = R.from_rotvec(rv)
    T = np.eye(4)  
    T[:3, :3] = r.as_matrix()  
    T[:3, 3] = t  
    return T 


def weight_average(pred_pose, pred_inlier, obv_pose, obv_inlier):
    pred_is_good = (pred_inlier is not None and pred_inlier >= 20)
    obv_si_good = (obv_inlier is not None and obv_inlier >= 20)

    if pred_is_good and obv_si_good:
        pred_v = pose_to_vector(pred_pose)
        obv_v = pose_to_vector(obv_pose)

        rate = pred_inlier / (pred_inlier + obv_inlier)
        post_v = rate * pred_v + (1 - rate) * obv_v
        post_pose = vector_to_pose(post_v)

        return post_pose, True
    elif obv_si_good:
        return obv_pose, True
    elif pred_is_good:
        return pred_pose, True
    else:
        return None, False


class PosePredictor():
    def __init__(self): 
        self.last_image = None
        self.last_pose = None

        self.candidates_2d = None
        self.candidates_3d = None

        self.points_2d = None
        self.points_3d = None
        self.weights = None

        self.image_shape = None
        self.max_points = 500
        self.repro_error_thr = 4
        self.nms_radius = 5

        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.init = False
        self.lost_num = 0

        self.debug_idx = 0


    def generate_new_points(self, candidates_2d, candidates_3d, original_2d, original_3d):
        candidate_is_good = candidates_2d is not None and candidates_3d is not None and len(candidates_2d) > 0
        original_is_good = original_2d is not None and original_3d is not None and len(original_2d) > 0

        if not candidate_is_good and not original_is_good:
            return None, None

        candidates_2d = candidates_2d if candidate_is_good else np.zeros((0, 2), dtype=np.float32)
        original_2d = original_2d if original_is_good else np.zeros((0, 2), dtype=np.float32)
        Nc, No = len(candidates_2d), len(original_2d)
        keep_flag = torch.zeros((Nc+No), dtype=torch.int)

        keep_num = dsacstar.update_points(torch.from_numpy(candidates_2d), torch.from_numpy(original_2d), 
                keep_flag, self.image_shape[1], self.image_shape[0], float(self.nms_radius), self.max_points)

        return keep_flag[:No].numpy(), keep_flag[No:].numpy()


        new_2d, new_3d = [], []
        keep_flag = (keep_flag.numpy() > 0)
        if original_is_good:
            new_2d.append(original_2d[keep_flag[:No]])
            new_3d.append(original_3d[keep_flag[:No]])

        if candidate_is_good:
            new_2d.append(candidates_2d[keep_flag[No:]])
            new_3d.append(candidates_3d[keep_flag[No:]])  

        return np.vstack(new_2d), np.vstack(new_3d)



    def update(self, image, pose, predict_points_2d, track_status, new_points_2d, new_points_3d, K):
        observation_is_good = new_points_2d is not None and new_points_3d is not None and (len(new_points_2d) > 20)
        prediction_is_good = self.points_3d is not None and predict_points_2d is not None and (np.sum(track_status) > 20)

        if not prediction_is_good and not observation_is_good:
            return 

        pose_v = pose_to_inverse_vertor(pose)

        good_new_points_2d, good_new_points_3d = None, None
        if observation_is_good:
            new_proj, _ = cv2.projectPoints(new_points_3d, pose_v[:3], pose_v[3:], K, None)
            new_repro_errors = np.linalg.norm((new_points_2d - new_proj.reshape(-1, 2)), axis=1)
            inliers = new_repro_errors < self.repro_error_thr
            good_new_points_2d, good_new_points_3d, new_repro_errors = new_points_2d[inliers], new_points_3d[inliers], new_repro_errors[inliers]


        if not self.init:
            if observation_is_good:
                self.last_image = image
                self.last_pose = pose

                self.points_2d = good_new_points_2d
                self.points_3d = good_new_points_3d
                self.weights = np.ones_like(new_repro_errors)

                self.image_shape = image.shape
                self.init = True
            return 


        good_tracked_points_2d, good_tracked_points_3d = None, None
        if prediction_is_good:
            tracked_points_2d, tracked_points_3d = predict_points_2d[track_status], self.points_3d[track_status]
            tracked_proj, _ = cv2.projectPoints(tracked_points_3d, pose_v[:3], pose_v[3:], K, None)
            tracked_repro_errors = np.linalg.norm((tracked_points_2d - tracked_proj.reshape(-1, 2)), axis=1)
            inliers = tracked_repro_errors < self.repro_error_thr
            good_tracked_points_2d, good_tracked_points_3d, tracked_repro_errors = \
                    tracked_points_2d[inliers], tracked_points_3d[inliers], tracked_repro_errors[inliers]
            tracked_weight = self.weights[track_status][inliers]


        tracked_keep, new_keep = self.generate_new_points(good_new_points_2d, good_new_points_3d, good_tracked_points_2d, good_tracked_points_3d)


        new_2d, new_3d, new_weight = [], [], []
        # update 3d points
        if tracked_keep is not None and np.sum(tracked_keep) > 0:
            tracked_keep = tracked_keep > 0
            good_tracked_points_3d, good_tracked_points_2d, tracked_repro_errors, last_weight = \
                    good_tracked_points_3d[tracked_keep], good_tracked_points_2d[tracked_keep], tracked_repro_errors[tracked_keep], tracked_weight[tracked_keep]

            Rwc, twc = pose[:3, :3], pose[:3, 3]
            RK_inv = Rwc @ np.linalg.inv(K)
            X0C = good_tracked_points_3d - twc.reshape(1, 3)

            X0M = torch.from_numpy(X0C).float()
            _ = dsacstar.compute_perpendicular(torch.from_numpy(good_tracked_points_2d).float(),
                    torch.from_numpy(X0C).float(), torch.from_numpy(RK_inv).float(), X0M)
            X0M = X0M.numpy()

            updated_weight = last_weight + tracked_repro_errors
            k = 0.01 * tracked_repro_errors / updated_weight

            X1 = good_tracked_points_3d + k.reshape(-1, 1) * X0M
            
            new_2d.append(good_tracked_points_2d)
            new_3d.append(X1)
            new_weight.append(updated_weight)

        if new_keep is not None and np.sum(new_keep) > 0:
            new_keep = new_keep > 0

            new_2d.append(good_new_points_2d[new_keep])
            new_3d.append(good_new_points_3d[new_keep])
            new_weight.append(new_repro_errors[new_keep])


        self.points_2d, self.points_3d, self.weights = (None, None, None) if len(new_2d) < 1 else (np.vstack(new_2d), np.vstack(new_3d), np.concatenate(new_weight))

        self.last_image = image
        self.last_pose = pose

    def predict(self, image, K, opt):
        if self.points_2d is None or len(self.points_2d) < 20:
            return None, None, None, None


        cur_points, st, err = cv2.calcOpticalFlowPyrLK(self.last_image, image, self.points_2d, None, **self.lk_params)
        valid_flag = (cur_points[:, 0] > 0) & (cur_points[:, 0] < image.shape[1]-1) & (cur_points[:, 1] > 0) & (cur_points[:, 1] < image.shape[0]-1)

        good_idx = st == 1
        good_idx = good_idx[:, 0] & valid_flag

        if(np.sum(good_idx) < 20):
            return None, None, None, None

        # compute relative rotation and translation
        good_last = self.points_2d[good_idx]
        good_cur = cur_points[good_idx]
        good_last_3d = self.points_3d[good_idx]
        track_status = good_idx

        F, mask = cv2.findFundamentalMat(good_last, good_cur, cv2.FM_RANSAC)
        mask = mask[:, 0] > 0
        good_last = good_last[mask]
        good_cur = good_cur[mask]
        good_last_3d = good_last_3d[mask]
        track_status[good_idx] *= mask


        if(len(good_cur) < 20):
            return None, None, None, None


        inliers = torch.zeros(len(good_cur), dtype=torch.int)
        focal_length, ppX, ppY = K[0, 0], K[0, 2], K[1, 2]
        out_pose = torch.zeros((4, 4))

        inlier_count = dsacstar.forward_sequence_rgb(
            torch.from_numpy(good_last_3d).float(),
            torch.from_numpy(good_cur).float(),
            inliers,
            out_pose,
            4,
            focal_length,
            ppX,
            ppY,
            opt.maxpixelerror)

        if inlier_count < 20:
            return None, None, None, None

        return out_pose.numpy(), inlier_count, cur_points, track_status
        


