

import os
import cv2
import numpy as np
import time
import argparse

from pathlib import Path
import torch
from torch import nn

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def makedir(d):
  if not os.path.exists(d):
    os.makedirs(d)


class Feature():
    def __init__(self,
                 feature_num = 512,
                 ):
        self.feature_num = feature_num
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = SuperPoint(max_num_keypoints=feature_num).eval().cuda()
        self.matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher


    def detect(self, image_path):
        image = load_image(image_path).cuda()
        features = self.detector.extract(image)
        return features

    def match(self, feats0, feats1):
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        matches01 = rbd(matches01)
        return matches01['matches'].cpu()


def find_inliers(kpts1, kpts2):
    if len(kpts1) < 10 or len(kpts2) < 10:
        return None
    F, mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC, 20)
    mask = mask.reshape(-1) if (mask is not None and np.sum(mask) > 100) else None
    return mask

def draw_keypoints(image, kpts):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for pt1 in kpts:
        cv2.circle(result_img, tuple(pt1), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)  # 绿色，实心，半径为3
    return result_img

def draw_matches(last_image, image, last_kpts, kpts, track_ids, save_path):
    drawed_img1 = draw_keypoints(last_image, last_kpts)
    drawed_img2 = draw_keypoints(image, kpts)

    h1, w1, _ = drawed_img1.shape
    h2, w2, _ = drawed_img2.shape
    result_img = np.ones((max(h1, h2), w1 + w2 + 10, 3), dtype=np.uint8) * 255
    result_img[:h1, :w1] = drawed_img1
    result_img[:h2, w1 + 10:] = drawed_img2

    for i, tid1 in enumerate(track_ids):
        if tid1 == -1:
            continue  # 跳过没有track的点

        pt1 = tuple(last_kpts[tid1])
        pt1 = tuple(round(x) for x in pt1)

        pt2 = kpts[i]
        pt2 = tuple([pt2[0] + w1 + 10, pt2[1]])
        pt2 = tuple(round(x) for x in pt2)

        cv2.line(result_img, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)  # 绿色，粗2个像素，50%透明
        overlay = result_img.copy()
        cv2.addWeighted(overlay, 0.5, result_img, 1 - 0.5, 0, result_img)  # 设置透明度为0.5


    cv2.imwrite(save_path, result_img)



def process_sequence(seq_root, feature):
    rgb_root = os.path.join(seq_root, "rgb")
    tracking_root = os.path.join(seq_root, "matching")
    pose_root = os.path.join(seq_root, "poses")
    cali_root = os.path.join(seq_root, "calibration")
    matching_vis_root = os.path.join(seq_root, "matching_vis")

    makedir(tracking_root)
    makedir(matching_vis_root)
    image_names = os.listdir(rgb_root)
    image_names.sort()

    ref_id, num_since_last_ref = None, 0
    last_keypoints, last_track_ids, last_pose, last_image, last_K = None, None, None, None, None
    for i in range(len(image_names)):
        image_name = image_names[i]
        image_path = os.path.join(rgb_root, image_name)
        image = cv2.imread(image_path)

        image_idx = image_name.split('.')[0]
        tracking_file = os.path.join(tracking_root, image_idx + ".matching.txt")

        add_new_keyframe = (i == 0) 
        keypoints = feature.detect(image_path)
        kpts = rbd(keypoints)['keypoints'].cpu().numpy()
        M = len(kpts)
        print("M = {}, image_path = {}".format(M, image_path))
        if not add_new_keyframe:
            matches = feature.match(last_keypoints, keypoints)
            last_kpts = rbd(last_keypoints)['keypoints'].cpu().numpy()

            inliers = find_inliers(last_kpts[matches[:, 0]], kpts[matches[:, 1]])
            if inliers is not None:
                matches = matches[inliers]

                parallax = last_kpts[matches[:, 0]] - kpts[matches[:, 1]]
                avg_parallax = np.mean(np.linalg.norm(parallax, axis=1))
                num_tracked = len(matches)
                
                track_ids = np.full(M, -1)
                track_ids[matches[:, 1]] = matches[:, 0]

                # save_path = os.path.join(matching_vis_root, image_name)
                # draw_matches(last_image, image, last_kpts, kpts, track_ids, save_path)

                add_new_keyframe = (num_tracked < 0.5 * len(last_kpts)) or (avg_parallax > 0.15 * min(image.shape[0], image.shape[1]))
            else:
                add_new_keyframe = True

        if add_new_keyframe:
            
            track_ids = [id for id in range(M)]
            track_ids = np.array(track_ids)

            ref_id = i

            last_keypoints = keypoints
            last_track_ids = track_ids
            last_image = image
            num_since_last_ref = 0

        num_since_last_ref += 1

        meta_data = np.array([i, ref_id, M])
        ids_and_kpts = np.hstack((track_ids[:, np.newaxis], kpts))
        save_info = np.vstack((meta_data[np.newaxis, :], ids_and_kpts))
        np.savetxt(tracking_file, save_info)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Image feature matching for a specific sequence is processed through a specified list of data paths and scenes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('dataroot', type=Path,
                        help='The root catalog of the dataset, e.g. "/home/user/project/datasets"')
    
    parser.add_argument('sequences', type=str, nargs='+',
                        help='A list of sequences to be processed, e.g. "pgt_7scenes_chess pgt_7scenes_heads"')

    args = parser.parse_args()

    # dataroot = "/home/xukuan/project/seq_ace/seq_scr/datasets"
    # sequences = ['pgt_7scenes_chess', 'pgt_7scenes_heads', 'pgt_7scenes_pumpkin', 'pgt_7scenes_fire', 'pgt_7scenes_office', 'pgt_7scenes_redkitchen', 'pgt_7scenes_stairs']
    # sequences = ['Cambridge_GreatCourt', 'Cambridge_KingsCollege', 'Cambridge_OldHospital', 'Cambridge_ShopFacade', 'Cambridge_StMarysChurch']
    # sequences = ['wayspots_bears', 'wayspots_cubes', 'wayspots_inscription', 'wayspots_lawn', 'wayspots_map', 'wayspots_squarebench', 'wayspots_statue', 'wayspots_tendrils', 'wayspots_therock', 'wayspots_wintersign']
    # sequences = ['pgt_12scenes_apt1_kitchen', 'pgt_12scenes_apt1_living', 'pgt_12scenes_apt2_bed', 
    #              'pgt_12scenes_apt2_kitchen', 'pgt_12scenes_apt2_living', 'pgt_12scenes_apt2_luke', 
    #              'pgt_12scenes_office1_gates362', 'pgt_12scenes_office1_gates381', 'pgt_12scenes_office1_lounge', 
    #              'pgt_12scenes_office1_manolis', 'pgt_12scenes_office2_5a', 'pgt_12scenes_office2_5b']


    time0 = time.time()

    train_feature_num = 1000
    train_feature = Feature(train_feature_num)
    for seq in args.sequences:
        print("processing {} ....".format(seq))
        seq_root = args.dataroot / seq
        traing_root = seq_root / "train"
        process_sequence(str(traing_root), train_feature)

    time1 = time.time()
    print("time = {}".format(time1 - time0))