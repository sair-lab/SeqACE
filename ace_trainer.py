# Copyright © Niantic, Inc. 2022.

import logging
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network import Regressor
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

import cv2
import os

_logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TrainerACE:
    def __init__(self, options):
        self.options = options

        self.device = torch.device('cuda:0')

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = False

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Create dataset.
        self.dataset = CamLocDataset(
            root_dir=self.options.scene / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            num_clusters=self.options.num_clusters,  # Optional clustering for Cambridge experiments.
            cluster_idx=self.options.cluster_idx,    # Optional clustering for Cambridge experiments.
            load_kpts=True,
        )

        _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self.options.scene,
            len(self.dataset),
            self.dataset.mean_cam_center[0],
            self.dataset.mean_cam_center[1],
            self.dataset.mean_cam_center[2]))

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous
        )
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)

        # Setup learning rate scheduler.
        steps_per_epoch = self.options.training_buffer_size // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       epochs=self.options.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = self.options.epochs * self.options.training_buffer_size // self.options.batch_size
        self.iterations_output = 100 # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )

        self.repro_cross_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )


        # Will be filled at the beginning of the training process.
        self.training_buffer = None

        # Generate video of training process
        if self.options.render_visualization:
            # infer rendering folder from map file name
            target_path = vutil.get_rendering_target_path(
                self.options.render_target_path,
                self.options.output_map_file)
            self.ace_visualizer = ACEVisualizer(
                target_path,
                self.options.render_flipped_portrait,
                self.options.render_map_depth_filter,
                mapping_vis_error_threshold=self.options.render_map_error_threshold)
        else:
            self.ace_visualizer = None

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        if self.ace_visualizer is not None:

            # Setup the ACE render pipeline.
            self.ace_visualizer.setup_mapping_visualisation(
                self.dataset.pose_files,
                self.dataset.rgb_files,
                self.iterations // self.iterations_output + 1,
                self.options.render_camera_z_offset
            )

        creating_buffer_time = 0.
        training_time = 0.

        self.training_start = time.time()

        # Create training buffer.
        buffer_start_time = time.time()
        self.create_training_buffer()
        buffer_end_time = time.time()
        creating_buffer_time += buffer_end_time - buffer_start_time
        _logger.info(f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s.")

        # Train the regression head.
        for self.epoch in range(self.options.epochs):
            epoch_start_time = time.time()
            self.run_epoch()
            training_time += time.time() - epoch_start_time
            
        # Save trained model.
        self.save_model()

        end_time = time.time()
        _logger.info(f'Done without errors. '
                     f'Creating buffer time: {creating_buffer_time:.1f} seconds. '
                     f'Training time: {training_time:.1f} seconds. '
                     f'Total time: {end_time - self.training_start:.1f} seconds.')

        if self.ace_visualizer is not None:

            # Finalize the rendering by animating the fully trained map.
            vis_dataset = CamLocDataset(
                root_dir=self.options.scene / "train",
                mode=0,
                use_half=self.options.use_half,
                image_height=self.options.image_resolution,
                augment=False) # No data augmentation when visualizing the map

            vis_dataset_loader = torch.utils.data.DataLoader(
                vis_dataset,
                shuffle=False, # Process data in order for a growing effect later when rendering
                num_workers=self.num_data_loader_workers)

            self.ace_visualizer.finalize_mapping(self.regressor, vis_dataset_loader)

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=1,
                                             drop_last=False)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(dataset=self.dataset,
                                         shuffle=False,
                                         batch_size=1,
                                        #  sampler=batch_sampler,
                                        #  batch_size=None,
                                        #  worker_init_fn=seed_worker,
                                        #  generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            'features': torch.empty((self.options.training_buffer_size, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device),
            'target_px': torch.empty((self.options.training_buffer_size, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((self.options.training_buffer_size, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                          device=self.device),
            'target_px2': torch.empty((self.options.training_buffer_size, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv2': torch.empty((self.options.training_buffer_size, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics2': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv2': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                          device=self.device),
            'track_flag': torch.empty((self.options.training_buffer_size, 1), dtype=torch.bool,
                                          device=self.device),
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            last_pixel_positions_N2, last_gt_pose_inv, last_intrinsics, last_intrinsics_inv, last_ref_id, last_index_map = None, None, None, None, None, None
            while buffer_idx < self.options.training_buffer_size:
                dataset_passes += 1
                for image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, file_path, _, track_info, keypoint_mask in training_dataloader:

                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        features_BCHW = self.regressor.get_features(image_B1HW)

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features_BCHW.shape

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask_B1HW = image_mask_B1HW.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        continue

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                    features_NC = normalize_shape(features_BCHW)
                    pixel_positions_N2 = normalize_shape(pixel_positions_B2HW)

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    gt_pose_inv = gt_pose_inv_B44[:, :3]
                    gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)


                    batch_data = {
                        'features': features_NC,
                        'target_px': pixel_positions_N2,
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv,
                        'target_px2': pixel_positions_N2,
                        'gt_poses_inv2': gt_pose_inv,
                        'intrinsics2': intrinsics,
                        'intrinsics_inv2': intrinsics_inv,
                        'track_flag': torch.zeros((features_NC.shape[0], 1), dtype=torch.bool).to(self.device, non_blocking=True)
                    }

                    # ####################### select keypoints instead of sampling ############
                    track_info = track_info[0]
                    keypoint_mask = keypoint_mask[0]
                    keypoints = track_info[1:, 1:]
                    track_ids = track_info[1:, 0]

                    _, _, iH, iW = image_B1HW.shape
                    scale_x, scale_y = float(W) / iW, float(H) / iH

                    id, ref_id, Np = track_info[0].numpy().tolist()
                    keypoints = keypoints[keypoint_mask[1:]] - self.regressor.OUTPUT_SUBSAMPLE / 2
                    
      
                    keypoints[:, 0] *= scale_x  
                    keypoints[:, 1] *= scale_y  
                    keypoints = torch.round(keypoints)
                    keypoints[:, 0] = torch.clamp(keypoints[:, 0], min=0, max=W-1)
                    keypoints[:, 1] = torch.clamp(keypoints[:, 1], min=0, max=H-1)

                    x = keypoints[:, 0]  
                    y = keypoints[:, 1] 
                    index_tensor = y * W + x
                    index_tensor = index_tensor.view(-1).int().to(features_BCHW.device)
                    # ####################### select keypoints end ############################


                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)

                    # Over-sample according to image mask.
                    features_to_select = self.options.samples_per_image * B

                    ################################################
                    keypoint_num = len(index_tensor)
                    left_buffer_size = self.options.training_buffer_size - buffer_idx
                    if left_buffer_size <= keypoint_num:
                        sample_idxs = index_tensor[:left_buffer_size, ...]
                        features_to_select = left_buffer_size
                    else:
                        image_mask_N1[index_tensor] = 0
                        feature_to_add = min(left_buffer_size - keypoint_num, features_to_select - keypoint_num)
                        sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                        feature_to_add,
                                                        replacement=True,
                                                        generator=self.sampling_generator)
                        sample_idxs = torch.cat((index_tensor, sample_idxs))
                        features_to_select = feature_to_add + keypoint_num

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    if id == ref_id: 
                        # keyframe
                        last_pixel_positions_N2 = batch_data['target_px']
                        last_gt_pose_inv = batch_data['gt_poses_inv']
                        last_intrinsics = batch_data['intrinsics']
                        last_intrinsics_inv = batch_data['intrinsics_inv']
                        last_ref_id = ref_id

                        index_map = torch.full((len(keypoint_mask[1:]),), -1, dtype=torch.int)
                        valid_indices = torch.nonzero(keypoint_mask[1:], as_tuple=True)[0]
                        index_map[valid_indices] = torch.arange(torch.sum(keypoint_mask[1:]).item(), dtype=torch.int)
                        last_index_map = index_map
                    elif last_ref_id is not None and ref_id == last_ref_id and left_buffer_size > keypoint_num:
                        # frame
                        valid_track_ids = track_ids[keypoint_mask[1:]]
                        valid_track_ids = torch.round(valid_track_ids).int()
                        good_idx = torch.arange(len(valid_track_ids))

                        valid_track_flag = valid_track_ids >= 0
                        valid_track_ids = valid_track_ids[valid_track_flag]
                        good_idx = good_idx[valid_track_flag]

                        new_index = last_index_map[valid_track_ids]
                        valid_new_flag = new_index >= 0
                        valid_track_ids = new_index[valid_new_flag]
                        good_idx = good_idx[valid_new_flag]

                        batch_data['target_px2'][good_idx] = last_pixel_positions_N2[valid_track_ids]
                        batch_data['gt_poses_inv2'][good_idx] = last_gt_pose_inv[valid_track_ids]
                        batch_data['intrinsics2'][good_idx] = last_intrinsics[valid_track_ids]
                        batch_data['intrinsics_inv2'][good_idx] = last_intrinsics_inv[valid_track_ids]
                        batch_data['track_flag'][good_idx] = True
                    else:
                        # segmented sequence by clusting
                        pass


                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                    buffer_idx = buffer_offset
                    if buffer_idx >= self.options.training_buffer_size:
                        break

        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()

    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True


        # Shuffle indices.
        random_indices = torch.randperm(self.options.training_buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, self.options.training_buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > self.options.training_buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer['features'][random_batch_indices].contiguous(),
                self.training_buffer['target_px'][random_batch_indices].contiguous(),
                self.training_buffer['gt_poses_inv'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics_inv'][random_batch_indices].contiguous(),
                self.training_buffer['target_px2'][random_batch_indices].contiguous(),
                self.training_buffer['gt_poses_inv2'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics2'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics_inv2'][random_batch_indices].contiguous(),
                self.training_buffer['track_flag'][random_batch_indices].contiguous()
            )
            self.iteration += 1


    def compute_reprojection_error_loss(self, pred_scene_coords_b41, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, cross=False):
      # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max


        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        if cross:
            loss_valid = self.repro_cross_loss.compute(valid_reprojection_error_b1, self.iteration) 
        else:
            loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration) 

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        return loss, valid_mask_b1, reprojection_error_b1


    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, target_px_b2_2, gt_inv_poses_b34_2, Ks_b33_2, invKs_b33_2, track_flag):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        loss1, valid_mask_b1, reprojection_error_b1 = self.compute_reprojection_error_loss(pred_scene_coords_b41, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33)

        track_flag = track_flag.view(-1)
        loss_track, valid_mask_track, reprojection_error_b1_track = self.compute_reprojection_error_loss(pred_scene_coords_b41[track_flag], target_px_b2_2[track_flag], gt_inv_poses_b34_2[track_flag], 
                                                                Ks_b33_2[track_flag], invKs_b33_2[track_flag], True)
        avg_loss_track = loss_track / torch.sum(track_flag)
        loss = loss1 / batch_size + 0.5 * loss_track / torch.sum(track_flag)


        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # median_depth = float(pred_cam_coords_b31[:, 2].median())

            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                         f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s')

            track_valid = float(valid_mask_track.sum() / torch.sum(track_flag))
            _logger.info(f'Valid: {track_valid * 100:.1f}%, loss1: {loss1:.1f}, avg_loss1: {loss1/batch_size:.2f}, '
                         f'loss_track: {loss_track:.1f}, avg_loss_track: {avg_loss_track:.2f}')

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def save_model(self):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.state_dict()
        for k, v in head_state_dict.items():
            head_state_dict[k] = head_state_dict[k].half()
        torch.save(head_state_dict, self.options.output_map_file)
        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")