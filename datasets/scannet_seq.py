import os
import os.path as osp
from glob import glob
import numpy as np
import cv2

SLAM3R_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
import sys 
sys.path.insert(0, SLAM3R_DIR) 
 
from slam3dr.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from slam3dr.utils.image import imread_cv2
from collections import deque


class ScanNet_Seq(BaseStereoViewDataset):
    """
    ScanNet(v2) RGBD sequence loader (for validation / training).
    Expected scene folder layout (typical):
      <ROOT>/<scene_id>/
        sensor_data/frame-000000.color.jpg
        sensor_data/frame-000000.depth.png          # uint16 in mm
        sensor_data/frame-000000.posecd.txt           # 4x4 cam2world
        intrinsic/intrinsic_color.txt  # 4x4 or 3x3 (we handle both)
    """
    def __init__(
        self,
        ROOT="../scannet/",
        scene_id=None,
        num_views=2,
        num_frames=5,
        sample_freq=1,
        start_freq=1, kf_every=1,
        *args, **kwargs
    ):
        self.ROOT = ROOT
        self.scene_id = scene_id
        self.num_fetch_views = num_views
        self.sample_freq = sample_freq
        self.start_freq = start_freq
        self.kf_every = kf_every
        super().__init__(*args, **kwargs)

    def sample_frames(self, img_idxs, rng):
        num_frames = self.num_frames
        thresh = int(self.min_thresh + self.train_ratio * (self.max_thresh - self.min_thresh))
                
        img_indices = list(range(len(img_idxs)))
        
        selected_indices = []
        
        initial_valid_range = max(len(img_indices)//num_frames, len(img_indices) - thresh * (num_frames - 1))
        current_index = rng.choice(img_indices[:initial_valid_range])

        selected_indices.append(current_index)
        
        while len(selected_indices) < num_frames:
            next_min_index = current_index + 1
            next_max_index = min(current_index + thresh, len(img_indices) - (num_frames - len(selected_indices)))
            possible_indices = [i for i in range(next_min_index, next_max_index + 1) if i not in selected_indices]
        
            if not possible_indices:
                break
            
            current_index = rng.choice(possible_indices)
            selected_indices.append(current_index)
        
        if len(selected_indices) < num_frames:
            return self.sample_frames(img_idxs, rng)

        selected_img_ids = [img_idxs[i] for i in selected_indices]
        
        if rng.choice([True, False]):
            selected_img_ids.reverse()
        
        return selected_img_ids
    

    def sample_frame_idx(self, img_idxs, rng, full_video=False):
        if not full_video:
            img_idxs = self.sample_frames(img_idxs, rng)
        else:
            img_idxs = img_idxs[::self.kf_every]
        
        return img_idxs

    def _load_data(self, base_dir=None):
        self.folder = {'train': 'scans', 'val': 'scans', 'test': 'scans_test'}[self.split]
        
        if self.scene_id is None:
            meta_split = osp.join(base_dir, 'splits', f'scannetv2_{self.split}.txt')  # for train and test records
            
            if not osp.exists(meta_split):
                raise FileNotFoundError(f"Split file {meta_split} not found")
            
            with open(meta_split) as f:
                self.scene_list = f.read().splitlines()
                
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")
            
        else:
            if isinstance(self.scene_id, list):
                self.scene_list = self.scene_id
            else:
                self.scene_list = [self.scene_id]
                
            print(f"Scene_id: {self.scene_id}")

        # region
        # assert self.scene_id is not None, "ScanNet_Seq requires scene_id"

        # scene_dir = osp.join(self.ROOT, self.scene_id)
        # color_dir = osp.join(scene_dir, "color")
        # depth_dir = osp.join(scene_dir, "depth")
        # pose_dir = osp.join(scene_dir, "pose")
        # intrinsic_path = osp.join(scene_dir, "intrinsic", "intrinsic_color.txt")

        # self.images = sorted(glob(osp.join(color_dir, "*.jpg")))
        # if len(self.images) == 0:
        #     self.images = sorted(glob(osp.join(color_dir, "*.png")))
        # assert len(self.images) > 0, f"no images found in {color_dir}"

        # # intrinsics
        # Kraw = np.loadtxt(intrinsic_path).astype(np.float32)
        # if Kraw.shape == (4, 4):
        #     K = Kraw[:3, :3]
        # else:
        #     K = Kraw
        # assert K.shape == (3, 3)

        # self.intrinsics = []
        # self.trajectories = []
        # self.depths = []
        # self.sceneids = []
        # for img_path in self.images:
        #     stem = osp.splitext(osp.basename(img_path))[0]
        #     did = int(stem)

        #     depth_path = osp.join(depth_dir, f"{did}.png")
        #     pose_path = osp.join(pose_dir, f"{did}.txt")
        #     if not (osp.exists(depth_path) and osp.exists(pose_path)):
        #         # skip missing frames
        #         continue

        #     c2w = np.loadtxt(pose_path).astype(np.float32)
        #     if not np.isfinite(c2w).all():
        #         continue

        #     self.intrinsics.append(K.copy())
        #     self.trajectories.append(c2w)
        #     self.depths.append(depth_path)
        #     self.sceneids.append(0)

        # self.intrinsics = np.stack(self.intrinsics, axis=0)
        # self.trajectories = np.stack(self.trajectories, axis=0)
        # self.images = [osp.basename(p) for p in self.images[:len(self.depths)]]

        # # build sliding-window pairs
        # self.pairs = []
        # image_num = len(self.depths)
        # for i in range(0, image_num, self.start_freq):
        #     last_id = i + (self.num_fetch_views - 1) * self.sample_freq
        #     if last_id >= image_num:
        #         break
        #     self.pairs.append([i + j * self.sample_freq for j in range(self.num_fetch_views)])

        # self.scene_names = [self.scene_id]
        # endregion

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]

        # Load metadata
        intri_path = osp.join(self.ROOT, self.folder, scene_id, 'intrinsic/intrinsic_depth.txt')
        intrinsics = np.loadtxt(intri_path).astype(np.float32)[:3, :3]

        # Load image data
        data_path = osp.join(self.ROOT, self.folder, scene_id, 'sensor_data')
        num_files = len([name for name in os.listdir(data_path) if 'color' in name])  

        img_idxs_ = [f'{i:06d}' for i in range(num_files)]
        imgs_idxs = self.sample_frame_idx(img_idxs_, rng, full_video=self.full_video)
        imgs_idxs = deque(imgs_idxs)
        views = []


        while len(imgs_idxs) > 0:
            view_idx = imgs_idxs.popleft()
            img_path = osp.join(data_path, f'frame-{view_idx}.color.jpg')
            depth_path = osp.join(data_path, f'frame-{view_idx}.depth.png')
            pose_path = osp.join(data_path, f'frame-{view_idx}.pose.txt')
            # img_name = self.images[view_idx]
            # img_path = osp.join(color_dir, img_name)
            # depth_path = self.depths[view_idx]

            rgb_image = imread_cv2(img_path)
            depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0  
            camera_pose = np.loadtxt(pose_path).astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset="ScanNet",
                label=f"{self.scene_id}_{view_idx}",
                instance=f"{idx}_{view_idx}",
            ))
        return views