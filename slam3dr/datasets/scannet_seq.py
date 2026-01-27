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
        num_seq=100, num_views=5, 
        min_thresh=1, max_thresh=100, 
        test_id=None, full_video=False, kf_every=1,
        ROOT="../scannet/",
        *args, **kwargs
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq # number of sequences per scene
        self.num_views = num_views
        if num_views < 0:
            full_video = True
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self._load_data(base_dir=self.ROOT)

    def sample_frames(self, img_idxs, rng):
        num_frames = self.num_views
        thresh = int(self.min_thresh + 0.5 * (self.max_thresh - self.min_thresh))
                
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
        self.folder = {'train': 'scans', 'val': 'scans', 'test': 'scans'}[self.split]  # scans_test
        
        if self.test_id is None:
            meta_split = osp.join(base_dir, 'data_splits', f'scannetv2_{self.split}.txt')  # for train and test records
            
            if not osp.exists(meta_split):
                raise FileNotFoundError(f"Split file {meta_split} not found")
            
            with open(meta_split) as f:
                self.scene_list = f.read().splitlines()
                
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")
            
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
                
            print(f"Scene_id: {self.test_id}")

    
    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def _get_views(self, idx, resolution, rng, attempts=0):
        scene_id = self.scene_list[idx // self.num_seq]

        # Load metadata
        intri_path = osp.join(self.ROOT, self.folder, scene_id, 'intrinsic/intrinsic_depth.txt')
        intri = np.loadtxt(intri_path).astype(np.float32)[:3, :3]

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

            rgb_image = imread_cv2(img_path)
            depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0  
            camera_pose = np.loadtxt(pose_path).astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=view_idx
            )
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {img_path}")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__()-1)
                        return self._get_views(new_idx, resolution, rng)
                    return self._get_views(idx, resolution, rng, attempts+1)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset="ScanNet",
                label=f"{scene_id}_{view_idx}",
                instance=f"{idx}_{view_idx}",
            ))
        return views