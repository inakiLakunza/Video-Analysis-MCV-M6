""" Dataset class for HMDB51 dataset. """

import sys
import os
import random
import numpy as np
from enum import Enum

from glob import glob, escape
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.utils import save_image

from PIL import Image



class FiveCropTSNHMDB51Dataset(Dataset):
    """
    Dataset class for HMDB51 dataset.
    """

    class Split(Enum):
        """
        Enum class for dataset splits.
        """
        TEST_ON_SPLIT_1 = 1
        TEST_ON_SPLIT_2 = 2
        TEST_ON_SPLIT_3 = 3

    class Regime(Enum):
        """
        Enum class for dataset regimes.
        """
        TRAINING = 1
        TESTING = 2
        VALIDATION = 3

    CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]


    def __init__(
        self, 
        videos_dir: str, 
        annotations_dir: str, 
        split: Split, 
        regime: Regime, 
        clip_length: int, 
        crop_size: int, 
        temporal_stride: int,
        n_segments: int
    ) -> None:
        """
        Initialize HMDB51 dataset.

        Args:
            videos_dir (str): Directory containing video files.
            annotations_dir (str): Directory containing annotation files.
            split (Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            regime (Regimes): Dataset regime (TRAINING, TESTING, VALIDATION).
            split (Splits): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            clip_length (int): Number of frames of the clips.
            crop_size (int): Size of spatial crops (squares).
            temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        """
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride
        self.n_segments = n_segments

        self.annotation = self._read_annotation()
        self.transform = self._create_transform()


    def _read_annotation(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: Dataframe containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in FiveCropTSNHMDB51Dataset.CLASS_NAMES:
            annotation_file = os.path.join(self.annotations_dir, class_name + split_suffix)
            df = pd.read_csv(annotation_file, sep=" ").dropna(axis=1, how='all') # drop empty columns
            df.columns = ['video_name', 'train_or_test']
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={'video_name': 'video_path'})
            df['video_path'] = os.path.join(self.videos_dir, class_name, '') + df['video_path'].replace('\.avi$', '', regex=True)
            df = df.rename(columns={'train_or_test': 'class_id'})
            df['class_id'] = FiveCropTSNHMDB51Dataset.CLASS_NAMES.index(class_name)
            annotation += [df]

        return pd.concat(annotation, ignore_index=True)


    def _create_transform(self) -> v2.Compose:
        """
        Create transform based on the dataset regime.

        Returns:
            v2.Compose: Transform for the dataset.
        """
        if self.regime == FiveCropTSNHMDB51Dataset.Regime.TRAINING:
            return v2.Compose([
                v2.Resize(self.crop_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            ## WTF evaluar amb VenterCrop?
            return v2.Compose([
                v2.Resize(self.crop_size), # Shortest side of the frame to be resized to the given size
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def get_num_classes(self) -> int:
        """
        Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(FiveCropTSNHMDB51Dataset.CLASS_NAMES)


    def __len__(self) -> int:
        """
        Get the length (number of videos) of the dataset.

        Returns:
            int: Length (number of videos) of the dataset.
        """
        return len(self.annotation)


    def __getitem__(self, idx: int) -> tuple:
        """
        Get item (video) from the dataset.

        Args:
            idx (int): Index of the item (video).

        Returns:
            tuple: Tuple containing video, label, and video path.
        """
        df_idx = self.annotation.iloc[idx]

        # Get video path from the annotation dataframe and check if it exists
        video_path = df_idx['video_path']

        assert os.path.exists(video_path)

        # Read frames' paths from the video
        frame_paths = sorted(glob(os.path.join(escape(video_path), "*.jpg"))) # get sorted frame paths
        video_len = len(frame_paths)
        
        
                
        frame_paths = np.array_split(frame_paths, self.n_segments)
        max_duration = max([len(i) for i in frame_paths])



        # Read frames from the video with the desired temporal subsampling
        video = None
        #for idx in range(clip_begin, clip_end, self.temporal_stride*self.clip_length):
        for idx, chunk in enumerate(frame_paths):
            for ii, path in enumerate(chunk):                  
                frame = read_image(path)  # (C, H, W)
                if video is None:
                    video = torch.zeros((self.n_segments, max_duration, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)

                video[idx][ii] = frame
                
        random_indexes = torch.randint(0, video.size(1), (self.clip_length,))
        
        sampled_video  =  video[:, random_indexes, :, :, :]
        

        # Get label from the annotation dataframe and make sure video was read
        label = df_idx['class_id']
        assert video is not None
        

        return sampled_video, label, video_path

    
    def collate_fn(self, batch: list) -> dict:
        """
        Collate function for creating batches.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """
        # [(clip1, label1, path1), (clip2, label2, path2), ...] 
        #   -> ([clip1, clip2, ...], [label1, label2, ...], [path1, path2, ...])
        unbatched_clips, unbatched_labels,  paths = zip(*batch)
        
        
        # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)  [(S, T, C, H, W)]
        transformed_clips = [self.transform(clip).permute(0, 2, 1, 3, 4) for clip in unbatched_clips]
        
        try_clip = transformed_clips[0]
        print(try_clip.shape)

        try_frame = try_clip[0, :, 0, :, :]
        print(try_frame.shape)
        save_image(try_frame, "try.png")

        five_crops = v2.functional.five_crop(try_frame, self.crop_size)
        for i in range(len(five_crops)):
            crop = five_crops[i]
            save_image(crop, f"try_crop{i}.png")
        
        
        
        final_transfored_clips = []
        print(f'Primero: {transformed_clips[0].shape}') # (4, 3, 4, 182, 315)
        print(f'Ultimo: {transformed_clips[-1].shape}') # (4, 3, 4, 182, 242)
        new_window = []
        for cl in transformed_clips:
            crops = v2.functional.five_crop(cl, self.crop_size)
            print("len crops: ", len(crops))
            concatenated_frames = []
            # LEN CROPS 5 (ES UNA TUPLA CON LOS FIVE CROPS)
            for i in range(len(crops)):
                # SHAPE OF EACH CROP: (4, 3, 4, 182, 182)
                print(i)
                new_window.append(crops[i])
                concatenated_frames.append(crops[i])
                print("shape crops[i]", crops[i].shape)
                # crop = crops[i]
                # print(crop.shape)
                # tensor = crop[0, :, i, :, :]
                # tensor = tensor.permute(1, 2, 0)
                # image = Image.fromarray(tensor)
                # image.save(f'out_{i}.png')
                # save_image(crop[0, :, i, :, :], f'img{i}.png')
            # tt = torch.stack([torch.stack([v2.PILToTensor()(crop) for crop in crops])]).view(1, 5*self.n_segments, 3, self.clip_length, self.crop_size, self.crop_size)
            # final_transfored_clips.append(tt)
            # concatenated_crops = torch.cat(concatenated_frames, 2)
            # new_window.append(concatenated_crops)
        new_window = torch.stack(new_window) # 80, 4, 3, 4, 182, 182
        new_window = new_window.view(16, 4, 4, 5, 3, 182, 182)
        print(new_window.shape)
        sys.exit()

        # Concatenate clips along the batch dimension: 
        # B * [(C, T, H, W)] -> B * [(1, C, T, H, W)] -> (B, C, T, H, W)
        batched_clips = torch.cat([d for d in final_transfored_clips], dim=0)
        batched_clips = batched_clips.view(batched_clips.shape[0]*batched_clips.shape[1], 3, -1, self.crop_size, self.crop_size)

        return dict(
            clips=batched_clips, # (B, C, T, H, W)
            labels=torch.tensor(unbatched_labels), # (K,)
            paths=paths  # no need to make it a tensor
        )