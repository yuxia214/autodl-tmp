import os
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from toolkit.utils.read_data import *


class Data_Feat_AV(Dataset):
    """Audio-Video only feature dataset for attention_robust_v10."""

    def __init__(self, args, names, labels):
        self.names = names
        self.labels = labels

        if args.audio_feature is None or args.video_feature is None:
            raise ValueError("audio_feature and video_feature must be provided for AV-only training.")

        feat_root = config.PATH_TO_FEATURES[args.dataset]
        audio_root = os.path.join(feat_root, args.audio_feature)
        video_root = os.path.join(feat_root, args.video_feature)
        print(f'audio feature root: {audio_root}')
        print(f'video feature root: {video_root}')

        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale
        assert self.feat_scale >= 1
        if self.feat_type != 'utt':
            raise ValueError("attention_robust_v10 currently supports feat_type='utt' only.")

        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')

        audios, videos = self.feature_scale_compress_av(audios, videos, self.feat_scale)
        audios, videos = self.align_to_utt_av(audios, videos)
        self.audios, self.videos = audios, videos

    @staticmethod
    def feature_scale_compress_av(audios, videos, scale_factor=1):
        for ii in range(len(audios)):
            audios[ii] = func_mapping_feature(audios[ii], math.ceil(len(audios[ii]) / scale_factor))
            videos[ii] = func_mapping_feature(videos[ii], math.ceil(len(videos[ii]) / scale_factor))
        return audios, videos

    @staticmethod
    def align_to_utt_av(audios, videos):
        for ii in range(len(audios)):
            audios[ii] = np.mean(audios[ii], axis=0)
            videos[ii] = np.mean(videos[ii], axis=0)
        return audios, videos

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        instance = dict(
            audio=self.audios[index],
            video=self.videos[index],
            emo=self.labels[index]['emo'],
            val=self.labels[index]['val'],
            name=self.names[index],
        )
        return instance

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        videos = [instance['video'] for instance in instances]

        batch = dict(
            audios=torch.FloatTensor(np.array(audios)),
            videos=torch.FloatTensor(np.array(videos)),
        )

        emos = torch.LongTensor([instance['emo'] for instance in instances])
        vals = torch.FloatTensor([instance['val'] for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names

    def get_featdim(self):
        print(f'audio dimension: {self.adim}; text dimension: 0; video dimension: {self.vdim}')
        return self.adim, 0, self.vdim
