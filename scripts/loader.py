import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data

from torch.autograd import Variable
import torchvision.transforms.functional as FT


log = logging.getLogger(__name__)

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


def augmentation(img, flip, angle, shift):
    img = FT.to_pil_image(img)

    if flip:
        img = FT.hflip()

    img = FT.affine(img, angle=angle, translate=(shift, 0))

    return FT.to_tensor(img).numpy()


def load_volume(path, augment=False):
    vol = np.load(path)

    # crop middle
    pad = int((vol.shape[2] - INPUT_DIM) / 2)
    vol = vol[:, pad:-pad, pad:-pad]

    # standardize
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

    # normalize
    vol = (vol - MEAN) / STDDEV

    if augment:
        # ensure we flip, rotate and shift all images in the volume
        # by the same amount
        flip = random.random() > 0.5
        angle = (random.random() * 50.0) - 25.0
        shift = random.randint(-25, 25)

        vol = np.stack(
            [augmentation(v, flip, angle, shift) for v in vol],
            axis=0
        )

    # convert to RGB
    vol = np.stack((vol,) * 3, axis=1)

    return torch.FloatTensor(vol)


class Dataset(data.Dataset):
    def __init__(self,
                 series,
                 path_df,
                 diagnosis=None,
                 label_df=None,
                 use_gpu=True,
                 augment=False):
        super().__init__()
        self.use_gpu = use_gpu
        self.series = series
        self.augment = augment

        self.paths = {
            idx: row[self.series] for idx, row in path_df.iterrows()
        }
        self.labels = None

        if label_df is not None:
            # we only need to keep paths for the relevant labels
            self.paths = {idx: self.paths[idx] for idx in label_df.index}

        if diagnosis is not None:
            self.diagnosis = diagnosis
            self.labels = {
                idx: row[self.diagnosis] for idx, row in label_df.iterrows()
            }

            neg_weight = np.mean(list(self.labels.values()))
            self.weights = [neg_weight, 1 - neg_weight]

        self.cases = sorted(list(self.paths.keys()))

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(
            prediction, target, weight=Variable(weights_tensor)
        )
        return loss

    def __getitem__(self, index):
        case = self.cases[index]
        vol_tensor = load_volume(self.paths[case], self.augment)

        label_tensor = (
            None if self.labels is None else
            torch.FloatTensor([self.labels[case]])
        )

        return vol_tensor, label_tensor, case

    def __len__(self):
        return len(self.cases)


def paths_to_df(paths):
    df = pd.DataFrame()

    for path in paths:
        series, case = path.split('/')[-2:]
        df.loc[case, series] = path

    assert(np.all(pd.notnull(df).values))
    return df


def load_data(paths,
              series,
              label_df=None,
              diagnosis=None,
              use_gpu=False,
              is_full=False,
              augment=False):

    path_df = paths_to_df(paths)

    # For inference
    if label_df is None:
        dataset = Dataset(
            series=series,
            path_df=path_df,
            diagnosis=diagnosis,
            label_df=label_df,
            use_gpu=use_gpu
        )
        loader = data.DataLoader(
            dataset, batch_size=1, num_workers=1, shuffle=False
        )
        return loader, None, None

    if is_full:
        train_df = label_df[label_df.split in ['train', 'valid']]
        valid_df = label_df[label_df.split == 'test']
        test_df = None
    else:
        train_df = label_df[label_df.split == 'train']
        valid_df = label_df[label_df.split == 'valid']
        test_df = label_df[label_df.split == 'test']

    train_dataset = Dataset(
        series=series,
        path_df=path_df,
        diagnosis=diagnosis,
        label_df=train_df,
        use_gpu=use_gpu,
        augment=augment
    )
    log.debug('Train dataset had {} instances'.format(len(train_dataset)))
    valid_dataset = Dataset(
        series=series,
        path_df=path_df,
        diagnosis=diagnosis,
        label_df=valid_df,
        use_gpu=use_gpu
    )
    log.debug('Valid dataset had {} instances'.format(len(valid_dataset)))

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,  # 1 if use_gpu else 0,
        shuffle=True
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=0,  # 1 if use_gpu else 0,
        shuffle=False
    )

    if test_df is not None:
        test_dataset = Dataset(
            series=series,
            path_df=path_df,
            diagnosis=diagnosis,
            label_df=test_df,
            use_gpu=use_gpu
        )
        log.debug('Test dataset had {} instances'.format(len(test_dataset)))
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=0,  # 1 if use_gpu else 0,
            shuffle=False
        )
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader
