# Originally Forked from https://github.com/ExplorerFreda/VGNSL?files=1
# Modified by Noriyuki Kojima (nk654@cornell.edu)

import nltk
import numpy as np
import os

import torch
import torch.utils.data as data


class PrecompDataset(data.Dataset):
    """ load precomputed captions and image features """

    def __init__(self, data_path, data_split, vocab,
                 load_img=True, img_dim=2048, no_se_tokens=False, image_noise=''):
        self.vocab = vocab
        self.no_se_tokens = no_se_tokens
        self.image_noise = image_noise

        # captions
        self.captions = list()
        with open(os.path.join(data_path, f'{data_split}_caps.txt'), 'r') as f:
            for line in f:
                self.captions.append(line.strip().lower().split())
            f.close()
        self.length = len(self.captions)
        # image features
        if load_img:
            self.images = np.load(os.path.join(
                data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // 5, img_dim))
        # each image can have 1 caption or 5 captions
        if self.images.shape[0] != self.length:
            self.im_div = 5
            # assert self.images.shape[0] * 5 == self.length
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # image
        img_id = index // self.im_div
        if self.image_noise == "shuffle":
            img_id = np.random.randint(len(self.images), size=1)[0]
        image = torch.tensor(self.images[img_id])
        # caption
        if self.no_se_tokens:
            caption = [self.vocab(token) for token in self.captions[index]]
        else:
            caption = [self.vocab(token)
                       for token in ['<start>'] + self.captions[index] + ['<end>']]
        caption = torch.tensor(caption)
        return image, caption, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, ids, img_ids = zipped_data
    images = torch.stack(images, 0)
    targets = torch.zeros(len(captions), len(captions[0])).long()
    lengths = [len(cap) for cap in captions]
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = cap[:end]
    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, batch_size=128,
                       shuffle=True, num_workers=2, load_img=True,
                       img_dim=2048, no_se_tokens=False, image_noise=''):
    dset = PrecompDataset(data_path, data_split, vocab, load_img,
                          img_dim, no_se_tokens=no_se_tokens, image_noise=image_noise)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return data_loader


def get_debug_loaders(data_path, vocab, batch_size, workers, is_shuffle=True, data_mode="global", image_noise=''):
    train_loader = get_precomp_loader(
        data_path, 'debug', vocab, batch_size, False, workers, image_noise=image_noise
    )
    val_loader = get_precomp_loader(
        data_path, 'debug', vocab, batch_size, False, workers, image_noise=image_noise
    )
    return train_loader, val_loader


def get_train_loaders(data_path, vocab, batch_size, workers, is_shuffle=True, data_mode="global", image_noise=''):
    train_loader = get_precomp_loader(
        data_path, 'train', vocab, batch_size, bool(is_shuffle), workers, image_noise=image_noise
    )  # TODO: fix
    val_loader = get_precomp_loader(
        data_path, 'dev', vocab, batch_size, False, workers, image_noise=image_noise
    )
    return train_loader, val_loader


def get_val_test_loaders(data_path, vocab, batch_size, workers, is_shuffle=True, data_mode="global", image_noise=''):
    val_loader = get_precomp_loader(
        data_path, 'dev', vocab, batch_size, False, workers, image_noise=image_noise
    )
    """
    test_loader = get_precomp_loader(
        data_path, 'test', vocab, batch_size, False, workers, image_noise=image_noise
    )
    """
    return val_loader, None


def get_eval_loader(data_path, split_name, vocab, batch_size, workers,
                    load_img=False, img_dim=2048):
    eval_loader = get_precomp_loader(
        data_path, split_name, vocab, batch_size, False, workers,
        load_img=load_img, img_dim=img_dim
    )
    return eval_loader
