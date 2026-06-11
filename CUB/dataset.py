"""
General utils for training, evaluation and data loading
"""
import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from CUB.config import BASE_DIR, N_ATTRIBUTES
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
                img_path = img_path.replace('images/', '')
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1

                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples

def load_data(
    pkl_paths, 
    use_attr, 
    no_img, 
    batch_size, 
    uncertain_label=False, 
    n_class_attr=2, 
    image_dir='images', 
    resampling=False, 
    resol=299
):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """

    resized_resol = int(resol * 256/224)
    is_training = any(['train.pkl' in f for f in pkl_paths])
    
    if is_training:
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])
    else:
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), # implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])

    dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform)

    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling:
        sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    
    return loader

def find_class_imbalance(
    pkl_file, 
    multiple_attr=False, 
    attr_idx=-1
):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """

    # ======================================

    # 计算存储在 pkl_file 中的二元属性标签的类别不平衡比率
    # 如果 attr_idx >= 0，则仅返回对应属性 id 的比率
    # 如果 multiple_attr 为 True，则分别返回每个属性的不平衡比率；否则，计算所有属性的整体不平衡情况

    # ======================================

    imbalance_ratio = []

    # 加载 train.pkl
    # CUB 实测 data 是 4796 条样本
    # 每条样本的 attribute_label 是 112 维 0/1 向量
    data = pickle.load(open(os.path.join(BASE_DIR, pkl_file), 'rb'))

    # 4796 条 数据
    n = len(data)
    
    # 每条数据的 attribute_label 向量的维数 ( 112 维)
    # 即，一共有 112 个 概念
    # 即，概念瓶颈的维度数为 112
    n_attr = len(data[0]['attribute_label'])

    if attr_idx >= 0:
        n_attr = 1

    if multiple_attr:
        # 每个 attr 一个 ratio

        # [0, 0, ..., 0]，112 维
        # 用来统计，在整个 训练集 中，
        # 对于每个 概念attr ，有多少张 数据样本 用到了这个 概念attr
        n_ones = [0] * n_attr
        # [4796, 4796, ..., 4796]，112 维
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]

    # 对于每个 数据样本
    for d in data:
        # 拿到该 数据样本 的 112 维的 attribute_label 向量
        labels = d['attribute_label']

        if multiple_attr:
            # 统计当前的 数据样本 用到了哪些 概念attr
            # 累加进 n_ones
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)

    # 统计后，n_ones[i] = 训练集 中第 i 个 attr 为 1 的 数据样本 数

    # 算 ratio
    # 如果是 multiple_attr=False，则只 append 一个值
    # 如果是 multiple_attr=False，用 4796 * 112 = 537152 整个这么大的范围算负正比
    for j in range(len(n_ones)):
        imbalance_ratio.append(
            # total[j] = 4796 或 4796 * 112
            # ( total[j] - n_ones[j] ) / n_ones[j] = n_zeros / n_ones = 负样本数 / 正样本数
            total[j] / n_ones[j] - 1
        )
    # 例：imbalance[0] = 10.78
    # 反推：n_ones[0] = ( 1 / (10.78 + 1) ) * 4796 = 407
    # 即，第 0 个 attr 有 407 个 数据样本 上是 1
    # 即，这 407 个 数据样本 都用到了这个 概念attr
    # 407 / 4796 = 8.5%
    # 
    # 例：imbalance[N] = 3.93
    # 反推：n_ones[0] = ( 1 / (3.93 + 1) ) * 4796 = 973
    # 即，第 N 个 attr 有 973 个 数据样本 上是 1
    # 即，这 973 个 数据样本 都用到了这个 概念attr
    # 973 / 4796 = 20.3%
    # 
    # ratio 越大 -> 该 attr 越稀有
    # ratio 越小 -> 该 attr 越常见

    if not multiple_attr:
        # 112 个值都相同
        imbalance_ratio *= n_attr

    # 最终 imbalance 是长度 112 的列表
    return imbalance_ratio