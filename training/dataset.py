from PIL import Image
import os
import os.path
import pickle
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
from utils import *

def get_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)

def normalize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std(unbiased=False)  # use population std to match NumPy default
    return (tensor - mean) / (std+ 1e-8)



class Custom_Dataset_MetaFolder(data.Dataset):

    def __init__(self, real_root='', syn_root= '', mode='', transform=None, args=None):

        self.transform = transform
        self.mode = mode
        self.args = args
        self.meta = False
        class_names = sorted([d for d in os.listdir(os.path.join(real_root, 'train'))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        # Initialize the input features from offline generative model
        if 'SD_' in args.feature and self.mode=='train':
            if not args.no_meta:
                self.feature = []
                with open(f'{real_root}/{args.feature_file}.pkl', "rb") as f:
                    feature_dict = pickle.load(f)
                if args.feature_norm:
                    SD_Features = ['SD_real_center', 'SD_syn_center', 'SD_real_3nearest', 'SD_syn_3nearest', 'SD_knn_acc', 'SD_weighted_knn_acc', 'SD_distance_median']
                    for sd_feature in SD_Features:
                        real_values = list(feature_dict['Real'][sd_feature].values())
                        syn_values = list(feature_dict['Syn'][sd_feature].values())
                        all_values = torch.cat(real_values + syn_values, dim=0)
                        mean = all_values.mean()
                        std = all_values.std()
                        for cls in class_names:
                            feature_dict['Real'][sd_feature][cls] = (feature_dict['Real'][sd_feature][cls] - mean) / std
                            feature_dict['Syn'][sd_feature][cls] = (feature_dict['Syn'][sd_feature][cls] - mean) / std
            else:
                self.feature = [0] * 100000

        # Prepare real data
        self.samples = []
        self.labels = []
        self.flags = [] # Indicate whether it is synthetic or real
        for cls_name in class_names:
            if args.dataset == 'PACS' and mode == 'test': #Use the training set of target domain as the test dataset
                cls_folder = os.path.join(os.path.join(real_root, 'train'), cls_name)
            else:
                cls_folder = os.path.join(os.path.join(real_root, mode), cls_name)

            image_files = sorted(os.listdir(cls_folder))
            for index, fname in enumerate(image_files):
                if fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")or fname.lower().endswith(".png"):
                    fpath = os.path.join(cls_folder, fname)
                    self.samples.append(fpath)
                    self.labels.append(self.class_to_idx[cls_name])
                    self.flags.append(0)

                    if 'SD_' in args.feature and not args.no_meta and mode=='train':
                        SD_Features = ['SD_real_center', 'SD_syn_center', 'SD_real_3nearest', 'SD_syn_3nearest', 'SD_knn_acc', 'SD_weighted_knn_acc', 'SD_distance_median']
                        feature_list = []
                        for sd_feature in SD_Features:
                            if sd_feature in args.feature:
                                feature_list.append(feature_dict['Real'][sd_feature][cls_name][index])
                        self.feature.append(torch.cat(feature_list, dim=0))

        if mode == 'val' and args.dataset == 'waterbirds2':
            extra = 'extra'
            for cls_name in class_names:
                cls_folder = os.path.join(os.path.join(real_root, extra), cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg") or fname.lower().endswith(".png"):
                        fpath = os.path.join(cls_folder, fname)
                        self.samples.append(fpath)
                        self.labels.append(self.class_to_idx[cls_name])
                        self.flags.append(0)
        print(f"{len(self.samples)} Real Images in {mode} dataset.")


        # Synthetic Data
        if mode=='train' and args.syn:
            syn_samples = []
            syn_labels = []
            syn_flags = []

            for cls_name in class_names:
                cls_folder = os.path.join(syn_root, cls_name)
                image_files = sorted(os.listdir(cls_folder))
                for index, fname in enumerate(image_files):
                    if fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")or fname.lower().endswith(".png"):

                        fpath = os.path.join(cls_folder, fname)
                        syn_samples.append(fpath)
                        syn_labels.append(self.class_to_idx[cls_name])
                        syn_flags.append(1)

                        if 'SD_' in args.feature and not args.no_meta:
                            SD_Features = ['SD_real_center', 'SD_syn_center', 'SD_real_3nearest', 'SD_syn_3nearest', 'SD_knn_acc', 'SD_weighted_knn_acc', 'SD_distance_median']
                            feature_list = []
                            for sd_feature in SD_Features:
                                if sd_feature in args.feature:
                                    feature_list.append(feature_dict['Syn'][sd_feature][cls_name][index])
                            self.feature.append(torch.cat(feature_list, dim=0))

            print(f"{len(syn_flags)} Synthetic Images in {mode} dataset.")

            if args.only_syn:
                self.samples, self.labels, self.flags = syn_samples, syn_labels, syn_flags
            else:
                self.samples+= syn_samples
                self.labels+=syn_labels
                self.flags+=syn_flags

        # Initialize class weight for WaterBirds following originial paper
        if self.args.dataset == 'waterbirds2' and self.mode == 'train' and self.args.class_weights:
            self.class_weights = get_counts(self.labels).to(device=args.device)

        # Initialize the input features from pretrained classification model
        if 'pre_' in self.args.feature and self.mode=='train':
            if not self.args.no_meta:
                with open(f"{real_root}/{args.pre_feature_file}.pkl", "rb") as f:
                    scores = pickle.load(f)
                pre_feature_list = []
                if 'pre_gradient' in args.feature:
                    pre_feature_list.append(normalize_tensor(scores["grad_norm"]))
                if 'pre_logit' in args.feature:
                    pre_feature_list.append(normalize_tensor(scores["error_norm"]))
                if 'pre_forgetting' in args.feature:
                    pre_feature_list.append(normalize_tensor(scores["forgetting"]))
                self.pre_feature = torch.stack(pre_feature_list, dim=1)
            else:
                self.pre_feature = [0]*100000

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        img, target = self.samples[index], self.labels[index]
        img = Image.open(img).convert('RGB')
        img = self.transform(img)

        if self.mode=='train' and not self.meta:
            if 'SD_' in self.args.feature:
                feature = self.feature[index]
            else:
                feature = 0

            if 'pre_' in self.args.feature:
                pre_feature = self.pre_feature[index]
            else:
                pre_feature = 0

            return index, img, target, (feature, pre_feature), self.flags[index]

        return img, target





def build_dataset(args, real_root):

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [d.strip() for d in cuda_env.split(",") if d.strip()]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    if args.dataset == 'waterbirds2':
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    elif args.dataset in ['PACS', 'celeba', 'texture']:
        jitter = 0.4
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=[0.8, 1.0]),
            transforms.ColorJitter(brightness=jitter,
                                   contrast=jitter,
                                   saturation=jitter,
                                   hue=min(0.5, jitter)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('Not Support')

    if args.dataset == 'PACS':
        real_root = os.path.join(Data_Path[args.dataset], args.source_domain)
    syn_root = os.path.join(real_root, args.syn_file)
    drop_last = True

    train_data = Custom_Dataset_MetaFolder(real_root=real_root, syn_root=syn_root, mode='train', transform=train_transform, args=args)
    if len(devices) > 1 and args.different_batch_across_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size//int(os.environ["WORLD_SIZE"]), num_workers=args.prefetch,
                                                  pin_memory=True, drop_last=drop_last, sampler=train_sampler)
    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.prefetch, shuffle=True, pin_memory=True, drop_last=drop_last)


    score_data = Custom_Dataset_MetaFolder(real_root=real_root, syn_root=syn_root, mode='train', transform=test_transform, args=args)
    scoreloader = torch.utils.data.DataLoader(score_data, batch_size=1, num_workers=args.prefetch, shuffle=False, pin_memory=True, drop_last=False)

    if args.dataset == 'PACS':
        real_root = os.path.join(Data_Path[args.dataset], args.target_domain)

    test_data = Custom_Dataset_MetaFolder(real_root=real_root, syn_root=syn_root, mode='test',transform=test_transform, args=args)
    testloader = [torch.utils.data.DataLoader(test_data, batch_size=args.val_batch_size, num_workers=args.prefetch, pin_memory=True)]

    valid_data = Custom_Dataset_MetaFolder(real_root=real_root, syn_root=syn_root, mode='val', transform=test_transform, args=args)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=args.val_batch_size, num_workers=args.prefetch, pin_memory=True)

    if args.meta_no_augmentation:
        meta_transform = test_transform
    else:
        meta_transform = train_transform

    meta_data = Custom_Dataset_MetaFolder(real_root=real_root, syn_root=syn_root, mode='val', transform=meta_transform, args=args)
    metaloader = torch.utils.data.DataLoader(meta_data, batch_size=args.meta_batch_size, num_workers=args.prefetch, shuffle=True, pin_memory=True, drop_last=drop_last)

    if hasattr(args, "original_cls_training") and args.original_cls_training == True:

        if len(devices) > 1 and args.different_batch_across_gpu:
            train_selector_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            train_dataloader_selector = torch.utils.data.DataLoader(train_data,
                                                                    batch_size=args.training_bz_for_selector//int(os.environ["WORLD_SIZE"]),
                                                                    num_workers=args.prefetch,
                                                                    pin_memory=True, drop_last=drop_last, sampler=train_selector_sampler)
        else:
            train_dataloader_selector = torch.utils.data.DataLoader(train_data, batch_size=args.training_bz_for_selector, num_workers=args.prefetch, shuffle=True, pin_memory=True, drop_last=drop_last)

        return trainloader, metaloader, testloader, validloader, scoreloader, train_dataloader_selector

    return trainloader, metaloader, testloader, validloader, scoreloader




