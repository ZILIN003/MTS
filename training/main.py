import argparse
import torch.optim
from model import *
from utils import *
import wandb
import time
from dataset import build_dataset
import torch.nn.functional as F
import os
import pickle
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import copy
from torch.optim.lr_scheduler import LambdaLR, StepLR

parser = argparse.ArgumentParser(description='Meta_Weight_Net')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--wdb_name', type=str, default='test')
parser.add_argument('--paint_interval', type=int, default=100)

parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)
parser.add_argument('--meta_optimizer', type=str, default='Adam')
parser.add_argument('--meta_dropout', type=float, default=0.)
parser.add_argument('--meta_lr', type=float, default=1e-5)
parser.add_argument('--meta_weight_decay', type=float, default=0.)
parser.add_argument('--meta_momentum', type=float, default=0.9)
parser.add_argument('--meta_scheduler', default=False, action='store_true')
parser.add_argument('--no_meta', action='store_true') # Indicate whether train a selector or not
parser.add_argument('--meta_warmup_epoch', type=int, default=0)
parser.add_argument('--pseudo_lr', type=float, default=-1.)
parser.add_argument('--meta_net_path', type=str, default=None)
parser.add_argument('--meta_no_augmentation', default=False, action='store_true')

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--lr_min', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', action='store_true')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--decay_epoch', type=int, default=-1)
parser.add_argument('--lr_decay_factor', type=float, default=0.1)
parser.add_argument('--cos_scheduler', type=bool, default=False)
parser.add_argument('--model_id', type=str, default='resnet18')
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--scheduler_epoch', default=False, action='store_true')

parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--val_batch_size', type=int, default=128)
parser.add_argument('--meta_batch_size', type=int, default=128)
parser.add_argument('--source_domain', type=str, default='_')
parser.add_argument('--target_domain', type=str, default='_')
parser.add_argument('--prefetch', type=int, default=4)
parser.add_argument('--class_weights', default=False, action='store_true')
parser.add_argument('--syn', action='store_true')
parser.add_argument('--only_syn', action='store_true')
parser.add_argument('--save_file', type=str, default=None)
parser.add_argument('--save_interval', type=int, default=500)
parser.add_argument('--feature', type=str, default='') #latent_space_feature
parser.add_argument('--syn_file', type=str, default='generated_train/blended')
parser.add_argument('--score_path', type=str, default='')

parser.add_argument('--class_embed_size', type=int, default=4)
parser.add_argument('--feature_norm', type=str, choices=['batch_norm', 'ema_norm'], default=None)
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--feature_file', type=str, default='SD_ALL_feature')
parser.add_argument('--pre_feature_file', type=str, default='scores_from_pretrained_model')
parser.add_argument('--ema_decay', type=float, default=0.9)
parser.add_argument('--get_score', default=False, action='store_true')
parser.add_argument('--pseudo_eval', default=False, action='store_true')
parser.add_argument('--train_aug', default=False, action='store_true')
parser.add_argument('--product_eps', type=float, default=1e-4)
parser.add_argument('--original_cls_training', default=False, action='store_true') # Train the selector with the training batch different with the training batch used by classifier, when original_cls_training==True
parser.add_argument('--training_bz_for_selector', type=int, default=256)

args = parser.parse_args()
print(args)


def meta_weight_net():

    set_cudnn(device=args.device)
    set_seed(seed=args.seed)

    num_classes = get_class_num(args)
    input_dimension = 22
    meta_net = MLP(hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers, input_dim = input_dimension, num_classes=num_classes, args=args).to(device=args.device)
    print(f"Meta Net Input Feature: {args.feature}, Input Dimension: {input_dimension}")
    print("Number of Selection Model Parameters: ", sum(p.numel() for p in meta_net.parameters()))

    if 'resnet' in args.model_id:
        net = ResNet(args.model_id, num_classes, args.dataset).to(device=args.device)
        if args.pretrained:
            checkpoint = torch.load(Model_Weights[args.model_id], map_location=args.device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            net.load_state_dict(filtered_state_dict, strict=False)
    else:
        raise ValueError(f'No model named {args.model_id}')
    print("Number of Classification Model Parameters: ", sum(p.numel() for p in net.parameters()))

    if hasattr(args, "original_cls_training") and args.original_cls_training==True:
        train_dataloader, meta_dataloader, test_dataloader, valid_dataloader, score_dataloader, train_dataloader_selector = build_dataset(args, Data_Path[args.dataset])
        train_dataloader_selector_iter = iter(train_dataloader_selector)
    else:
        train_dataloader, meta_dataloader, test_dataloader, valid_dataloader, score_dataloader = build_dataset(args, Data_Path[args.dataset])
    print("Steps per Epoch: ", len(train_dataloader))
    print("Total Steps: ", args.max_epoch * len(train_dataloader))

    if args.get_score: # Pretrain a classifier without selection to get some features
        forget_stats = SimpleNamespace()
        num_train_examples = len(train_dataloader.dataset)
        forget_stats.prev_accs = np.zeros(num_train_examples, dtype=np.int32)
        forget_stats.num_forgets = np.zeros(num_train_examples, dtype=float)
        forget_stats.never_correct = np.arange(num_train_examples, dtype=np.int32)

    class_weights = train_dataloader.dataset.class_weights if hasattr(train_dataloader.dataset, 'class_weights') else None
    print('Class Weight: ', class_weights)
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights).to(device=args.device)
    criterion_val = torch.nn.CrossEntropyLoss().to(device=args.device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    if args.dataset in ['PACS', 'celeba', 'texture'] and args.decay_epoch > 0 and not args.cos_scheduler:
        def learning_rate_lambda(epoch, warmup_epochs = args.warmup_epoch, decay_epoch= args.decay_epoch, decay_factor=args.lr_decay_factor):
            if epoch < warmup_epochs:
                return (epoch + 1) / (warmup_epochs + 1)
            elif epoch >= decay_epoch:
                return decay_factor
            else:
                return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda=learning_rate_lambda)
    else:
        T_max = args.max_epoch if args.scheduler_epoch else args.max_epoch * len(train_dataloader)
        if args.warmup_epoch>0:
            num_warmup_steps = args.warmup_epoch if args.scheduler_epoch else args.warmup_epoch * len(train_dataloader)
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / num_warmup_steps, 1.0))
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max - num_warmup_steps, eta_min = args.lr_min)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr_min)


    torch.set_printoptions(precision=20)
    flag = 0
    total_start_time = time.time()


    for epoch in range(args.max_epoch):

        print(f"Epoch {epoch}/{args.max_epoch} - Training...")
        for iteration, (idxs, inputs, labels, extra_features, flags) in enumerate(train_dataloader):

            inputs, labels = inputs.to(args.device), labels.to(args.device)

            #Update the classifier
            if args.no_meta:
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())

                if args.get_score:
                    accs = (outputs.argmax(dim=1) == labels).cpu().numpy().astype(np.int32)
                    forget_stats.num_forgets[idxs[forget_stats.prev_accs[idxs] > accs]] += 1
                    forget_stats.prev_accs[idxs] = accs
                    forget_stats.never_correct = np.setdiff1d(forget_stats.never_correct, idxs[accs.astype(bool)], True)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not args.scheduler_epoch:
                scheduler.step()
            flag+=1

        if args.scheduler_epoch:
            scheduler.step()


    if args.get_score and args.no_meta:
        print("Computing Scores......")
        net.eval()
        loss_tensor = torch.zeros(len(score_dataloader.dataset))
        grad_norm_tensor = torch.zeros(len(score_dataloader.dataset))
        error_norm_tensor = torch.zeros(len(score_dataloader.dataset))
        for idx, inputs, labels, SD_features, flags in tqdm(score_dataloader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            loss_val, grad_norm, error_norm = compute_metrics_for_single_sample(net, inputs, labels, criterion, ord=2)
            loss_tensor[idx] = loss_val
            grad_norm_tensor[idx] = grad_norm
            error_norm_tensor[idx] = error_norm

        save_dict = {
            "loss": loss_tensor,
            "grad_norm": grad_norm_tensor,
            "error_norm": error_norm_tensor,
            "forgetting": torch.Tensor(forget_stats.num_forgets),
            "never_forgetting": torch.Tensor(forget_stats.never_correct),

        }

        if args.source_domain != '_':
            with open(f"{Data_Path[args.dataset]}/{args.source_domain}/{args.pre_feature_file}.pkl", "wb") as f:
                pickle.dump(save_dict, f)
        else:
            with open(f"{Data_Path[args.dataset]}/{args.pre_feature_file}.pkl", "wb") as f:
                pickle.dump(save_dict, f)

    total_time = time.time() - total_start_time  # Calculate total training time
    print(f"Total training time: {total_time/60:.2f} mins")

if __name__ == '__main__':

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [d.strip() for d in cuda_env.split(",") if d.strip()]

    wandb.init(project="WeightMetaNet",
               name=args.wdb_name,
               tags=[args.dataset, args.model_id, args.target_domain]
               )
    config_para = wandb.config
    config_para.seed = args.seed
    config_para.batch_size = args.batch_size
    config_para.val_batch_size = args.val_batch_size
    config_para.lr = args.lr
    config_para.momentum = args.momentum
    config_para.weight_decay = args.weight_decay
    config_para.meta_lr = args.meta_lr
    config_para.meta_weight_decay = args.meta_weight_decay
    config_para.max_epoch = args.max_epoch
    config_para.meta_scheduler = args.meta_scheduler
    config_para.syn = args.syn
    config_para.only_syn = args.only_syn
    config_para.syn_file = args.syn_file
    config_para.feature = args.feature
    config_para.save_file = args.save_file
    config_para.dataset = args.dataset

    meta_weight_net()






