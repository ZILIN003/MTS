import argparse
import torch.optim
from model import *
from utils import *
import wandb
import time
from dataset import build_dataset
import torch.nn.functional as F
import os
import copy
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.nn.functional as dist_nn


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
parser.add_argument('--different_batch_across_gpu', default=False, action='store_true')

args = parser.parse_args()
print(args)

def meta_weight_net():

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.manual_seed_all(args.seed)

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

    print('Using DataParallel')
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    meta_net = DDP(meta_net, device_ids=[local_rank], output_device=local_rank)

    if hasattr(args, "original_cls_training") and args.original_cls_training==True:
        train_dataloader, meta_dataloader, test_dataloader, valid_dataloader, score_dataloader, train_dataloader_selector = build_dataset(args, Data_Path[args.dataset])
        train_dataloader_selector_iter = iter(train_dataloader_selector)
    else:
        train_dataloader, meta_dataloader, test_dataloader, valid_dataloader, score_dataloader = build_dataset(args, Data_Path[args.dataset])
    print("Steps per Epoch: ", len(train_dataloader))
    print("Total Steps: ", args.max_epoch * len(train_dataloader))
    if meta_dataloader is not None:
        meta_dataloader_iter = iter(meta_dataloader)

    class_weights = train_dataloader.dataset.class_weights if hasattr(train_dataloader.dataset, 'class_weights') else None
    print('Class Weight: ', class_weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device=args.device)
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

    if not args.no_meta:
        decay = []
        no_decay = []
        for name, param in meta_net.named_parameters():
            if name.endswith("bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        params = [{'params': decay, 'weight_decay': args.meta_weight_decay},
                {'params': no_decay, 'weight_decay': 0.0}]
        meta_optimizer = torch.optim.AdamW(params, lr=args.meta_lr)

        if args.meta_warmup_epoch > 0:
            num_meta_warmup_steps = args.meta_warmup_epoch * len(train_dataloader)
            if args.meta_scheduler:
                warmup_meta_scheduler = torch.optim.lr_scheduler.LambdaLR(meta_optimizer, lr_lambda=lambda step: min((step + 1) / num_meta_warmup_steps, 1.0))
                meta_decay_steps = args.max_epoch * len(train_dataloader) - num_meta_warmup_steps
                _meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max = meta_decay_steps)
                meta_scheduler = torch.optim.lr_scheduler.SequentialLR(meta_optimizer, schedulers=[warmup_meta_scheduler, _meta_scheduler], milestones=[num_meta_warmup_steps])
            else:
                meta_scheduler = torch.optim.lr_scheduler.LambdaLR(meta_optimizer, lr_lambda=lambda step: min((step + 1) / num_meta_warmup_steps, 1.0))
        else:
            if args.meta_scheduler:
                meta_decay_steps = args.max_epoch * len(train_dataloader)
                meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=meta_decay_steps)
            else:
                meta_scheduler = torch.optim.lr_scheduler.LambdaLR(meta_optimizer, lr_lambda=lambda step: 1.0)

    if args.feature_norm:
        ema = ScalarEMA(args)

    torch.set_printoptions(precision=20)
    best_acc = -1.0
    flag = 0
    total_start_time = time.time()

    # ############
    # train_dataloader_copy = copy.deepcopy(train_dataloader)
    # train_dataloader_copy_iter = iter(train_dataloader_copy)
    # ############

    for epoch in range(args.max_epoch):
        epoch_start_time = time.time()

        train_dataloader.sampler.set_epoch(epoch)
        print(f"Epoch {epoch}/{args.max_epoch} - Training...")
        for iteration, (idxs, inputs, labels, extra_features, flags) in enumerate(train_dataloader):

            inputs, labels = inputs.to(args.device), labels.to(args.device)
            flags = flags.view(-1, 1).to(args.device)
            cur_bz = inputs.shape[0]

            if not args.no_meta:

                # ############
                # try:
                #     idxs, inputs, labels, extra_features, flags = next(train_dataloader_copy_iter)
                # except StopIteration:
                #     train_dataloader_copy_iter = iter(train_dataloader_copy)
                #     train_dataloader_copy.sampler.set_epoch(flag)
                #     idxs, inputs, labels, extra_features, flags = next(train_dataloader_copy_iter)
                #
                # inputs, labels = inputs.to(args.device), labels.to(args.device)
                # flags = flags.view(-1, 1).to(args.device)
                # cur_bz = inputs.shape[0]
                # ############

                try:
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                meta_inputs, meta_labels = meta_inputs.to(args.device), meta_labels.to(args.device)

                pseudo_net = copy.deepcopy(net.module).to(local_rank)
                dist.barrier()
                pseudo_net = DDP(pseudo_net, device_ids=[local_rank], output_device=local_rank)

                ############# Get the features of selection network - Start
                with torch.no_grad():
                    pseudo_net.eval()
                    meta_feature_list = []

                    all_images = torch.cat([inputs, meta_inputs], dim=0)
                    all_outputs, all_features = pseudo_net(all_images, return_features=True)
                    pseudo_features, val_anchor_features = all_features[:len(inputs)], all_features[len(inputs):]
                    pseudo_cossim = torch.matmul(
                        F.normalize(pseudo_features, dim=1, eps=1e-6),
                        F.normalize(val_anchor_features, dim=1, eps=1e-6).T)

                    if 'CLS_nearest_real_cos' in args.feature:
                        nearest_cossim, nearest_indices = torch.topk(pseudo_cossim, k=args.topk, largest=True, dim=1)
                        if args.feature_norm:
                            nearest_cossim = ema.normalize('CLS_nearest_real_cos', nearest_cossim)
                        meta_feature_list.append(nearest_cossim)
                    if 'CLS_center_cos' in args.feature:
                        pseudo_center = val_anchor_features.mean(dim=0, keepdim=True)
                        pseudo_cossim_to_center = F.cosine_similarity(pseudo_features,
                                                                      pseudo_center.expand_as(pseudo_features),
                                                                      dim=1).unsqueeze(1)
                        if args.feature_norm:
                            pseudo_cossim_to_center = ema.normalize('CLS_center_cos', pseudo_cossim_to_center)
                        meta_feature_list.append(pseudo_cossim_to_center)

                    if 'SD' in args.feature:
                        SD_features = extra_features[0]
                        meta_feature_list.append(SD_features.float().to(args.device))
                    if 'pre' in args.feature:
                        pre_features = extra_features[1].to(args.device)
                        meta_feature_list.append(pre_features)

                if 'class' in args.feature:
                    meta_feature_list.append(meta_net.module.class_embed(labels))
                if 'group' in args.feature:
                    meta_feature_list.append(meta_net.module.group_embed(flags.squeeze()))
                ############# Get the features of selection network - End

                pseudo_net.train()
                pseudo_outputs, pseudo_features = pseudo_net(inputs, return_features=True)
                pseudo_loss_vector = F.cross_entropy(pseudo_outputs, labels.long(), reduction='none')
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))

                meta_feature = torch.cat(meta_feature_list, dim=1)
                meta_net.train()
                pseudo_weight = meta_net(meta_feature)

                if class_weights is not None:
                    pseudo_weight = pseudo_weight * class_weights[labels].unsqueeze(dim=1)
                local_sum = pseudo_weight.sum()
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                global_sum = local_sum
                target_sum = cur_bz * world_size
                pseudo_weight = pseudo_weight / global_sum * target_sum
                pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)

                pseudo_optimizer = torch.optim.SGD(pseudo_net.parameters(), lr=0.0)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.zero_grad()
                pseudo_loss.backward()
                pseudo_optimizer.step()

                #################### Calculate the Gradient of Selector - Start
                # 1. Get Validation Gradient Vector Gamma
                if args.pseudo_eval:
                    pseudo_net.eval()
                meta_outputs = pseudo_net(meta_inputs)
                meta_loss = criterion_val(meta_outputs, meta_labels.long())

                if local_rank == 0:
                    wandb.log({"Meta Loss": meta_loss,
                              "Meta Learning Rate": meta_optimizer.param_groups[0]["lr"]}, commit=False)
                val_mean_grad = torch.autograd.grad(meta_loss, pseudo_net.parameters()) # theta
                val_mean_grad_vec = torch.cat([p.reshape(-1).detach().clone() for p in val_mean_grad])

                pseudo_net = copy.deepcopy(net.module).to(local_rank)
                dist.barrier()
                pseudo_net = DDP(pseudo_net, device_ids=[local_rank], output_device=local_rank)

                if args.original_cls_training == True:  # You can switch it to 'False' for acceleration

                    original_inputs = [copy.deepcopy(inputs), copy.deepcopy(labels),
                                       copy.deepcopy(meta_feature_list[:-2]), copy.deepcopy(flags)]

                    try:
                        idxs, inputs, labels, extra_features, flags = next(train_dataloader_selector_iter)
                    except StopIteration:
                        train_dataloader_selector_iter = iter(train_dataloader_selector)
                        train_dataloader_selector.sampler.set_epoch(flag)
                        idxs, inputs, labels, extra_features, flags = next(train_dataloader_selector_iter)
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    flags = flags.view(-1, 1).to(args.device)
                    cur_bz = inputs.shape[0]

                    ############# 2.1 Get the features of selection network for the new training batch - Start
                    with torch.no_grad():
                        pseudo_net.eval()
                        meta_feature_list = []

                        all_images = torch.cat([inputs, meta_inputs], dim=0)
                        all_outputs, all_features = pseudo_net(all_images, return_features=True)
                        pseudo_features, val_anchor_features = all_features[:len(inputs)], all_features[len(inputs):]
                        pseudo_cossim = torch.matmul(
                            F.normalize(pseudo_features, dim=1, eps=1e-6),
                            F.normalize(val_anchor_features, dim=1, eps=1e-6).T)

                        if 'CLS_nearest_real_cos' in args.feature:
                            nearest_cossim, nearest_indices = torch.topk(pseudo_cossim, k=args.topk, largest=True,
                                                                         dim=1)
                            if args.feature_norm:
                                nearest_cossim = ema.normalize('CLS_nearest_real_cos', nearest_cossim)
                            meta_feature_list.append(nearest_cossim)
                        if 'CLS_center_cos' in args.feature:
                            pseudo_center = val_anchor_features.mean(dim=0, keepdim=True)
                            pseudo_cossim_to_center = F.cosine_similarity(pseudo_features,
                                                                          pseudo_center.expand_as(pseudo_features),
                                                                          dim=1).unsqueeze(1)
                            if args.feature_norm:
                                pseudo_cossim_to_center = ema.normalize('CLS_center_cos', pseudo_cossim_to_center)
                            meta_feature_list.append(pseudo_cossim_to_center)

                    if 'SD' in args.feature:
                        SD_features = extra_features[0]
                        meta_feature_list.append(SD_features.float().to(args.device))
                    if 'pre' in args.feature:
                        pre_features = extra_features[1].to(args.device)
                        meta_feature_list.append(pre_features)

                    meta_feature_list.append(None)
                    meta_feature_list.append(None)
                    ############# 2.1 Get the features of selection network for the new training batch - End

                # 3. theta+epsilon*Gamma
                product_eps = args.product_eps / torch.norm(val_mean_grad_vec)
                for p, grad in zip(pseudo_net.parameters(), val_mean_grad):
                    p.data = p.data + product_eps * grad.clone()
                pseudo_net.train()
                pseudo_outputs, pseudo_features = pseudo_net(inputs, return_features=True)
                pseudo_loss_vector = F.cross_entropy(pseudo_outputs, labels.long(), reduction='none')
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))

                meta_feature_list = meta_feature_list[:-2]
                meta_feature_list.append(meta_net.module.class_embed(labels))
                meta_feature_list.append(meta_net.module.group_embed(flags.squeeze()))
                meta_feature = torch.cat(meta_feature_list, dim=1)

                pseudo_weight = meta_net(meta_feature)
                local_sum = pseudo_weight.sum()
                local_sum = dist_nn.all_reduce(local_sum, op=dist.ReduceOp.SUM)  # local_sum =
                global_sum = local_sum
                target_sum = cur_bz * world_size
                pseudo_weight = pseudo_weight / global_sum * target_sum
                pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)
                meta_grad_1 = torch.autograd.grad(pseudo_loss, meta_net.parameters(), retain_graph=False)

                # 4. theta-epsilon*Gamma
                for p, grad in zip(pseudo_net.parameters(), val_mean_grad):
                    p.data = p.data - 2 * product_eps * grad.clone()
                pseudo_outputs, pseudo_features = pseudo_net(inputs, return_features=True)
                pseudo_loss_vector = F.cross_entropy(pseudo_outputs, labels.long(), reduction='none')
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))

                meta_feature_list = meta_feature_list[:-2]
                meta_feature_list.append(meta_net.module.class_embed(labels))
                meta_feature_list.append(meta_net.module.group_embed(flags.squeeze()))
                meta_feature = torch.cat(meta_feature_list, dim=1)

                pseudo_weight = meta_net(meta_feature)
                local_sum = pseudo_weight.sum()
                local_sum = dist_nn.all_reduce(local_sum, op=dist.ReduceOp.SUM)  # local_sum =
                global_sum = local_sum
                target_sum = cur_bz * world_size
                pseudo_weight = pseudo_weight / global_sum * target_sum
                pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)
                meta_grad_2 = torch.autograd.grad(pseudo_loss, meta_net.parameters(), retain_graph=False)

                # 5. Vector Matrix Product
                HvProduct = tuple(g1 - g2 for g1, g2 in zip(meta_grad_1, meta_grad_2))
                constant = 1.0 + args.momentum if args.nesterov else 1.0
                if args.pseudo_lr>0:
                    gradient = tuple(-args.pseudo_lr * g * constant / (2*product_eps) for g in HvProduct)  # phi
                else:
                    gradient = tuple(-optimizer.param_groups[0]["lr"] * g * constant / (2*product_eps) for g in HvProduct)

                avg_grad = []
                for g in gradient:
                    g_avg = g.detach().clone()
                    dist.all_reduce(g_avg, op=dist.ReduceOp.SUM)
                    g_avg /= world_size
                    avg_grad.append(g_avg)
                gradient = tuple(avg_grad)
                for p, grad in zip(meta_net.parameters(), gradient):
                    p.grad = grad
                #################### Finish Gradient Calculation

                meta_optimizer.step()
                meta_scheduler.step()
                del pseudo_net, pseudo_optimizer
                torch.cuda.empty_cache()

            if args.no_meta:
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())
            else:
                if args.original_cls_training == True:

                    inputs, labels, meta_feature_list, flags = original_inputs[0],original_inputs[1],original_inputs[2],original_inputs[3]
                    meta_feature_list.append(meta_net.module.class_embed(labels))
                    meta_feature_list.append(meta_net.module.group_embed(flags.squeeze()))
                    meta_feature = torch.cat(meta_feature_list, dim=1)
                    cur_bz = inputs.shape[0]

                net.train()
                outputs, features = net(inputs, return_features=True)
                loss_vector = F.cross_entropy(outputs, labels.long(), reduction='none')
                loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

                with torch.no_grad():
                    meta_net.eval()
                    weight = meta_net(meta_feature)
                    if class_weights is not None:
                        weight = weight * class_weights[labels].unsqueeze(dim=1)
                    local_sum = weight.sum()
                    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                    global_sum = local_sum
                    target_sum = cur_bz * world_size
                    weight = weight / global_sum * target_sum
                loss = torch.mean(weight * loss_vector_reshape)

                if local_rank == 0:
                    wandb.log({"Unweighted Training Loss": torch.mean(loss_vector_reshape.detach()),
                               "Mean Weight": torch.mean(weight),
                               "Max_Prop": torch.max(weight/cur_bz),
                               "All Data Weight Variance after Normalization": weight.var()}, commit=False)

            if local_rank == 0:
                wandb.log({"Training Loss": loss,
                           "Learning Rate": optimizer.param_groups[0]["lr"]})

            if flag % args.paint_interval==0:
                print(f'Epoch: {epoch}, Training Loss: {loss:.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not args.scheduler_epoch:
                scheduler.step()
            flag+=1

        if args.scheduler_epoch:
            scheduler.step()

        if local_rank == 0:
            if (epoch+1) % args.test_interval == 0:
                print('Computing Test Result...')

                test_loss, test_accuracy, test_accuracy_5 = compute_loss_accuracy(
                    net=net.module,
                    data_loader=test_dataloader[0],
                    criterion=criterion_val,
                    device=args.device,
                )

                if best_acc<test_accuracy:
                    best_acc = test_accuracy
                if local_rank == 0:
                    wandb.log({"Test Loss": test_loss,
                               "Test Accuracy": test_accuracy,
                               "Best Test Accuracy": best_acc}, commit = False)
                    print('Epoch: {}, (Loss, Accuracy) Test: ({:.4f}, {:.2%})'.format(
                        epoch,
                        test_loss,
                        test_accuracy
                    ))

        if local_rank == 0:
            wandb.log({"Epoch": epoch}, commit=False)
        epoch_time = time.time() - epoch_start_time  # Calculate epoch time
        print(f"Epoch {epoch} completed in {epoch_time/60:.2f} mins")


    total_time = time.time() - total_start_time  # Calculate total training time
    print(f"Total training time: {total_time/60:.2f} mins")


if __name__ == '__main__':

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [d.strip() for d in cuda_env.split(",") if d.strip()]

    print('Using DataParallel')
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    setup_for_distributed(is_master=(local_rank == 0))

    if local_rank == 0:

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

    meta_weight_net()



