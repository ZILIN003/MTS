import random
import numpy as np
from time import sleep
import wandb

def set_cudnn(device='cuda'):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def stop_epoch(time=3):
    try:
        print('can break now')
        for i in range(time):
            sleep(1)
        print('wait for next epoch')
        return False
    except KeyboardInterrupt:
        return True


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.

    with torch.inference_mode():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = move_batch_to_device(inputs,device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            del outputs, inputs, labels

    return total_loss / (batch_idx + 1), correct / len(data_loader.dataset), 0


def compute_loss_accuracy_new(net, data_loader, criterion, device):
    net.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_loss = 0.

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()

            _, pred_top5 = outputs.topk(5, dim=1)
            correct_top1 += (pred_top5[:, 0] == labels).sum().item()
            correct_top5 += sum([labels[i] in pred_top5[i] for i in range(labels.size(0))])

    avg_loss = total_loss / (batch_idx + 1)
    acc_top1 = correct_top1 / len(data_loader.dataset)
    acc_top5 = correct_top5 / len(data_loader.dataset)

    return avg_loss, acc_top1, acc_top5


Data_Path = {'waterbirds2': 'data/waterbirds',
             'PACS': 'data/pacs',
             'celeba':'data/celeba',
             'texture': 'data/texture'}

Model_Weights = {
    'resnet50': 'pretrained_models/resnet50-0676ba61.pth' #Download from https://github.com/pytorch/vision/tree/main/references/classification#resnet
}


def get_class_num(args):
    if args.dataset == 'waterbirds2':
        num_classes = 2
    elif args.dataset in ['PACS', 'celeba']:
        num_classes = 7
    elif args.dataset == 'texture':
        num_classes = 16
    else:
        raise ValueError("Not Support Dataset {}".format(args.dataset))
    print("{} classes in dataset {}".format(num_classes, args.dataset))

    return num_classes



import math
import torch

class ScalarEMA:
    def __init__(self, args=None, eps=1e-8):
        self.args = args
        self.decay = args.ema_decay
        self.eps = eps
        self.ema_stats = {}  # key -> {"mean": float, "var": float}

    def update(self, key: str, x: torch.Tensor):
        x = x.detach()  # avoid tracking gradients
        mean = x.mean().item()
        var = x.var(unbiased=False).item()

        if key not in self.ema_stats:
            # Initialize EMA for this key
            self.ema_stats[key] = {"mean": mean, "var": var}
        else:
            # Update EMA for this key
            m = self.ema_stats[key]["mean"]
            v = self.ema_stats[key]["var"]
            self.ema_stats[key]["mean"] = self.decay * m + (1 - self.decay) * mean
            self.ema_stats[key]["var"]  = self.decay * v + (1 - self.decay) * var

    def normalize(self, key: str, x: torch.Tensor):

        if self.args.feature_norm == 'batch_norm':
            mean = x.mean()
            std = torch.sqrt(x.var(unbiased=False) + self.eps)
        elif self.args.feature_norm == 'ema_norm':
            self.update(key, x)
            wandb.log({f'{key} Mean': self.ema_stats[key]["mean"],
                       f'{key} Var': self.ema_stats[key]["var"]}, commit=False)
            if key not in self.ema_stats:
                raise RuntimeError(f"EMA for key '{key}' not initialized. Call update(key, x) first.")
            mean = self.ema_stats[key]["mean"]
            std = math.sqrt(self.ema_stats[key]["var"] + self.eps)
        return (x - mean) / std

    def get_stats(self, key: str):
        if key not in self.ema_stats:
            raise RuntimeError(f"EMA for key '{key}' not initialized.")
        mean = self.ema_stats[key]["mean"]
        std = math.sqrt(self.ema_stats[key]["var"] + self.eps)
        return mean, std


import torch.nn.functional as F
def compute_metrics_for_single_sample(net, input_tensor, label, criterion, ord=2):
    """
    Computes loss, gradient norm, and prediction error norm for a single sample.

    Args:
        net: PyTorch model
        input_tensor: Tensor of shape [1, ...]
        label: Tensor of shape [1]
        criterion: loss function (e.g., nn.CrossEntropyLoss())
        ord: which norm to use for prediction error (e.g., 2 for L2)

    Returns:
        loss_value: scalar float
        grad_norm: scalar float (L2 norm of gradients)
        error_norm: scalar float (||softmax(pred) - one-hot(label)||)
    """
    # Forward pass
    output = net(input_tensor)  # logits, shape [1, num_classes]
    loss = criterion(output, label)
    loss_value = loss.item()

    # === 1. Gradient norm ===
    net.zero_grad()
    grads = torch.autograd.grad(loss, net.parameters(), retain_graph=False, create_graph=False)
    grad_norm = torch.sqrt(sum(g.norm(2) ** 2 for g in grads if g is not None)).item()

    # === 2. Prediction error norm ===
    probs = F.softmax(output, dim=1)  # shape [1, C]
    one_hot_label = F.one_hot(label, num_classes=output.size(1)).float()  # shape [1, C]
    error = probs - one_hot_label  # shape [1, C]
    error_norm = torch.norm(error, p=ord, dim=1).item()

    return loss_value, grad_norm, error_norm



def move_batch_to_device(batch, device):
    """Move dict of tensors to device."""
    if isinstance(batch, (list, tuple)):
        return tuple(v.to(device) for v in batch)
    else:
        # assume a single tensor
        return batch.to(device)

def cat_batch(batch_1, batch_2):
    """Move dict of tensors to device."""
    return tuple(torch.cat([b1, b2], dim=0) for b1,b2 in zip(batch_1,batch_2))


import torch.distributed as dist
def ddp_global_mean_var_max(x):
    world_size = dist.get_world_size()
    local_n = x.numel()
    total_n = local_n * world_size

    # local sums
    local_sum = x.sum()
    local_sq_sum = (x * x).sum()
    local_max = x.max()

    # global reductions
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

    # global stats
    global_mean = local_sum / total_n
    global_var = (local_sq_sum / total_n) - global_mean * global_mean
    global_max = local_max

    return global_mean.cpu(), global_var.cpu(), global_max.cpu()



def setup_for_distributed(is_master: bool):
    """
    This function disables printing when not in master process.
    """
    import builtins as __builtin__

    # Save the original print
    builtin_print = __builtin__.print

    # Redefine print
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        # Only master process (or forced) prints
        if is_master or force:
            builtin_print(*args, **kwargs)

    # Replace the built-in print with the new one
    __builtin__.print = print
