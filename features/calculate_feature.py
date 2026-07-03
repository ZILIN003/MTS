import torch.nn.functional as F
import pickle
import torch
import os

# waterbirds celeba texture sketch/photo/art/cartoon
root_path = 'data/waterbirds'

class_names = sorted([d for d in os.listdir(os.path.join(root_path, 'train'))])
with open(f'{root_path}/real_ebd.pkl', 'rb') as f:
    real_feature = pickle.load(f)
with open(f'{root_path}/syn_ebd.pkl', 'rb') as f:
    syn_feature = pickle.load(f)

feature = {'Syn':{}, 'Real':{}}
feature_dict_1 = {}
feature_dict_2 = {}
feature_dict_3 = {}
feature_dict_4 = {}
feature_dict_3_new = {}
feature_dict_4_new = {}
for index, cls in enumerate(class_names):
    syn_f = syn_feature[cls].float()
    real_f = real_feature[cls].float()
    # Normalize a and b for cosine similarity
    syn_f_norm = F.normalize(syn_f, dim=1)
    real_f_norm = F.normalize(real_f, dim=1)

    # 1. Cosine between each element in a and mean of b
    mean_b = F.normalize(real_f.mean(dim=0, keepdim=True), dim=1)  # [1, 4096]
    cos_a_mean_b = torch.matmul(syn_f_norm, mean_b.T)  # [500, 1]
    feature_dict_1[cls] = cos_a_mean_b

    # 2. Cosine between each element in a and mean of a
    mean_a = F.normalize(syn_f.mean(dim=0, keepdim=True), dim=1)  # [1, 4096]
    cos_a_mean_a = torch.matmul(syn_f_norm, mean_a.T)  # [500, 1]
    feature_dict_2[cls] = cos_a_mean_a

    # 3. Cosine between each element in a and nearest 3 elements in b
    cos_matrix_ab = torch.matmul(syn_f_norm, real_f_norm.T)
    top3_cos_ab_vals, top3_cos_ab_ids = torch.topk(cos_matrix_ab, k=3, dim=1)  # [500, 3]
    feature_dict_3[cls] = top3_cos_ab_vals
    feature_dict_3_new[cls] = top3_cos_ab_vals[:,0].unsqueeze(1)

    # 4. Cosine between each element in a and nearest 3 elements in a (excluding self)
    cos_matrix_a = torch.matmul(syn_f_norm, syn_f_norm.T)
    cos_matrix_a.fill_diagonal_(-1)
    top3_cos_a_vals, top3_cos_a_ids = torch.topk(cos_matrix_a, k=3, dim=1)  # [500, 3]
    feature_dict_4[cls] = top3_cos_a_vals
    feature_dict_4_new[cls] = top3_cos_a_vals[:,0].unsqueeze(1)


feature['Syn']['SD_real_center'] = feature_dict_1
feature['Syn']['SD_syn_center'] = feature_dict_2
feature['Syn']['SD_real_3nearest'] = feature_dict_3
feature['Syn']['SD_syn_3nearest'] = feature_dict_4
feature['Syn']['SD_real_nearest'] = feature_dict_3_new
feature['Syn']['SD_syn_nearest'] = feature_dict_4_new



feature_dict_5 = {}
feature_dict_6 = {}
feature_dict_7 = {}
feature_dict_8 = {}
feature_dict_7_new = {}
feature_dict_8_new = {}
for index, cls in enumerate(class_names):
    syn_f = syn_feature[cls].float()
    real_f = real_feature[cls].float()
    # Normalize a and b for cosine similarity
    syn_f_norm = F.normalize(syn_f, dim=1)
    real_f_norm = F.normalize(real_f, dim=1)

    # 1. Cosine between each element in a and mean of b
    mean_b = F.normalize(real_f.mean(dim=0, keepdim=True), dim=1)  # [1, 4096]
    cos_a_mean_b = torch.matmul(real_f_norm, mean_b.T)  # [500, 1]
    feature_dict_5[cls] = cos_a_mean_b

    # 2. Cosine between each element in a and mean of a
    mean_a = F.normalize(syn_f.mean(dim=0, keepdim=True), dim=1)  # [1, 4096]
    cos_a_mean_a = torch.matmul(real_f_norm, mean_a.T)  # [500, 1]
    feature_dict_6[cls] = cos_a_mean_a

    # 3. Cosine between each element in a and nearest 3 elements in b
    cos_matrix_ab = torch.matmul(real_f_norm, real_f_norm.T)
    cos_matrix_ab.fill_diagonal_(-1)
    top3_cos_ab_vals, top3_cos_ab_ids = torch.topk(cos_matrix_ab, k=3, dim=1)  # [500, 3]
    feature_dict_7[cls] = top3_cos_ab_vals
    feature_dict_7_new[cls] = top3_cos_ab_vals[:,0].unsqueeze(1)

    # 4. Cosine between each element in a and nearest 3 elements in a (excluding self)
    cos_matrix_a = torch.matmul(real_f_norm, syn_f_norm.T)
    top3_cos_a_vals, top3_cos_a_ids = torch.topk(cos_matrix_a, k=3, dim=1)  # [500, 3]
    feature_dict_8[cls] = top3_cos_a_vals
    feature_dict_8_new[cls] = top3_cos_a_vals[:,0].unsqueeze(1)

feature['Real']['SD_real_center'] = feature_dict_5
feature['Real']['SD_syn_center'] = feature_dict_6
feature['Real']['SD_real_3nearest'] = feature_dict_7
feature['Real']['SD_syn_3nearest'] = feature_dict_8
feature['Real']['SD_real_nearest'] = feature_dict_7_new
feature['Real']['SD_syn_nearest'] = feature_dict_8_new


all_features = []
all_labels = []
all_class_names = []

sample_index = 0
index_ranges = {"Real": {}, "Syn": {}}
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}

# Store start/end index per (source, class)
for cls in class_names:
    real_f = real_feature[cls].float()
    syn_f = syn_feature[cls].float()

    # Track index range
    index_ranges["Real"][cls] = list(range(sample_index, sample_index + real_f.size(0)))
    all_features.append(real_f)
    all_labels.extend([cls] * real_f.size(0))
    sample_index += real_f.size(0)

    index_ranges["Syn"][cls] = list(range(sample_index, sample_index + syn_f.size(0)))
    all_features.append(syn_f)
    all_labels.extend([cls] * syn_f.size(0))
    sample_index += syn_f.size(0)

features = torch.cat(all_features, dim=0)  # [N, D]
features = F.normalize(features, dim=1)
similarity = features @ features.T
similarity.fill_diagonal_(-float("inf"))
knn_values, knn_indices = similarity.topk(k=5, dim=1)

labels_tensor = torch.tensor([class_to_index[label] for label in all_labels])
neighbor_labels = labels_tensor[knn_indices]
target_labels = labels_tensor.unsqueeze(1).expand_as(neighbor_labels)

correct = (neighbor_labels == target_labels).sum(dim=1)
acc_per_sample = correct.float() / 5.0

# Boolean matrix of whether each neighbor matches the target
correct_matrix = (neighbor_labels == target_labels)
# print((knn_values < 0).sum().item())

# row_min = knn_values.min(dim=1, keepdim=True).values
# row_max = knn_values.max(dim=1, keepdim=True).values
# knn_values = (knn_values - row_min) / (row_max - row_min)

weights = knn_values / (knn_values.sum(dim=1, keepdim=True) + 1e-8)  # Shape: [N, 5]
weighted_acc_per_sample = (correct_matrix * weights).sum(dim=1)  # Shape: [N]

results = {"Real": {}, "Syn": {}}

for cls in class_names:
    for src in ["Real", "Syn"]:
        feature[src].setdefault('SD_knn_acc',{})
        feature[src].setdefault('SD_knn_class', {})
        feature[src].setdefault('SD_weighted_knn_acc', {})
        idxs = index_ranges[src][cls]
        accs = acc_per_sample[idxs]
        weighted_accs = weighted_acc_per_sample[idxs]
        neighbors = knn_indices[idxs]

        feature[src]['SD_knn_acc'][cls] = accs.unsqueeze(1)
        feature[src]['SD_weighted_knn_acc'][cls] = weighted_accs.unsqueeze(1)
        feature[src]['SD_knn_class'][cls] = labels_tensor[neighbors]



for index, cls in enumerate(class_names):
    syn_f = syn_feature[cls].float()  # shape: [Nsyn, D]
    real_f = real_feature[cls].float()  # shape: [Nreal, D]
    syn_f_norm = F.normalize(syn_f, dim=1)
    real_f_norm = F.normalize(real_f, dim=1)

    all_embeddings = torch.cat([syn_f_norm, real_f_norm], dim=0)  # shape: [Nsyn + Nreal, D]
    mean_embedding = F.normalize(torch.cat([syn_f, real_f], dim=0).mean(dim=0, keepdim=True), dim=1)

    cos_sim_to_mean = torch.matmul(all_embeddings, mean_embedding.T)
    cos_sim_matrix = torch.matmul(all_embeddings, all_embeddings.T)
    cos_sim_matrix.fill_diagonal_(-float('inf'))
    nearest_cos_sim, _ = cos_sim_matrix.max(dim=1)
    nearest_cos_sim = nearest_cos_sim.unsqueeze(1)

    feature['Syn'].setdefault('SD_center', {})
    feature['Real'].setdefault('SD_center', {})
    feature['Syn']['SD_center'][cls] = cos_sim_to_mean[:syn_f.size(0)]
    feature['Real']['SD_center'][cls] = cos_sim_to_mean[syn_f.size(0):]

    feature['Syn'].setdefault('SD_nearest', {})
    feature['Real'].setdefault('SD_nearest', {})
    feature['Syn']['SD_nearest'][cls] = nearest_cos_sim[:syn_f.size(0)]
    feature['Real']['SD_nearest'][cls] = nearest_cos_sim[syn_f.size(0):]


for index, cls in enumerate(class_names):
    syn_f = syn_feature[cls].float()  # shape: [Nsyn, D]
    real_f = real_feature[cls].float()  # shape: [Nreal, D]

    all_embeddings = torch.cat([syn_f, real_f], dim=0)  # shape: [Nsyn + Nreal, D]
    mean_embedding = all_embeddings.mean(dim=0)

    dists = torch.norm(all_embeddings - mean_embedding, p=2, dim=1)
    median_dist = dists.median()
    # abs_dev = (dists - median_dist).abs().unsqueeze(1)  # shape: [Nsyn + Nreal, 1]

    dists_diff = (dists - median_dist).abs()
    sorted_indices = torch.argsort(dists_diff)
    ranks = torch.empty_like(sorted_indices, dtype=torch.float)
    ranks[sorted_indices] = torch.arange(len(dists_diff), dtype=torch.float)
    percentiles = ranks / (len(dists_diff) - 1)
    abs_dev = percentiles.unsqueeze(1)

    # Split back
    feature['Syn'].setdefault('SD_distance_median', {})
    feature['Real'].setdefault('SD_distance_median', {})
    feature['Syn']['SD_distance_median'][cls] = abs_dev[:syn_f.size(0)]
    feature['Real']['SD_distance_median'][cls] = abs_dev[syn_f.size(0):]



with open(f'{root_path}/SD_ALL_feature.pkl', 'wb') as f:
    pickle.dump(feature, f)