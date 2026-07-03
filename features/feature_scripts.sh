


#Texture
CUDA_VISIBLE_DEVICES=0 python main_all_flower.py --batch_size 128 --max_epoch 30 --prefetch 8 --wdb_name 'Real+Syn S1' \
--meta_lr 1e-3 --meta_weight_decay 1e-4 --meta_optimizer 'AdamW' --meta_scheduler --weight_decay 5e-4 --loss_norm 0 --dataset 'texture' --syn --syn_file 'train_intervene' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--no_meta --lr 0.001 --pre_feature_file 'scores_from_pretrained_model' --test_interval 2 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --scheduler_epoch --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--get_score --warmup_epoch 5 --decay_epoch 27


# CelebA
CUDA_VISIBLE_DEVICES=0 python main_all_flower.py --batch_size 128 --max_epoch 30 --prefetch 8 --wdb_name 'Real+Syn S1' \
--meta_lr 1e-3 --meta_weight_decay 1e-4 --meta_optimizer 'AdamW' --meta_scheduler --weight_decay 5e-4 --loss_norm 0 --dataset 'celeba' --syn --syn_file 'train_intervene' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--no_meta --lr 0.001 --pre_feature_file 'scores_from_pretrained_model' --test_interval 2 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --scheduler_epoch --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--get_score --warmup_epoch 5 --decay_epoch 27


#WaterBirds
CUDA_VISIBLE_DEVICES=1 python main_all_flower.py --batch_size 128 --wdb_name 'Real+Syn S1' --max_epoch 30 \
--meta_lr 1e-3 --meta_weight_decay 1e-4 --meta_optimizer 'Adam' --meta_scheduler --weight_decay 1e-4 --loss_norm 0 --dataset 'waterbirds2' --syn --syn_file 'instructpix2pix_filtered_&_sampled_839' --class_weights \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--no_meta --pre_feature_file 'scores_from_pretrained_model' --lr 0.001 --test_interval 2 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --scheduler_epoch --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1  --get_score



