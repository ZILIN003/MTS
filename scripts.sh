export MASTER_PORT=25000



#WaterBirds
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --wdb_name 'WB' --max_epoch 450 \
--meta_lr 1e-3 --meta_weight_decay 1e-4 --meta_optimizer 'AdamW' --meta_scheduler --weight_decay 1e-4 --loss_norm 0 --dataset 'waterbirds2' --syn --syn_file 'instructpix2pix_filtered_&_sampled_839' --class_weights \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--pre_feature_file 'scores_from_pretrained_model' --lr 0.001 --test_interval 15 --model_id resnet50 --pretrained --val_batch_size 128 --scheduler_epoch --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 --product_eps 1e-2 --pseudo_lr 0.01 --pseudo_eval \
--original_cls_training --training_bz_for_selector 1024 --different_batch_across_gpu



#Texture
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 250 --prefetch 2 --wdb_name 'T' \
--meta_lr 1e-4 --meta_weight_decay 1e-4 --meta_optimizer 'AdamW' --meta_scheduler --weight_decay 5e-4 --loss_norm 0 --dataset 'texture' --syn --syn_file 'train_intervene' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --pre_feature_file 'scores_from_pretrained_model' --test_interval 10 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --scheduler_epoch --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--warmup_epoch 25 --decay_epoch 225 --meta_warmup_epoch 25 --feature_file 'SD_ALL_feature' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 --original_cls_training --training_bz_for_selector 1024 --meta_batch_size 256 --different_batch_across_gpu




#CelebA
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 300 --prefetch 8 --wdb_name 'C' \
--meta_lr 5e-4 --meta_weight_decay 1e-4 --meta_optimizer 'AdamW' --meta_scheduler --weight_decay 5e-4 --loss_norm 0 --dataset 'celeba' --syn --syn_file 'train_intervene' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --pre_feature_file 'scores_from_pretrained_model' --test_interval 20 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --scheduler_epoch --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--warmup_epoch 30 --decay_epoch 270 --meta_warmup_epoch 30 --feature_file 'SD_ALL_feature' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 --original_cls_training --training_bz_for_selector 1024 --meta_batch_size 256 --different_batch_across_gpu





#PACS

####art as source
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 1000 --weight_decay 0.0005 --prefetch 8 --wdb_name 'art2cartoon' \
--meta_lr 1e-4 --meta_weight_decay 1e-4 --meta_warmup_epoch 100 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'art_painting_to_cartoon' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 100 --decay_epoch 900 --scheduler_epoch --test_interval 50 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_cartoon_50' --feature_file 'SD_ALL_feature_cartoon' --source_domain 'art_painting' --target_domain 'cartoon' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 128 --different_batch_across_gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 1000 --weight_decay 0.0005 --prefetch 8 --wdb_name 'art2photo' \
--meta_lr 5e-4 --meta_weight_decay 1e-4 --meta_warmup_epoch 100 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'art_painting_to_photo' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 100 --decay_epoch 900 --scheduler_epoch --test_interval 50 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_photo_50' --feature_file 'SD_ALL_feature_photo' --source_domain 'art_painting' --target_domain 'photo' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 128 --different_batch_across_gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 1000 --weight_decay 0.0005 --prefetch 8 --wdb_name 'art2sketch' \
--meta_lr 5e-4 --meta_weight_decay 1e-4 --meta_warmup_epoch 100 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'art_painting_to_sketch' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 100 --decay_epoch 900 --scheduler_epoch --test_interval 50 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_sketch_50' --feature_file 'SD_ALL_feature_sketch' --source_domain 'art_painting' --target_domain 'sketch' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 256 --different_batch_across_gpu



####cartoon as source
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 800 --weight_decay 0.0005 --prefetch 8 --wdb_name 'cartoon2art' \
--meta_lr 1e-4 --meta_weight_decay 1e-4 --meta_warmup_epoch 80 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'cartoon_to_art_painting' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 80 --decay_epoch 720 --scheduler_epoch --test_interval 40 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_art_50' --feature_file 'SD_ALL_feature_art' --source_domain 'cartoon' --target_domain 'art_painting' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 208 --different_batch_across_gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 800 --weight_decay 0.0005 --prefetch 8 --wdb_name 'cartoon2photo' \
--meta_lr 1e-3 --meta_weight_decay 1e-4 --meta_warmup_epoch 80 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'cartoon_to_photo' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 80 --decay_epoch 720 --scheduler_epoch --test_interval 40 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_photo_50' --feature_file 'SD_ALL_feature_photo' --source_domain 'cartoon' --target_domain 'photo' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 128 --different_batch_across_gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 800 --weight_decay 0.0005 --prefetch 8 --wdb_name 'cartoon2sketch' \
--meta_lr 1e-4 --meta_weight_decay 1e-4 --meta_warmup_epoch 80 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'cartoon_to_sketch' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 80 --decay_epoch 720 --scheduler_epoch --test_interval 40 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_sketch_50' --feature_file 'SD_ALL_feature_sketch' --source_domain 'cartoon' --target_domain 'sketch' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 256 --different_batch_across_gpu





####sketch as source
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 900 --weight_decay 0.0005 --prefetch 8 --wdb_name 'sketch2art' \
--meta_lr 1e-4 --meta_weight_decay 1e-4 --meta_warmup_epoch 90 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'sketch_to_art_painting' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 90 --decay_epoch 810 --scheduler_epoch --test_interval 30 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_art_50' --feature_file 'SD_ALL_feature_art' --source_domain 'sketch' --target_domain 'art_painting' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 208 --different_batch_across_gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 900 --weight_decay 0.0005 --prefetch 8 --wdb_name 'sketch2photo' \
--meta_lr 1e-4 --meta_weight_decay 1e-4 --meta_warmup_epoch 90 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'sketch_to_photo' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 90 --decay_epoch 810 --scheduler_epoch --test_interval 30 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_photo_50' --feature_file 'SD_ALL_feature_photo' --source_domain 'sketch' --target_domain 'photo' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 128 --different_batch_across_gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 900 --weight_decay 0.0005 --prefetch 8 --wdb_name 'sketch2cartoon' \
--meta_lr 1e-4 --meta_weight_decay 1e-4 --meta_warmup_epoch 90 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'sketch_to_cartoon' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 90 --decay_epoch 810 --scheduler_epoch --test_interval 30 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_cartoon_50' --feature_file 'SD_ALL_feature_cartoon' --source_domain 'sketch' --target_domain 'cartoon' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 128 --different_batch_across_gpu




####photo as source
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 1100 --weight_decay 0.0005 --prefetch 8 --wdb_name 'photo2art' \
--meta_lr 1e-3 --meta_weight_decay 1e-4 --meta_warmup_epoch 100 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'photo_to_art_painting' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 100 --decay_epoch 990 --scheduler_epoch --test_interval 50 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_art_50' --feature_file 'SD_ALL_feature_art' --source_domain 'photo' --target_domain 'art_painting' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 208 --different_batch_across_gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 1100 --weight_decay 0.0005 --prefetch 8 --wdb_name 'photo2sketch' \
--meta_lr 1e-3 --meta_weight_decay 1e-4 --meta_warmup_epoch 100 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'photo_to_sketch' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 100 --decay_epoch 990 --scheduler_epoch --test_interval 50 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_sketch_50' --feature_file 'SD_ALL_feature_sketch' --source_domain 'photo' --target_domain 'sketch' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 256 --different_batch_across_gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT github/main_parallel.py --batch_size 1024 --max_epoch 1100 --weight_decay 0.0005 --prefetch 8 --wdb_name 'photo2cartoon' \
--meta_lr 1e-3 --meta_weight_decay 1e-4 --meta_warmup_epoch 100 --meta_optimizer 'AdamW' --meta_scheduler --loss_norm 0 --dataset 'PACS' --syn --syn_file 'photo_to_cartoon' --nesterov \
--feature 'CLS_nearest_real_cos, class, CLS_center_cos, group, SD_real_center, SD_syn_center, SD_real_3nearest, SD_syn_3nearest, SD_weighted_knn_acc, SD_distance_median, pre_gradient, pre_logit, pre_forgetting' \
--lr 0.001 --warmup_epoch 100 --decay_epoch 990 --scheduler_epoch --test_interval 50 --all_real --model_id resnet50 --pretrained --val_batch_size 128 --lr_min 0.0 --weight_store_interval 50 --feature_norm 'batch_norm' --meta_net_num_layers 2 --seed 1 \
--pre_feature_file 'scores_from_pretrained_model_cartoon_50' --feature_file 'SD_ALL_feature_cartoon' --source_domain 'photo' --target_domain 'cartoon' --pseudo_eval --meta_no_augmentation --train_aug --product_eps 1e-2 --pseudo_lr 0.001 \
--original_cls_training --training_bz_for_selector 1024 --meta_batch_size 128 --different_batch_across_gpu








#meta_lr search: {1e-3, 5e-4, 1e-4}









