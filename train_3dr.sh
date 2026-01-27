#    #!/bin/bash
MODEL="Image2PointsModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11)"

TRAIN_DATASET="2000 @ Co3d_Seq(num_views=11, sel_num=3, degree=180, mask_bg='rand', split='train', aug_crop=16, resolution=224, transform=ColorJitter, seed=233) + \
        2000 @ ScanNet_Seq(num_views=11, num_seq=100, max_thresh=20, split='train', resolution=224, transform=ColorJitter, aug_crop=256, seed=666)"

TEST_DATASET="1000 @ Co3d_Seq(num_views=11, sel_num=3, degree=180, mask_bg='rand', split='test', resolution=224, seed=666) + \
        500 @ ScanNet_Seq(num_views=11, num_seq=50, max_thresh=20, split='test', resolution=224, seed=666)"

# Stage 1: Train the i2p model for pointmap prediction
PRETRAINED="checkpoints/i2p/slam3dr_i2p_stage1/checkpoint-last.pth" # "" i2p checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth
TRAIN_OUT_DIR="checkpoints/i2p/slam3dr_i2p_stage1"

torchrun --nproc_per_node=2 train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --model "$MODEL" \
    --train_criterion "Jointnorm_ConfLoss(Jointnorm_Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Jointnorm_Regr3D(L21, norm_mode='avg_dis')" \
    --pretrained $PRETRAINED \
    --pretrained_type "slam3r" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 1 --epochs 40 --batch_size 16 --accum_iter 1 \
    --save_freq 2 --keep_freq 20 --eval_freq 1 --print_freq 20 --num_workers 8 \
    --save_config\
    --freeze "encoder"\
    --loss_func 'i2p' \
    --output_dir $TRAIN_OUT_DIR \
    --ref_id -1


# # Stage 2: Train a simple mlp to predict the correlation score
PRETRAINED="checkpoints/i2p/slam3dr_i2p_stage1/checkpoint-last.pth"
TRAIN_OUT_DIR="checkpoints/i2p/slam3dr_i2p"

torchrun --nproc_per_node=2 train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --model "$MODEL" \
    --train_criterion "Jointnorm_ConfLoss(Jointnorm_Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Jointnorm_Regr3D(L21, gt_scale=True)" \
    --pretrained $PRETRAINED \
    --pretrained_type "slam3r" \
    --lr 1e-6 --min_lr 1e-8 --warmup_epochs 1 --epochs 10 --batch_size 16 --accum_iter 1 \
    --save_freq 2 --keep_freq 20 --eval_freq 1 --print_freq 20 --num_workers 8 \
    --save_config\
    --freeze "corr_score_head_only"\
    --loss_func "i2p_corr_score" \
    --output_dir $TRAIN_OUT_DIR \
    --ref_id -1

MODEL="Local2WorldModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 12, need_encoder=True)"

PRETRAINED="checkpoints/slam3r_l2w/checkpoint-last.pth"  #l2w 
TRAIN_OUT_DIR="checkpoints/slam3r_l2w"

torchrun --nproc_per_node=2 train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --model "$MODEL" \
    --train_criterion "Jointnorm_ConfLoss(Jointnorm_Regr3D(L21,norm_mode=None), alpha=0.2)" \
    --test_criterion "Jointnorm_Regr3D(L21, norm_mode=None)" \
    --pretrained $PRETRAINED \
    --pretrained_type "slam3r" \
    --lr 5e-5 --min_lr 5e-7 --warmup_epochs 2 --epochs 40 --batch_size 16 --accum_iter 1 \
    --save_freq 2 --keep_freq 20 --eval_freq 1 --print_freq 20 --num_workers 8\
    --save_config\
    --output_dir $TRAIN_OUT_DIR \
    --freeze "encoder"\
    --loss_func "l2w" \
    --ref_ids 0 1 2 3 4 5


