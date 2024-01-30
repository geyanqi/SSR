_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/daformer_sepaspp_mitb5.py',
    '../../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    '../../_base_/uda/dacs_a999_fdthings.py',
    '../../_base_/schedules/adamw.py', '../../_base_/schedules/poly10warm.py'
]
gpu_model = 'NVIDIAGeForceRTX2080Ti'
n_gpus = 1
seed = 0
model = dict(
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5'),
    decode_head=dict())
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
uda = dict(
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=32, _delete_=True))
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
name = '231228_2045_gta2cs_dacs_a999_fdthings_rcs001_cpl_m32-07-spta_daformer_sepaspp_mitb5_poly10warm_s0_256e7'
exp = 81
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fdthings_rcs0.01_cpl_m32-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
work_dir = 'work_dirs/local-exp81/231228_2045_gta2cs_dacs_a999_fdthings_rcs001_cpl_m32-07-spta_daformer_sepaspp_mitb5_poly10warm_s0_256e7'
git_rev = '67fe7dd3e81875bee6f07f4f90567ed5b71294ba'
