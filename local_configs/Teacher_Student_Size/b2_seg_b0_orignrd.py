_base_ = [
    '../_base_/datasets/ade20k_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)

b0_cfg = dict(
        type='EncoderDecoder',
        pretrained='pretrained/mit_b0.pth',
        backbone=dict(
            type='mit_b0',
            style='pytorch'),
        decode_head=dict(
            type='BilinearPADHead_fast',
            num_classes=150,
            upsample_factor=8,
            dyn_branch_ch=16,
            mask_head_ch=16,
            in_channels=256,
            in_index=3,
            channels=256,
            dilations=(1, 3, 6, 9),
            c1_in_channels=32,
            c1_channels=16,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            channel_reduce_factor=2,
            loss_decode=dict(
                type='CrossEntropyLoss', loss_weight=1.0)),
        # auxiliary_head=dict(
        #     type='FCNHead',
        #     in_channels=160,
        #     in_index=2,
        #     channels=128,
        #     num_convs=1,
        #     concat_input=False,
        #     dropout_ratio=0.1,
        #     num_classes=150,
        #     norm_cfg=norm_cfg,
        #     align_corners=False,
        #     loss_decode=dict(
        #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
)

b1_cfg = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='nrd_no_aspp',
        num_classes=150,
        upsample_factor=8,
        dyn_branch_ch=16,
        mask_head_ch=16,
        in_channels=512,
        in_index=3,
        channels=512,
        dilations=(1, 3, 6, 9),
        c1_in_channels=64,
        c1_channels=48,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        channel_reduce_factor=2,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)

b2_cfg = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b2.pth',
    backbone=dict(
        type='mit_b2',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        # type='MLPHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
)


b3_cfg = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b3.pth',
    backbone=dict(
        type='mit_b3',
        style='pytorch'),
    decode_head=dict(
        type='nrd_no_aspp',
        num_classes=150,
        upsample_factor=8,
        dyn_branch_ch=16,
        mask_head_ch=16,
        in_channels=512,
        in_index=3,
        channels=512,
        dilations=(1, 3, 6, 9),
        c1_in_channels=64,
        c1_channels=48,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        channel_reduce_factor=2,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)

b4_cfg = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b4.pth',
    backbone=dict(
        type='mit_b4',
        style='pytorch'),
    decode_head=dict(
        type='nrd_no_aspp',
        num_classes=150,
        upsample_factor=8,
        dyn_branch_ch=16,
        mask_head_ch=16,
        in_channels=512,
        in_index=3,
        channels=512,
        dilations=(1, 3, 6, 9),
        c1_in_channels=64,
        c1_channels=48,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        channel_reduce_factor=2,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)

t_num = '2'
s_num = '0'
cfg_t = eval(f'b{t_num}_cfg')
cfg_s = eval(f'b{s_num}_cfg')

model = dict(
    type='SDModule',
    cfg_s=cfg_s,
    cfg_t=cfg_t,
    distillation = [
        {'student_layer':'decode_head.out',
        'teacher_layer':'decode_head.linear_pred',
        'loss_name':'CGDLoss',
        'loss_config':{},
        },
    ],
    t_pretrain = f'./pretrained/segformer.b2.512x512.ade.160k.pth',  # 老师的预训练模型
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9,0.999), weight_decay=0.01,
                paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=4,
            workers_per_gpu=4)
evaluation = dict(interval=4000, metric='mIoU')

