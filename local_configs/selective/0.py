_base_ = [
    '../_base_/models/distill.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
log_config = dict(  
    interval=50, 
    hooks=[
        dict(type='TensorboardLoggerHook') 
        # dict(type='TextLoggerHook')
    ])
work_dir = './work_dirs/selective/kld_4'

model = dict(
        distillation = dict(
        layers=[
            ['decode_head.linear_pred','decode_head.linear_pred',[150,150],4]
        ],
        weights_init_strategy='equal',
        parse_mode='regular',
        use_attn=False,
        selective='distill_0'
    ),
    s_pretrain = './pretrained/mit_b1.pth',
    t_pretrain = './pretrained/segformer.b4.512x512.ade.160k.pth',
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

data = dict(samples_per_gpu=4)
evaluation = dict(interval=2000, metric='mIoU')  