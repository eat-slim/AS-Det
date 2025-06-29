lr = 0.001
wd = 0.01
max_epochs = 20
val_interval = 2

# learning rate / momentum schedule
lr /= 10
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=round(max_epochs * 0.4),
        eta_min=lr * 10,
        begin=0,
        end=round(max_epochs * 0.4),
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - round(max_epochs * 0.4),
        eta_min=lr * 1e-4,
        begin=round(max_epochs * 0.4),
        end=max_epochs * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    dict(
        type='CosineAnnealingMomentum',
        T_max=round(max_epochs * 0.4),
        eta_min=0.85,
        begin=0,
        end=round(max_epochs * 0.4),
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=max_epochs - round(max_epochs * 0.4),
        eta_min=0.95,
        begin=round(max_epochs * 0.4),
        end=max_epochs * 1,
        by_epoch=True,
        convert_to_iter_based=True)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=wd, betas=(0.95, 0.99)),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# training schedule for 20 epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
