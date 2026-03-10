muon_lr = 0.01
adam_lr = 0.001
wd = 0.01
betas = (0.9, 0.95)
max_epochs = 20
val_interval = 2

# learning rate / momentum schedule
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=round(max_epochs * 0.4),
        eta_min_ratio=10,
        begin=0,
        end=round(max_epochs * 0.4),
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - round(max_epochs * 0.4),
        eta_min_ratio=1e-5,
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
    constructor='MuonOptimWrapperConstructor',
    optimizer=dict(
        type='MuonWithAuxAdam',
        min_hidden_dims=16,
        lr1=muon_lr / 10,
        lr2=adam_lr / 10,
        weight_decay=wd,
        betas=betas,
    ),
    clip_grad=dict(
        max_norm=35,
        norm_type=2,
    )
)

# training schedule for 20 epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
