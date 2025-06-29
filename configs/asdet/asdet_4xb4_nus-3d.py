_base_ = [
    '../_base_/models/asdet_cbg_nus.py',
    '../_base_/datasets/nus-3d.py',
    '../_base_/schdulers/adam_onecycle_20e.py',
    '../_base_/default_runtime.py'
]


custom_imports = dict(imports=['asdet'], allow_failed_imports=False)


batch_size = 4
num_workers = 4
persistent_workers = True
ckpt_interval = 2
sync_bn = 'torch'


tasks = [
    dict(class_names=['car']),
    dict(class_names=['truck', 'construction_vehicle']),
    dict(class_names=['bus', 'trailer']),
    dict(class_names=['barrier']),
    dict(class_names=['motorcycle', 'bicycle']),
    dict(class_names=['pedestrian', 'traffic_cone']),
]


model = dict(
    bbox_head=dict(
        tasks=tasks,
    ),
)


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1 if persistent_workers else 0,
    persistent_workers=persistent_workers,
)
test_dataloader = dict(
    batch_size=1,
    num_workers=1 if persistent_workers else 0,
    persistent_workers=persistent_workers,
)


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=ckpt_interval),
    logger=dict(type='LoggerHook', interval=50))
custom_hooks = [
    dict(type='TensorboardHook', root='auto', interval=10, metric_format='nuscenes'),
    dict(type='PerformanceRecordHook', key='mAP'),
    dict(type='VisualTestHook', root='auto', interval=1000),
]
