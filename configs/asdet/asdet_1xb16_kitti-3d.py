_base_ = [
    '../_base_/models/asdet_kitti.py',
    '../_base_/datasets/kitti-3d.py',
    '../_base_/schdulers/adam_onecycle_80e.py',
    '../_base_/default_runtime.py'
]


custom_imports = dict(imports=['asdet'], allow_failed_imports=False)


batch_size = 16
num_workers = 8
persistent_workers = True
ckpt_interval = 10
sync_bn = 'torch'


class_names = ['Car', 'Pedestrian', 'Cyclist']
mean_size = dict(Car=[3.9, 1.6, 1.56], Pedestrian=[0.8, 0.6, 1.73], Cyclist=[1.76, 0.6, 1.73])


model = dict(
    bbox_head=dict(
        num_classes=len(class_names),
        bbox_coder=dict(
            type='AnchorFreeReBBoxCoder', num_dir_bins=12, with_rot=True,
            mean_size=[mean_size[k] for k in class_names]),
    ),
)


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers if persistent_workers else 0,
    persistent_workers=persistent_workers,
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers if persistent_workers else 0,
    persistent_workers=persistent_workers,
)


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=ckpt_interval,
                    save_best='Kitti metric/pred_instances_3d/KITTI/Overall_3D_AP40_moderate', rule='greater'),
    logger=dict(type='LoggerHook', interval=50))
custom_hooks = [
    dict(type='TensorboardHook', root='auto', interval=10, metric_format='kitti'),
    dict(type='PerformanceRecordHook', key='Overall_3D_AP40_moderate'),
    dict(type='VisualTestHook', root='auto', interval=1000),
]
randomness = dict(seed=3407)
