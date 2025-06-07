import os


dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
use_ground_plane = True
train_pkl = 'kitti_infos_train.pkl'
val_pkl = 'kitti_infos_val.pkl'
test_pkl = 'kitti_infos_val.pkl'
db_pkl = 'kitti_dbinfos_train.pkl'

class_names = ['Car', 'Pedestrian', 'Cyclist']
min_points = dict(Car=5, Pedestrian=10, Cyclist=10)
sample_groups = dict(Car=15, Pedestrian=15, Cyclist=15)
label_merge = dict(Van='Car')

point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='training/velodyne_reduced')
metainfo = dict(classes=class_names)
backend_args = None
input_points = 16384

db_sampler = dict(
    data_root=data_root,
    info_path=os.path.join(data_root, db_pkl),
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points={k: min_points[k] for k in class_names}),
    classes=class_names,
    sample_groups={k: sample_groups[k] for k in class_names},
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=use_ground_plane),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976],
        enable_prob=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1],
        enable_prob=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='PointSample', num_points=input_points),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=input_points),
    dict(type='Pack3DDetInputs', keys=['points'])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=input_points),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=train_pkl,
            data_prefix=data_prefix,
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            label_merge=label_merge,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))
val_dataloader = dict(
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file=val_pkl,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file=test_pkl,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=os.path.join(data_root, val_pkl),
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

