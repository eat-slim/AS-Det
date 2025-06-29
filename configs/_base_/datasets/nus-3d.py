import os


dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
train_pkl = 'nuscenes_infos_train.pkl'
val_pkl = 'nuscenes_infos_val.pkl'
test_pkl = 'nuscenes_infos_val.pkl'
db_pkl = 'nuscenes_dbinfos_train.pkl'

tasks = [
    dict(class_names=['car']),
    dict(class_names=['truck', 'construction_vehicle']),
    dict(class_names=['bus', 'trailer']),
    dict(class_names=['barrier']),
    dict(class_names=['motorcycle', 'bicycle']),
    dict(class_names=['pedestrian', 'traffic_cone']),
]
class_names = []
for task in tasks:
    class_names += task['class_names']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
metainfo = dict(classes=class_names)
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=os.path.join(data_root, db_pkl),
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=2,
        construction_vehicle=2,
        bus=2,
        trailer=2,
        barrier=2,
        motorcycle=2,
        bicycle=2,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='NuscenesPointSample', num_points_keyframe=16384, num_points_others=49152, voxel_size=(0.1, 0.1, 0.1)),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='NuscenesPointSample', num_points_keyframe=16384, num_points_others=49152, voxel_size=(0.1, 0.1, 0.1)),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points'])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='NuscenesPointSample', num_points_keyframe=16384, num_points_others=49152, voxel_size=(0.1, 0.1, 0.1)),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=train_pkl,
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
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
        ann_file=val_pkl,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_pkl,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=os.path.join(data_root, val_pkl),
    metric='bbox',
    jsonfile_prefix=f'results/nuscenes_val_results',
    backend_args=backend_args)
test_evaluator = val_evaluator
