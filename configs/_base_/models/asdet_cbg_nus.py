cls_weight = 1.0
loc_weight = 0.25

model = dict(
    type='ASDet',
    mode='point',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2AS',
        in_channels=5,
        num_points=(16384, 4096, 2048, 1024),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8), ()),
        num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 64), ()),
        sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 196, 256), (128, 256, 256)),
                     ()),
        aggregation_channels=(64, 128, 256, 256),
        fps_mods=(dict(type='D-FPS'), dict(type='D-FPS'), dict(type='AS', use_st=True), dict(type='AS', use_st=True)),
        query_mods=(('ball', 'ball', 'ball'), ('ball', 'ball', 'ball'), ('ball', 'ball', 'ball'), ()),
        dilated_group=(True, True, True, True),
        fp_channels=(),
        normalize_xyz=False,
        norm='BN',
        bias=False,
        out_indices=(0, 1, 2, 3),
        runtime_cfg={}
    ),
    bbox_head=dict(
        type='ASCBGNusHead',
        in_channels=256,
        decoder_layer_cfg=dict(
            src_idx=[-2, -3, -4],
            in_channels=(256, 128, 64),
            decoder_radii=((6.4,), (3.2,), (3.2,)),
            decoder_num_samples=((32,), (32,), (64,)),
            decoder_query_mods=(('ball-t3d',), ('ball-t3d',), ('hybrid-t3d',)),
            decoder_sa_channels=(((256, 512),), ((256, 512),), ((128, 256, 512),)),
            decoder_aggregation_channels=(512, 512, 512),
            decoder_pooling=('max', 'max', 'max'),
        ),
        pred_layer_cfg=dict(
            separate_head_cfg=[
                dict(out_channels=0, num_conv=2),  # cls
                dict(out_channels=3, num_conv=2),  # x, y, z
                dict(out_channels=3, num_conv=2),  # l, w, h
                dict(out_channels=2, num_conv=2),  # sin, cos
                dict(out_channels=2, num_conv=2),  # vx, vy
            ],
        ),
        vote_limit=(3.0, 3.0, 2.0),
        objectness_loss=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0 * cls_weight),
        center_loss=dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=1.0 * loc_weight),
        size_res_loss=dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=1.0 * loc_weight),
        dir_res_loss=dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=0.2 * loc_weight),
        velocity_loss=dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=1.0 * loc_weight),
        bbox_coder=dict(
            type='AnchorFreeNusBBoxCoder', with_rot=True, with_vel=True, absolute_height=False)
    ),
    # model training and testing settings
    train_cfg=dict(
        expand_dims_length=0.5,
        nearest_thr=5.0,
        inner_assign=True,
        as_cfg=dict(
            density_radii=[0.8, 1.6, 4.8],
            density_K=16,
        ),
    ),
    test_cfg=dict(
        nms_cfg=dict(type='rotate', iou_thr=0.2),
        max_obj_per_sample=1024,
        score_thr=0.1,
        max_output_num=500,
    )
)
