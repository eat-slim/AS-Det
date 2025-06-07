model = dict(
    type='ASDet',
    mode='point',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2AS',
        in_channels=4,
        num_points=(4096, 1024, 512, 256),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8), ()),
        num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 64), ()),
        sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 196, 256), (128, 256, 256)),
                     ()),
        aggregation_channels=(64, 128, 256, 256),
        fps_mods=(dict(type='D-FPS'), dict(type='AS', use_st=True),
                  dict(type='AS', use_st=True), dict(type='AS', use_st=True)),
        query_mods=(('ball', 'ball', 'ball'), ('ball', 'ball', 'ball'), ('ball', 'ball', 'ball'), ()),
        dilated_group=(True, True, True, True),
        fp_channels=(),
        normalize_xyz=False,
        norm='BN',
        bias=False,
        out_indices=(0, 1, 2, 3),
        runtime_cfg={},
    ),
    bbox_head=dict(
        type='ASMSCFAHead',
        decoder_layer_cfg = dict(
            src_idx=[-2, -3, -4],
            in_channels=(256, 128, 64),
            decoder_radii=((6.4,), (3.2,), (2.4,)),
            decoder_num_samples=((32,), (32,), (64,)),
            decoder_query_mods=(('ball-t3d',), ('ball-t3d',), ('hybrid-t3d',)),
            decoder_sa_channels=(((256, 512),), ((256, 512),), ((128, 256, 512),)),
            decoder_aggregation_channels=(512, 512, 512),
            decoder_pooling=('max', 'max', 'max'),
        ),
        pred_layer_cfg = dict(
            in_channels=256,
            shared_conv_channels=(256, 256),
            cls_conv_channels=(256,),
            reg_conv_channels=(256,),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            bias='auto'
        ),
        vote_limit=(3.0, 3.0, 2.0),
        objectness_loss=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0),
        center_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0, beta=1.0 / 9.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=0.2),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0, beta=1.0 / 4.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0, beta=1.0 / 9.0),
        corner_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0 / 8.0),
        proposal_coder=dict(
            type='AnchorFreeAbBBoxCoder', num_dir_bins=12, with_rot=True),
    ),
    # model training and testing settings
    train_cfg=dict(
        expand_dims_length=0.5,
        ignore_dims_length=0.1,
        use_bipartite_matching=False,
        pos_distance_thr=5.,
        as_cfg=dict(
            density_radii=[0.8, 1.6, 4.8],
            density_K=16,
        ),
    ),
    test_cfg=dict(
        nms_cfg=dict(type='rotate', iou_thr=0.1),
        nms_thr=0.1,
        use_rotate_nms=True,
        score_thr=0.1,
        per_class_proposal=False,
        max_output_num=100
    ),
)
