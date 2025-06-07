import math
import torch
import numpy as np
import open3d as o3d
from typing import List, Union
from math import pi


def show_pcd(pcds: List, colors: List = None, bboxes: List = None, bboxes_colors: List = None,
             balls: List = None, balls_colors: List = None, balls_radii: float = 0.1, window_name: str = "PCD",
             has_normals: bool = False, estimate_normals: bool = False, estimate_kwargs: dict = None,
             filter: bool = False, point_size: float = 1.0, line_width: float = 0.05,
             save_view_option_to: str = '' , render_option_json: str = '', view_option_json: str = '',
             capture_screen_image: str = '',
             width=2000, height=1500) -> None:
    """
    Args:
        pcds: [Array1, Array2, ...] Array.shape = (N, 3+)
        colors: [RGB1, RGB2, ...] RGB.shape = (3,), like [1, 0.5, 0] for R=1, G=0.5, B=0
        bboxes: [[Array1, Array2, ...]] Array.shape = (N, 7)
        colors: [RGB1, RGB2, ...] RGB.shape = (3,), like [1, 0.5, 0] for R=1, G=0.5, B=0
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    # vis.get_render_option().show_coordinate_frame = True
    # vis.get_render_option().save_to_json('open3d.json')
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    print(f'{window_name:*<30}')
    for i in range(len(pcds)):
        pcd_o3d = o3d.open3d.geometry.PointCloud()
        if isinstance(pcds[i], np.ndarray):
            pcd_points = pcds[i][:, :3]
        elif isinstance(pcds[i], torch.Tensor):
            pcd_points = pcds[i][:, :3].detach().cpu().numpy()
        else:
            pcd_points = np.array(pcds[i][:, :3])
        pcd_o3d.points = o3d.open3d.utility.Vector3dVector(pcd_points)

        if has_normals:
            if pcds[i].shape[1] < 6:
                print('Normals is NOT found')
            else:
                if isinstance(pcds[i], np.ndarray):
                    pcd_normals = pcds[i][:, 3:6]
                elif isinstance(pcds[i], torch.Tensor):
                    pcd_normals = pcds[i][:, 3:6].detach().cpu().numpy()
                else:
                    pcd_normals = np.array(pcds[i][:, 3:6])
                pcd_o3d.normals = o3d.open3d.utility.Vector3dVector(pcd_normals)

        if filter:
            pcd_o3d = pcd_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=3)[0]

        if estimate_normals:
            if estimate_kwargs is None:
                radius, max_nn = 1, 30
            else:
                assert 'radius' in estimate_kwargs.keys() and 'max_nn' in estimate_kwargs.keys()
                radius, max_nn = estimate_kwargs['radius'], estimate_kwargs['max_nn']
            pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

        if colors is not None:
            pcd_o3d.paint_uniform_color(colors[i])
        vis.add_geometry(pcd_o3d)
        print(pcd_o3d)

    if bboxes is not None:
        cnt = 0
        for i, bbox_set in enumerate(bboxes):
            cnt += len(bbox_set)
            color = bboxes_colors[i] if bboxes_colors is not None else [0, 0, 0]

            bbox_set = bbox2cylinder(bbox_set, color, radius=line_width)
            for bbox in bbox_set:
                vis.add_geometry(bbox)
        print(f'num_bboxes = {cnt}')

    if balls is not None:
        cnt = 0
        for i, ball_set in enumerate(balls):
            cnt += len(ball_set)
            for one_ball in ball_set:
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=balls_radii, resolution=5)
                mesh_sphere.translate(one_ball)
                if balls_colors is not None:
                    mesh_sphere.paint_uniform_color(balls_colors[i])
                vis.add_geometry(mesh_sphere)
        print(f'num_balls = {cnt}')

    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.line_width = 3.0

    if render_option_json != '':
        opt.load_from_json(render_option_json)
        print(f'load render option from {render_option_json}')
    if view_option_json != '':
        view_param = o3d.io.read_pinhole_camera_parameters(view_option_json)
        ctr.convert_from_pinhole_camera_parameters(view_param)
        print(f'load view option from {view_option_json}')

    vis.update_renderer()
    if capture_screen_image != '':
        vis.capture_screen_image(capture_screen_image, do_render=True)
        print(f'save screen image to {capture_screen_image}')
        print('*' * 30)
        return
    vis.run()
    if save_view_option_to != '':
        view_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(save_view_option_to, view_param)
        print(f'save view option to {save_view_option_to}')
    vis.destroy_window()
    print('*' * 30)


def bbox2o3d(bboxes: List, color=None):
    line_sets = []
    if color is None:
        color = [0, 0, 0]
    for bbox in bboxes:
        lines_box = np.array([[0, 1], [0, 3], [0, 4], [1, 2], [1, 5], [2, 3], [2, 6],
                              [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]])
        colors = np.array([color]).repeat(repeats=len(lines_box), axis=0)
        line_set = o3d.geometry.LineSet()
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        if len(bbox) == 7:
            bbox = bbox2corner(bbox)
        else:
            assert len(bbox) == 8 and len(bbox[0]) == 3
        line_set.points = o3d.utility.Vector3dVector(bbox)
        line_sets.append(line_set)
    return line_sets


def bbox2corner(bbox: Union[np.ndarray, torch.Tensor, list]):
    """
        4 ---------- 5
      / |          / |   ^ z
    0 ---------- 1   |   |   / y
    |   7 -------|-- 6   | /
    | /          | /     *------> x
    3 ---------- 2
    """
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().numpy()
    elif isinstance(bbox, list):
        bbox = np.array(bbox)
    elif isinstance(bbox, np.ndarray):
        pass
    else:
        raise TypeError('The type of the bbox must be one of np.ndarray, torch.Tensor or list.')
    # [x, y, z, l, w, h, a] => [[x1, y1, z1], [x2, y2, z2], ..., [x8, y8, z8]]
    x, y, z, l, w, h, a = bbox
    c0 = [x - l / 2, y - w / 2, z + h]
    c1 = [x + l / 2, y - w / 2, z + h]
    c2 = [x + l / 2, y - w / 2, z]
    c3 = [x - l / 2, y - w / 2, z]
    c4 = [x - l / 2, y + w / 2, z + h]
    c5 = [x + l / 2, y + w / 2, z + h]
    c6 = [x + l / 2, y + w / 2, z]
    c7 = [x - l / 2, y + w / 2, z]
    corners = np.array([c0, c1, c2, c3, c4, c5, c6, c7])  # (8, 3)
    center_xy = np.array([x, y])

    s, c = math.sin(a), math.cos(a)
    yaw_mat = torch.tensor([[c, -s], [s, c]])  # (2, 2)
    corners[:, :2] = (yaw_mat @ (corners[:, :2].T - center_xy[:, np.newaxis]) + center_xy[:, np.newaxis]).T

    return corners


def bbox2cylinder(bboxes: List, color=None, radius=0.05, resolution=10):
    """
        4 ---------- 5
      / |          / |   ^ z
    0 ---------- 1   |   |   / y
    |   7 -------|-- 6   | /
    | /          | /     *------> x
    3 ---------- 2
    """
    line_sets = []
    if color is None:
        color = [0, 0, 0]
    for bbox in bboxes:
        l, w, h = bbox[3:6]
        a = bbox[-1]
        if len(bbox) == 7:
            bbox = bbox2corner(bbox)
        else:
            assert len(bbox) == 8 and len(bbox[0]) == 3

        R_x_direction, R_y_direction, R_z_direction = np.eye(4), np.eye(4), np.eye(4)
        R_x_direction[:3, :3] = np.array([[math.cos(pi / 2), 0, math.sin(pi / 2)], [0, 1, 0], [-math.sin(pi / 2), 0, math.cos(pi / 2)]])
        R_y_direction[:3, :3] = np.array([[1, 0, 0], [0, math.cos(pi / 2), -math.sin(pi / 2)], [0, math.sin(pi / 2), math.cos(pi / 2)]])
        R_x_direction[0, -1] = l / 2
        R_y_direction[1, -1] = w / 2
        R_z_direction[2, -1] = h / 2

        mesh_cylinder01 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=l, resolution=resolution).transform(R_x_direction)
        mesh_cylinder45 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=l, resolution=resolution).transform(R_x_direction)
        mesh_cylinder32 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=l, resolution=resolution).transform(R_x_direction)
        mesh_cylinder76 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=l, resolution=resolution).transform(R_x_direction)
        mesh_cylinder04 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=w, resolution=resolution).transform(R_y_direction)
        mesh_cylinder15 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=w, resolution=resolution).transform(R_y_direction)
        mesh_cylinder26 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=w, resolution=resolution).transform(R_y_direction)
        mesh_cylinder37 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=w, resolution=resolution).transform(R_y_direction)
        mesh_cylinder30 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=resolution).transform(R_z_direction)
        mesh_cylinder21 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=resolution).transform(R_z_direction)
        mesh_cylinder65 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=resolution).transform(R_z_direction)
        mesh_cylinder74 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=resolution).transform(R_z_direction)

        init_R = np.array([[math.cos(a), -math.sin(a), 0, 0], [math.sin(a), math.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        R_01, R_45, R_32, R_76, R_04, R_15, R_26, R_37, R_30, R_21, R_65, R_74 = np.repeat(init_R[np.newaxis, :, :], axis=0, repeats=12)
        R_01[:3, -1] = bbox[0]
        R_45[:3, -1] = bbox[4]
        R_32[:3, -1] = bbox[3]
        R_76[:3, -1] = bbox[7]
        R_04[:3, -1] = bbox[0]
        R_15[:3, -1] = bbox[1]
        R_26[:3, -1] = bbox[2]
        R_37[:3, -1] = bbox[3]
        R_30[:3, -1] = bbox[3]
        R_21[:3, -1] = bbox[2]
        R_65[:3, -1] = bbox[6]
        R_74[:3, -1] = bbox[7]

        mesh_cylinder01 = mesh_cylinder01.transform(R_01)
        mesh_cylinder45 = mesh_cylinder45.transform(R_45)
        mesh_cylinder32 = mesh_cylinder32.transform(R_32)
        mesh_cylinder76 = mesh_cylinder76.transform(R_76)
        mesh_cylinder04 = mesh_cylinder04.transform(R_04)
        mesh_cylinder15 = mesh_cylinder15.transform(R_15)
        mesh_cylinder26 = mesh_cylinder26.transform(R_26)
        mesh_cylinder37 = mesh_cylinder37.transform(R_37)
        mesh_cylinder30 = mesh_cylinder30.transform(R_30)
        mesh_cylinder21 = mesh_cylinder21.transform(R_21)
        mesh_cylinder65 = mesh_cylinder65.transform(R_65)
        mesh_cylinder74 = mesh_cylinder74.transform(R_74)

        for mesh in [mesh_cylinder01, mesh_cylinder45, mesh_cylinder32, mesh_cylinder76, mesh_cylinder04, mesh_cylinder15,
                     mesh_cylinder26, mesh_cylinder37, mesh_cylinder30, mesh_cylinder21, mesh_cylinder65, mesh_cylinder74]:
            mesh.paint_uniform_color(color)
        line_sets += [mesh_cylinder01, mesh_cylinder45, mesh_cylinder32, mesh_cylinder76, mesh_cylinder04, mesh_cylinder15,
                      mesh_cylinder26, mesh_cylinder37, mesh_cylinder30, mesh_cylinder21, mesh_cylinder65, mesh_cylinder74]
        for corner_point in bbox:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.2 * radius, resolution=resolution)
            mesh_sphere.translate(corner_point)
            if color is not None:
                mesh_sphere.paint_uniform_color(color)
            line_sets.append(mesh_sphere)
    return line_sets

