import os
import pickle
import numpy as np
import torch


def load_points(points_file: str, transform=True, scene_dataset="replica"):
    r"""
    Helper method to load points data from files stored on disk and transform if necessary
    :param points_file: path to files containing points data
    :param transform: transform coordinate systems of loaded points for use in Habitat or not
    :param scene_dataset: name of scenes dataset ("replica", "mp3d", etc.)
    :return: points in transformed coordinate system for use with Habitat
    """
    points_data = np.loadtxt(points_file, delimiter="\t")
    if transform:
        if scene_dataset == "replica":
            points = list(zip(
                points_data[:, 1],
                points_data[:, 3] - 1.5528907,
                -points_data[:, 2])
            )
        elif scene_dataset == "mp3d":
            points = list(zip(
                points_data[:, 1],
                points_data[:, 3] - 1.5,
                -points_data[:, 2])
            )
        else:
            raise NotImplementedError
    else:
        points = list(zip(
            points_data[:, 1],
            points_data[:, 2],
            points_data[:, 3])
        )
    points_index = points_data[:, 0].astype(int)
    points_dict = dict(zip(points_index, points))
    assert list(points_index) == list(range(len(points)))
    return points_dict, points


def load_points_data(parent_folder, graph_file, transform=True, scene_dataset="replica"):
    r"""
    Main method to load points data from files stored on disk and transform if necessary
    :param parent_folder: parent folder containing files with points data
    :param graph_file: files containing connectivity of points per scene
    :param transform: transform coordinate systems of loaded points for use in Habitat or not
    :param scene_dataset: name of scenes dataset ("replica", "mp3d", etc.)
    :return: 1. points in transformed coordinate system for use with Habitat
             2. graph object containing information about the connectivity of points in a scene
    """
    points_file = os.path.join(parent_folder, 'points.txt')
    graph_file = os.path.join(parent_folder, graph_file)

    _, points = load_points(points_file, transform=transform, scene_dataset=scene_dataset)
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)
