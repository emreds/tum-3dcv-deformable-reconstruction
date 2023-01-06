import os
import json
import shutil
import numpy as np

from argparse import ArgumentParser
from os import path


CAM_TYPE = "unknown"


def save_json(data, out):
    with open(out, 'w') as f:
        json.dump(data, f)


def convert_intrinsics(intrinsics):
    cam_params = {}
    intrinsics = np.loadtxt(intrinsics)
    cam_params["cam_K"] = intrinsics.flatten().tolist()
    cam_params["depth_scale"] = 1

    return cam_params


def convert_cam_poses(poses_root, cam_params, out):
    scene = {}
    for file in os.listdir(poses_root):
        if not path.isfile(file) and not file.endswith('.txt'):
            print(f"Warning: Converting cam poses, unexpected file '{file}'")
        index = int(file[:-4])
        pose = np.loadtxt(path.join(poses_root, file))
        R = pose[0:3, 0:3]
        t = pose[0:3, 3]
        scene[index] = cam_params.copy()
        scene[index]["cam_R_w2c"] = R.flatten().tolist()
        scene[index]["cam_t_w2c"] = t.tolist()

    save_json(scene, path.join(out, "scene_camera.json"))
    return scene


def convert_object_poses(obj_poses_root, cam_poses, out):

    obj_poses = {}
    for i, file in enumerate(os.listdir(obj_poses_root), 1):
        if not path.isfile(file) and not file.endswith('.txt'):
            print(f"Warning: Converting object poses, unexpected file '{file}'")
        obj_poses[i] = np.loadtxt(path.join(obj_poses_root, file))

    scene_gt = dict.fromkeys(cam_poses.keys(), [])
    for index, pose in cam_poses.items():
        cam_pose = np.identity(4)
        cam_pose[:3, :3] = np.array(pose["cam_R_w2c"], dtype=float).reshape(3, 3)
        cam_pose[:3, 3] = pose["cam_t_w2c"]
        data_list = []
        for i, obj in obj_poses.items():
            cam_obj_pose = cam_pose * obj
            d = {"obj_id": i,
                 "cam_R_m2c": cam_obj_pose[:3, :3].flatten().tolist(),
                 "cam_t_m2c": cam_obj_pose[:3, 3].tolist()}
            data_list.append(d)
        scene_gt[index] = data_list

    save_json(scene_gt, path.join(out, "scene_gt.json"))


def compute_gt_info(masks, masks_visib, out):
    assert len(os.listdir(masks) == len(os.listdir(masks_visib)))
    
    for file in os.listdir(masks): 
        mask = np.load(path.join(masks, file))
        visib = np.load(path.join(masks_visib, file))
        
        obj_bb = []
        visib_bb = []


def transform_dataset(dataset, destination, train=True, scene_id=1):
    if path.exists(destination) and path.isdir(destination):
        if os.listdir(destination):
            print(f"Warning: The directory {destination} is not empty. If you "
                  "continue the content will be overwritten")
        else:
            print(f"Info: The directory {destination} exists, but is empty.")
    else:
        os.makedirs(destination)

    if path.exists(path.join(dataset, "intrinsics.txt")):
        cam_params = convert_intrinsics(path.join(dataset, "intrinsics.txt"))
        save_json(cam_params, path.join(destination, f"camera_{CAM_TYPE}.json"))

    else:
        print(f"Warning: Missing 'intrinsics.txt' in '{dataset}'")

    data_type = "train" if train else "test"
    data_dest = path.join(destination, f"{data_type}_{CAM_TYPE}", str(scene_id))
    if path.exists(data_dest):
        shutil.rmtree(data_dest)
    os.makedirs(data_dest)

    shutil.copytree(path.join(dataset, "rgb"), path.join(data_dest, "rgb"))
    shutil.copytree(path.join(dataset, "depth"), path.join(data_dest, "depth"))
    shutil.copytree(path.join(dataset, "instance"), path.join(data_dest, "mask"))
    # shutil.copytree(path.join(dataset, "instance"), path.join(data_dest, "mask_visib")) # TODO: Need this data

    cam_poses = convert_cam_poses(path.join(dataset, "camera_pose"), cam_params, data_dest)
    convert_object_poses(path.join(dataset, "obj_pose"), cam_poses, data_dest)


def main():
    argparse = ArgumentParser()
    argparse.add_argument("--dataset", required=True)
    argparse.add_argument("--destination")

    args = argparse.parse_args()
    transform_dataset(args.dataset,
                      f"{args.destination}_{path.basename(args.dataset)}")


if __name__ == "__main__":
    main()
