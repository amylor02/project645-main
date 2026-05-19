import numpy as np
import cv2 as cv
from cv2 import VideoCapture
import matplotlib.pyplot as plt
from collections import Counter

import transforms3d.euler as euler
import transforms3d.quaternions as quat

from pylab import *
from PIL import Image
import os
import getopt

import json  # For formatted printing

import read_bvh_hierarchy

import rotation2xyz as helper
from rotation2xyz import *


def get_pos_joints_index(raw_frame_data, non_end_bones, skeleton):
    pos_dic = helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    keys = OrderedDict()
    i = 0
    for joint in pos_dic.keys():
        keys[joint] = i
        i = i + 1
    return keys


def parse_frames(bvh_filename):
    bvh_file = open(bvh_filename, "r")
    lines = bvh_file.readlines()
    bvh_file.close()
    l = [lines.index(i) for i in lines if "MOTION" in i]
    data_start = l[0]

    # data_start = lines.index('MOTION\n')
    first_frame = data_start + 3

    num_params = len(lines[first_frame].split(" "))
    num_frames = len(lines) - first_frame

    data = np.zeros((num_frames, num_params))

    for i in range(num_frames):
        line = lines[first_frame + i].split(" ")
        line = line[0 : len(line)]

        line_f = [float(e) for e in line]

        data[i, :] = line_f

    return data


standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)
sample_data = parse_frames(standard_bvh_file)
joint_index = get_pos_joints_index(sample_data[0], non_end_bones, skeleton)


def get_frame_format_string(bvh_filename):
    bvh_file = open(bvh_filename, "r")

    lines = bvh_file.readlines()
    bvh_file.close()
    l = [lines.index(i) for i in lines if "MOTION" in i]
    data_end = l[0]
    # data_end = lines.index('MOTION\n')
    data_end = data_end + 2
    return lines[0 : data_end + 1]


def get_min_foot_and_hip_center(bvh_data):
    print(bvh_data.shape)
    lowest_points = []
    hip_index = joint_index["hip"]
    left_foot_index = joint_index["lFoot"]
    left_nub_index = joint_index["lFoot_Nub"]
    right_foot_index = joint_index["rFoot"]
    right_nub_index = joint_index["rFoot_Nub"]

    for i in range(bvh_data.shape[0]):
        frame = bvh_data[i, :]
        # print 'hi1'
        foot_heights = [
            frame[left_foot_index * 3 + 1],
            frame[left_nub_index * 3 + 1],
            frame[right_foot_index * 3 + 1],
            frame[right_nub_index * 3 + 1],
        ]
        lowest_point = min(foot_heights) + frame[hip_index * 3 + 1]
        lowest_points.append(lowest_point)

        # print lowest_point
    lowest_points = sort(lowest_points)
    num_frames = bvh_data.shape[0]
    quarter_length = int(num_frames / 4)
    end = 3 * quarter_length
    overall_lowest = mean(lowest_points[quarter_length:end])

    return overall_lowest


def sanity():
    for i in range(4):
        print("hi")


def get_motion_center(bvh_data):
    center = np.zeros(3)
    for frame in bvh_data:
        center = center + frame[0:3]
    center = center / bvh_data.shape[0]
    return center


def augment_train_frame_data(train_frame_data, T, axisR):
    hip_index = joint_index["hip"]
    hip_pos = train_frame_data[hip_index * 3 : hip_index * 3 + 3]

    # Translate joints based on hip position
    for i in range(int(len(train_frame_data) / 3)):
        if i != hip_index:
            train_frame_data[i * 3 : i * 3 + 3] = (
                train_frame_data[i * 3 : i * 3 + 3] + hip_pos
            )

    mat_r_augment = euler.axangle2mat(axisR[0:3], axisR[3])
    n = int(len(train_frame_data) / 3)
    for i in range(n):
        raw_data = train_frame_data[i * 3 : i * 3 + 3]
        new_data = np.dot(mat_r_augment, raw_data) + T
        train_frame_data[i * 3 : i * 3 + 3] = new_data

    hip_pos = train_frame_data[hip_index * 3 : hip_index * 3 + 3]

    for i in range(int(len(train_frame_data) / 3)):
        if i != hip_index:
            train_frame_data[i * 3 : i * 3 + 3] = (
                train_frame_data[i * 3 : i * 3 + 3] - hip_pos
            )

    return train_frame_data


def augment_train_data(train_data, T, axisR):
    result = list(
        map(lambda frame: augment_train_frame_data(frame, T, axisR), train_data)
    )
    return np.array(result)


# input a vector of data, with the first three data as translation and the rest the euler rotation
# output a vector of data, with the first three data as translation not changed and the rest to quaternions.
# note: the input data are in z, x, y sequence
def get_one_frame_training_format_data(raw_frame_data, non_end_bones, skeleton):
    pos_dic = helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    new_data = np.zeros(len(pos_dic.keys()) * 3)
    i = 0
    hip_pos = pos_dic["hip"]
    # print hip_pos

    for joint in pos_dic.keys():
        if joint == "hip":
            new_data[i * 3 : i * 3 + 3] = pos_dic[joint].reshape(3)
        else:
            new_data[i * 3 : i * 3 + 3] = pos_dic[joint].reshape(3) - hip_pos.reshape(3)
        i = i + 1
    # print new_data
    new_data = new_data * 0.01
    return new_data


def get_training_format_data(raw_data, non_end_bones, skeleton):
    new_data = []
    for frame in raw_data:
        new_frame = get_one_frame_training_format_data(frame, non_end_bones, skeleton)
        new_data = new_data + [new_frame]
    return np.array(new_data)


def get_weight_dict(skeleton):
    weight_dict = []
    for joint in skeleton:
        parent_number = 0.0
        j = joint
        while skeleton[joint]["parent"] != None:
            parent_number = parent_number + 1
            joint = skeleton[joint]["parent"]
        weight = pow(math.e, -parent_number / 5.0)
        weight_dict = weight_dict + [(j, weight)]
    return weight_dict


def get_train_data(bvh_filename):

    data = parse_frames(bvh_filename)
    train_data = get_training_format_data(data, non_end_bones, skeleton)
    center = get_motion_center(train_data)  # get the avg position of the hip
    center[1] = 0.0  # don't center the height

    new_train_data = augment_train_data(train_data, -center, [0, 1, 0, 0.0])

    return new_train_data


#### for euler ####
def get_one_frame_euler_training_format_data(
    raw_frame_data, translation_weight=weight_translation
):
    train_frame = np.array(raw_frame_data, dtype=np.float64, copy=True)
    train_frame[0:3] = train_frame[0:3] * translation_weight
    return train_frame


def get_euler_training_format_data(raw_data, translation_weight=weight_translation):
    new_data = []
    for frame in raw_data:
        new_data = new_data + [
            get_one_frame_euler_training_format_data(frame, translation_weight)
        ]
    return np.array(new_data)


def get_euler_train_data(bvh_filename, translation_weight=weight_translation):
    raw_data = parse_frames(bvh_filename)
    return get_euler_training_format_data(raw_data, translation_weight)


def wrap_angles_degrees(angle_values):
    wrapped_values = np.array(angle_values, dtype=np.float64, copy=True)
    return ((wrapped_values + 180.0) % 360.0) - 180.0


def wrap_euler_train_frame_data(train_frame_data):
    wrapped_frame = np.array(train_frame_data, dtype=np.float64, copy=True)
    wrapped_frame[3:] = wrap_angles_degrees(wrapped_frame[3:])
    return wrapped_frame


def wrap_euler_train_data(train_data):
    wrapped_data = np.array(train_data, dtype=np.float64, copy=True)
    wrapped_data[:, 3:] = wrap_angles_degrees(wrapped_data[:, 3:])
    return wrapped_data


def recenter_euler_root_translation(train_data, center_y=False):
    recentered_data = np.array(train_data, dtype=np.float64, copy=True)
    root_center = np.mean(recentered_data[:, 0:3], axis=0)
    if center_y == False:
        root_center[1] = 0.0
    recentered_data[:, 0:3] = recentered_data[:, 0:3] - root_center
    return recentered_data


def augment_euler_train_frame_data(train_frame_data, T, axisR):
    augmented_frame = wrap_euler_train_frame_data(train_frame_data)

    mat_r_augment = euler.axangle2mat(axisR[0:3], axisR[3])
    augmented_frame[0:3] = np.dot(mat_r_augment, augmented_frame[0:3]) + T

    root_angles = np.array(
        [augmented_frame[5], augmented_frame[4], augmented_frame[3]],
        dtype=np.float64,
    )
    root_rotation_matrix = helper.eulerAnglesToRotationMatrix_hip(root_angles)
    augmented_root_rotation = np.dot(mat_r_augment, root_rotation_matrix)
    augmented_root_angles = helper.rotationMatrixToEulerAngles(
        augmented_root_rotation
    ) * (180.0 / np.pi)

    augmented_frame[3] = augmented_root_angles[2]
    augmented_frame[4] = augmented_root_angles[1]
    augmented_frame[5] = augmented_root_angles[0]
    augmented_frame[3:] = wrap_angles_degrees(augmented_frame[3:])

    return augmented_frame


def augment_euler_train_data(train_data, T, axisR):
    result = list(
        map(lambda frame: augment_euler_train_frame_data(frame, T, axisR), train_data)
    )
    return np.array(result)


def write_euler_traindata_to_bvh(
    bvh_filename,
    train_data,
    translation_weight=weight_translation,
    format_filename=standard_bvh_file,
):
    output_data = np.array(train_data, dtype=np.float64, copy=True)
    output_data[:, 0:3] = output_data[:, 0:3] / translation_weight
    write_frames(format_filename, bvh_filename, output_data)


#### for euler end ####
#### for quaternion ####
def normalize_quaternion_unit(rotation_quaternion, epsilon=1e-8):
    quaternion = np.array(rotation_quaternion, dtype=np.float64, copy=True)
    norm = np.linalg.norm(quaternion)
    if norm < epsilon:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quaternion = quaternion / norm
    return quaternion


def canonicalize_quaternion(rotation_quaternion):
    quaternion = np.array(rotation_quaternion, dtype=np.float64, copy=True)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def normalize_quaternion(rotation_quaternion, epsilon=1e-8):
    quaternion = normalize_quaternion_unit(rotation_quaternion, epsilon)
    return canonicalize_quaternion(quaternion)


def enforce_quaternion_sequence_continuity(
    train_data, translation_size=3, quaternion_size=4, epsilon=1e-8
):
    continuous_data = np.array(train_data, dtype=np.float64, copy=True)
    if continuous_data.shape[0] == 0:
        return continuous_data

    quaternion_count = int((continuous_data.shape[1] - translation_size) / quaternion_size)
    for quaternion_index in range(quaternion_count):
        start_index = translation_size + quaternion_index * quaternion_size
        end_index = start_index + quaternion_size

        previous_quaternion = normalize_quaternion(
            continuous_data[0, start_index:end_index], epsilon
        )
        continuous_data[0, start_index:end_index] = previous_quaternion

        for frame_index in range(1, continuous_data.shape[0]):
            current_quaternion = normalize_quaternion_unit(
                continuous_data[frame_index, start_index:end_index], epsilon
            )
            if np.dot(previous_quaternion, current_quaternion) < 0.0:
                current_quaternion = -current_quaternion
            continuous_data[frame_index, start_index:end_index] = current_quaternion
            previous_quaternion = current_quaternion

    return continuous_data


def augment_quaternion_train_frame_data(train_frame_data, T, axisR):
    augmented_frame = np.array(train_frame_data, dtype=np.float64, copy=True)

    mat_r_augment = euler.axangle2mat(axisR[0:3], axisR[3])
    augmented_frame[0:3] = np.dot(mat_r_augment, augmented_frame[0:3]) + T

    augment_quaternion = normalize_quaternion(quat.axangle2quat(axisR[0:3], axisR[3]))
    root_quaternion = normalize_quaternion(augmented_frame[3:7])
    augmented_frame[3:7] = normalize_quaternion(
        quat.qmult(augment_quaternion, root_quaternion)
    )

    quaternion_count = int((len(augmented_frame) - 3) / 4)
    for quaternion_index in range(quaternion_count):
        start_index = 3 + quaternion_index * 4
        end_index = start_index + 4
        augmented_frame[start_index:end_index] = normalize_quaternion(
            augmented_frame[start_index:end_index]
        )

    return augmented_frame


def augment_quaternion_train_data(train_data, T, axisR):
    result = list(
        map(
            lambda frame: augment_quaternion_train_frame_data(frame, T, axisR),
            train_data,
        )
    )
    return enforce_quaternion_sequence_continuity(np.array(result))


def get_one_frame_quaternion_training_format_data(
    raw_frame_data, non_end_bones, translation_weight=weight_translation
):
    frame_data = np.array(raw_frame_data, dtype=np.float64, copy=True)
    quaternion_count = len(non_end_bones) + 1
    new_data = np.zeros(3 + quaternion_count * 4, dtype=np.float64)
    new_data[0:3] = frame_data[0:3] * translation_weight

    root_x_rotation = frame_data[5]
    root_y_rotation = frame_data[4]
    root_z_rotation = frame_data[3]
    root_rotation_matrix = helper.eulerAnglesToRotationMatrix_hip(
        [root_x_rotation, root_y_rotation, root_z_rotation]
    )
    new_data[3:7] = normalize_quaternion(quat.mat2quat(root_rotation_matrix))

    for bone_index, bone_name in enumerate(non_end_bones):
        raw_offset = 6 + bone_index * 3
        z_rotation = frame_data[raw_offset]
        x_rotation = frame_data[raw_offset + 1]
        y_rotation = frame_data[raw_offset + 2]
        rotation_matrix = helper.eulerAnglesToRotationMatrix(
            [x_rotation, y_rotation, z_rotation]
        )
        quat_offset = 7 + bone_index * 4
        new_data[quat_offset : quat_offset + 4] = normalize_quaternion(
            quat.mat2quat(rotation_matrix)
        )

    return new_data


def get_quaternion_training_format_data(
    raw_data, non_end_bones, translation_weight=weight_translation
):
    new_data = []
    for frame in raw_data:
        new_data = new_data + [
            get_one_frame_quaternion_training_format_data(
                frame, non_end_bones, translation_weight
            )
        ]
    return enforce_quaternion_sequence_continuity(np.array(new_data))


def get_quaternion_train_data(
    bvh_filename, non_end_bones, translation_weight=weight_translation
):
    raw_data = parse_frames(bvh_filename)
    return get_quaternion_training_format_data(
        raw_data, non_end_bones, translation_weight
    )


def get_frame_from_quaternion_training_format_data(
    train_frame_data, non_end_bones, translation_weight=weight_translation
):
    frame_data = np.zeros(6 + len(non_end_bones) * 3, dtype=np.float64)
    frame_data[0:3] = train_frame_data[0:3] / translation_weight

    root_quaternion = normalize_quaternion(train_frame_data[3:7])
    root_rotation_matrix = quat.quat2mat(root_quaternion)
    root_angles = helper.rotationMatrixToEulerAngles(root_rotation_matrix) * (
        180.0 / np.pi
    )
    frame_data[3] = root_angles[2]
    frame_data[4] = root_angles[1]
    frame_data[5] = root_angles[0]

    for bone_index, bone_name in enumerate(non_end_bones):
        quat_offset = 7 + bone_index * 4
        bone_quaternion = normalize_quaternion(
            train_frame_data[quat_offset : quat_offset + 4]
        )
        bone_rotation_matrix = quat.quat2mat(bone_quaternion)
        bone_angles = np.array(euler.mat2euler(bone_rotation_matrix, "syxz")) * (
            180.0 / np.pi
        )
        raw_offset = 6 + bone_index * 3
        frame_data[raw_offset] = bone_angles[2]
        frame_data[raw_offset + 1] = bone_angles[1]
        frame_data[raw_offset + 2] = bone_angles[0]

    return frame_data


def write_quaternion_traindata_to_bvh(
    bvh_filename,
    train_data,
    non_end_bones,
    translation_weight=weight_translation,
    format_filename=standard_bvh_file,
):
    output_data = []
    for frame in train_data:
        output_data = output_data + [
            get_frame_from_quaternion_training_format_data(
                frame, non_end_bones, translation_weight
            )
        ]
    write_frames(format_filename, bvh_filename, np.array(output_data))


#### for quaternion end ####
def write_frames(format_filename, out_filename, data):

    format_lines = get_frame_format_string(format_filename)

    num_frames = data.shape[0]
    format_lines[len(format_lines) - 2] = "Frames:\t" + str(num_frames) + "\n"

    bvh_file = open(out_filename, "w")
    bvh_file.writelines(format_lines)
    bvh_data_str = vectors2string(data)
    bvh_file.write(bvh_data_str)
    bvh_file.close()


def regularize_angle(a):

    if abs(a) > 180:
        remainder = a % 180
        print("hi")
    else:
        return a

    new_ang = -(sign(a) * 180 - remainder)

    return new_ang


def write_xyz_to_bvh(
    xyz_motion, skeleton, non_end_bones, format_filename, output_filename
):
    bvh_vec_length = len(non_end_bones) * 3 + 6

    out_data = np.zeros([len(xyz_motion), bvh_vec_length])
    for i in range(1, len(xyz_motion)):
        positions = xyz_motion[i]
        rotation_matrices, rotation_angles = helper.xyz_to_rotations_debug(
            skeleton, positions
        )
        new_motion1 = helper.rotation_dic_to_vec(
            rotation_angles, non_end_bones, positions
        )

        new_motion = np.array([round(a, 6) for a in new_motion1])
        new_motion[0:3] = new_motion1[0:3]

        out_data[i, :] = np.transpose(new_motion[:, np.newaxis])

    write_frames(format_filename, output_filename, out_data)


def write_traindata_to_bvh(bvh_filename, train_data):
    seq_length = train_data.shape[0]
    xyz_motion = []
    format_filename = standard_bvh_file
    for i in range(seq_length):
        data = np.array([round(a, 6) for a in train_data[i]])
        position = data_vec_to_position_dic(data, skeleton)
        xyz_motion.append(position)

    write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, bvh_filename)


def data_vec_to_position_dic(data, skeleton):
    data = data * 100
    hip_pos = data[joint_index["hip"] * 3 : joint_index["hip"] * 3 + 3]
    positions = {}
    for joint in joint_index:
        positions[joint] = data[joint_index[joint] * 3 : joint_index[joint] * 3 + 3]

    for joint in positions.keys():
        if joint == "hip":
            positions[joint] = positions[joint]
        else:
            positions[joint] = positions[joint] + hip_pos

    return positions


def get_pos_dic(frame, joint_index):
    positions = {}
    for key in joint_index.keys():
        positions[key] = frame[joint_index[key] * 3 : joint_index[key] * 3 + 3]
    return positions


#######################################################
#################### Write train_data to bvh###########


def vector2string(data):
    s = " ".join(map(str, data))

    return s


def vectors2string(data):
    s = "\n".join(map(vector2string, data))

    return s


def get_child_list(skeleton, joint):
    child = []
    for j in skeleton:
        parent = skeleton[j]["parent"]
        if parent == joint:
            child.append(j)
    return child


def get_norm(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def get_regularized_positions(positions):

    org_positions = positions
    new_positions = regularize_bones(org_positions, positions, skeleton, "hip")
    return new_positions


def regularize_bones(original_positions, new_positions, skeleton, joint):
    children = get_child_list(skeleton, joint)
    for child in children:
        offsets = skeleton[child]["offsets"]
        length = get_norm(offsets)
        direction = original_positions[child] - original_positions[joint]
        # print child
        new_vector = direction * length / get_norm(direction)
        # print child
        # print length, get_norm(direction)
        # print new_positions[child]
        new_positions[child] = new_positions[joint] + new_vector
        # print new_positions[child]
        new_positions = regularize_bones(
            original_positions, new_positions, skeleton, child
        )
    return new_positions


def get_regularized_train_data(one_frame_train_data):

    one_frame_train_data = one_frame_train_data * 100.0
    positions = {}
    for joint in joint_index:
        positions[joint] = one_frame_train_data[
            joint_index[joint] * 3 : joint_index[joint] * 3 + 3
        ]

    # print joint_index['hip']
    hip_pos = one_frame_train_data[joint_index["hip"] * 3 : joint_index["hip"] * 3 + 3]

    for joint in positions.keys():
        if joint == "hip":
            positions[joint] = positions[joint]
        else:
            positions[joint] = positions[joint] + hip_pos

    new_pos = get_regularized_positions(positions)

    new_data = np.zeros(one_frame_train_data.shape)
    i = 0
    for joint in new_pos.keys():
        if joint != "hip":
            new_data[i * 3 : i * 3 + 3] = new_pos[joint] - new_pos["hip"]
        else:
            new_data[i * 3 : i * 3 + 3] = new_pos[joint]
        i = i + 1
    new_data = new_data * 0.01
    return new_data


def check_length(one_frame_train_data):
    one_frame_train_data = one_frame_train_data * 100.0
    positions = {}
    for joint in joint_index:
        positions[joint] = one_frame_train_data[
            joint_index[joint] * 3 : joint_index[joint] * 3 + 3
        ]

    # print joint_index['hip']
    hip_pos = one_frame_train_data[joint_index["hip"] * 3 : joint_index["hip"] * 3 + 3]

    for joint in positions.keys():
        if joint == "hip":
            positions[joint] = positions[joint]
        else:
            positions[joint] = positions[joint] + hip_pos

    for joint in positions.keys():
        if skeleton[joint]["parent"] != None:
            p1 = positions[joint]
            p2 = positions[skeleton[joint]["parent"]]
            b = p2 - p1
            # print get_norm(b), get_norm(skeleton[joint]['offsets'])
