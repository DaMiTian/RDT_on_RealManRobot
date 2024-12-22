#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
import torch
from PIL import Image as PImage
import cv2

from scripts.agilex_model import create_model
from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs
import pdb

# sys.path.append("./")

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

observation_window = None

lang_embeddings = None

# debug
preload_images = None


class RM_controller:
    def __init__(self):
        self.arm_controller = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm_controller.rm_create_robot_arm("192.168.1.18", 8080)
    def get_qpos(self):
        succ, gripper_state = self.arm_controller.rm_get_gripper_state()
        gripper = gripper_state["actpos"]/1000
        qpos = self.arm_controller.rm_get_current_arm_state()[1]['pose']
        qpos.append(gripper)
        print(qpos)
        return qpos

    def move(self, d_action):
        action1 = [d_action[0], d_action[1], d_action[2], d_action[3], d_action[4], d_action[5]]
        action1 = [float(num) for num in action1]
        # 将科学计数法表示  的数字转换为小数形式并保留指定的位数
        action1 = [round(num, 4) if idx < 3 else round(num, 3) for idx, num in enumerate(action1)]
        print("delta_move:",action1)
        # 获取位置
        position = self.arm_controller.rm_get_current_arm_state()[1]['pose']
        # pdb.set_trace()
        # position[0] = position[0] + min(action1[0],0.05)
        # position[1] = position[1] + min(action1[1],0.05)
        # position[2] = position[2] + min(action1[2],0.05)
        # position[3] = position[3] + min(action1[3],0.02)
        # position[4] = position[4] + min(action1[4],0.02)
        # position[5] = position[5] + min(action1[5],0.02)
        position[0] = position[0] + action1[0]
        position[1] = position[1] + action1[1]
        position[2] = position[2] + action1[2]
        position[3] = position[3] + action1[3]
        position[4] = position[4] + action1[4]
        position[5] = position[5] + action1[5]
        position = [float(num) for num in position]
        print(position)
        print("gripper",d_action[6])
        if d_action[6]<0.9:
            self.arm_controller.rm_set_gripper_pick_on(300, 500, True, 1)
        else:
            self.arm_controller.rm_set_gripper_release(300, True, 1)
        # position = [round(num, 5) if idx < 3 else round(num, 3) for idx, num in enumerate(position)]
        
        self.arm_controller.rm_movej_p(position, 30, 0, 0, 1)


class Img_controller:
    def __init__(self):
        self.front = cv2.VideoCapture(0)
        self.right = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.right.start(config)
        self.t = 0
    def get_img(self):       
        ret, front_image = self.front.read()
        front_image =cv2.resize(front_image, (640, 480))
    
        front_image = PImage.fromarray(cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB))
        # # 对PIL图像进行左右翻转
        front_image = front_image.transpose(PImage.FLIP_LEFT_RIGHT)
        front_image = np.array(front_image)
        front_image = front_image[..., [2, 1, 0]]  # 将BGR转为RGB
        
        frames = self.right.wait_for_frames()
        color_frame = frames.get_color_frame()
        # 转换为numpy数组
        right_image = np.asanyarray(color_frame.get_data())
        right_image = right_image[..., [2, 1, 0]]
        cv2.imwrite(f"./demo/right_{self.t}.jpeg",right_image)
        cv2.imwrite(f"./demo/front_{self.t}.jpeg",front_image)
        self.t += 1 
        return front_image, right_image

class Controller:
    def __init__(self):
        self.right_arm_controller = RM_controller()
        self.img_controller = Img_controller()

# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 7,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# Get the observation from the ROS topic
def get_RM_observation(args, controller):
    puppet_arm_right = controller.right_arm_controller.get_qpos()
    img_front, img_right = controller.img_controller.get_img()
    return img_front, img_right, puppet_arm_right


# Update the observation window buffer
def update_observation_window(args, config, controller):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
            }
        )

    img_front, img_right, puppet_arm_right = get_RM_observation(args, controller)
    img_front = jpeg_mapping(img_front)
    img_left = None
    img_right = jpeg_mapping(img_right)

    qpos = np.array(puppet_arm_right)
    qpos = torch.from_numpy(qpos).float().cuda()
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_front,
                    config["camera_names"][1]: img_right,
                    config["camera_names"][2]: img_left,
                },
        }
    )


# RDT inference
def inference_fn(args, config, policy, t):
    global observation_window
    global lang_embeddings

    # print(f"Start inference_thread_fn: t={t}")
    while True:
        time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],
            observation_window[-2]['images'][config['camera_names'][2]],

            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]],
            observation_window[-1]['images'][config['camera_names'][2]]
        ]

        # fetch debug images in sequence [front, right, left]
        # image_arrs = [
        #     preload_images[config['camera_names'][0]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][2]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][1]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][0]][t],
        #     preload_images[config['camera_names'][2]][t],
        #     preload_images[config['camera_names'][1]][t]
        # ]
        # # encode the images
        # for i in range(len(image_arrs)):
        #     image_arrs[i] = cv2.imdecode(np.frombuffer(image_arrs[i], np.uint8), cv2.IMREAD_COLOR)
        # proprio = torch.from_numpy(preload_images['qpos'][t]).float().cuda()

        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]

        # for i, pos in enumerate(['f', 'r', 'l'] * 2):
        #     images[i].save(f'{t}-{i}-{pos}.png')

        # get last qpos in shape [7, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 7]
        proprio = proprio.unsqueeze(0)

        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")

        print(f"Model inference time: {time.time() - time1} s")

        # print(f"Finish inference_thread_fn: t={t}")
        return actions


# Main loop for the manipulation task
def model_inference(args, config, controller):
    global lang_embeddings

    # Load rdt model
    policy = make_policy(args)

    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]

    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    pre_action[:7] = np.array(
        [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875,
         0.00095367431640625, -0.3393220901489258]
    )
    action = None
    ptr = 0
    chunk_size = 4
    # Inference loop
    with torch.inference_mode():
        while True:
            # The current time step
            t = 0

            action_buffer = np.zeros([chunk_size, config['state_dim']])

            while t < max_publish_step:
                # Update observation window
                update_observation_window(args, config, controller)

                # When coming to the end of the action chunk
                if t % chunk_size == 0:
                    # Start inference
                    action_buffer = inference_fn(args, config, policy, t).copy()
                # print("action_buffer:")
                # print(action_buffer)
                raw_action = action_buffer[t % chunk_size]
                
                # raw_action = np.zeros(7)
                # for i in range(4):
                #     raw_action+=action_buffer[ptr]
                #     ptr+= 1
                action = raw_action
                # print("raw action:")
                # print(action)
                # Interpolate the original action sequence
                if args.use_actions_interpolation:
                    # print(f"Time {t}, pre {pre_action}, act {action}")
                    interp_actions = interpolate_action(args, pre_action, action)
                else:
                    interp_actions = action[np.newaxis, :]
                # Execute the interpolated actions one by one
                for act in interp_actions:
                    right_arm_action = act[:7]
                    # pdb.set_trace()
                    controller.right_arm_controller.move(right_arm_action)
                    # print(f"doing action: {act}")
                    time.sleep(0.1)
                t += 1

                print("Published Step", t)
                pre_action = action.copy()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int,
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int,
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)

    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)

    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store_true',
                        help='Whether to use the robot base to move around',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int,
                        help='The rate at which to publish the actions',
                        default=30, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int,
                        help='The control frequency of the robot',
                        default=25, required=False)

    parser.add_argument('--chunk_size', action='store', type=int,
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float,
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true',
                        help='Whether to use depth images',
                        default=False, required=False)

    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging', default=False)

    parser.add_argument('--config_path', type=str, default="configs/base.yaml",
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True,
                        help='Name or path to the pretrained model')

    parser.add_argument('--lang_embeddings_path', type=str, required=True,
                        help='Path to the pre-encoded language instruction embeddings')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    controller = Controller()
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    model_inference(args, config, controller)


if __name__ == '__main__':
    main()
