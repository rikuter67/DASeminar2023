import os
import json
import pdb
import sys

import cv2
import numpy as np
from tqdm import tqdm
from scipy import stats
from PIL import Image
# import plotly.graph_objects as go
# from ultralytics import YOLO
# import torch

from read_disparity import read_disparity_raw
# from processing import pre_processing
# from metrics import frame_error
from utils import *

from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

def train():
    video_path = "./train_videos"
    ano_path = "./train_annotations"
    scene_id_dict = {}

    for scene_id in tqdm(sorted(os.listdir(video_path))):
        if scene_id == ".gitignore":
            continue
        
        # if scene_id != "002":
        #     continue
        scene_id_path = os.path.join(video_path, scene_id)
        videoR = cv2.VideoCapture(os.path.join(scene_id_path, 'Right.mp4'))
        delta_t = 1 / videoR.get(cv2.CAP_PROP_FPS)

        scene_id_json_path = os.path.join(ano_path, scene_id + '.json')
        with open(scene_id_json_path, 'r') as f:
            ano_data = json.load(f)

        frame_list = sorted(os.listdir(os.path.join(scene_id_path, "disparity")))
        frame_num = len(frame_list)

        velocity_l = []
        distance = []
        alpha = 0.5
        
        # trackerの定義
        # params = cv2.TrackerKCF_Params()
        # params.detect_thresh = 0.4
        # params.interp_factor = 0.02
        # tracker = cv2.TrackerKCF_create(params)
        
        tracker = cv2.TrackerCSRT_create()

        for idx, frame_id in enumerate(frame_list):
            print(f"scene_id: {scene_id}", f"Frame No.:{idx + 1}/{frame_num}")
            frame_id_disparity_path = os.path.join(scene_id_path, "disparity", frame_id)
            frame_id_ano_data = ano_data['sequence'][idx]

            # 距離画像
            distance_img = read_disparity_raw(frame_id_disparity_path, frame_id_ano_data['inf_DP'])
            # 静止画
            _, frame = videoR.read()
            
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            transforms = weights.transforms()

            images = [transforms(Image.fromarray(frame))]

            model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
            model = model.eval()

            output = model(images)
            print([weights.meta["categories"][label] for label in output[0]['labels']])
            pdb.set_trace()
            # マスクの作成
            mask = np.zeros_like(frame)
            mask[:, (frame.shape[1] // 3):((frame.shape[1] // 3) * 2), :] = 1
            frame = frame * mask

            if idx == 0: # 最初のフレーム
                bbox = (
                    int(frame_id_ano_data['TgtXPos_LeftUp']),
                    int(frame_id_ano_data['TgtYPos_LeftUp']),
                    int(frame_id_ano_data['TgtWidth']),
                    int(frame_id_ano_data['TgtHeight'])
                )
                tracker.init(frame, bbox)
            else:   # 以降のフレーム
                ret, bbox = tracker.update(frame)
                
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color=(0,0,255))
            cv2.imwrite("test1.png", frame)
            
            
            resize_scale_box = [int(point / 4) for point in bbox]

            d_bbox = distance_img[resize_scale_box[1]:(resize_scale_box[1] + resize_scale_box[3]), resize_scale_box[0]:(resize_scale_box[0] + resize_scale_box[2])]

            try:
                d_est = dbscan(d_bbox, eps=3.0, min_samples=10)
                if idx == 0 or np.abs(d_est - distance[-1]) > 5:
                    d_est = distance[-1] if distance else d_est
                print(f"推定距離：{d_est:.3f}, 正解距離: {frame_id_ano_data['Distance_ref']}")
            except Exception as e:
                d_est = distance[-1] if distance else 0
                print(f"推定失敗：{d_est}, エラー：{e}")

            if idx == 0:
                velocity = 0
                velocity_l.append(velocity)
            elif idx == 1:
                velocity = frame_id_ano_data['OwnSpeed'] + ((d_est - distance[-1]) / delta_t) * 3.6
                velocity_l.append(velocity)
            else:
                # d_avg = (sum(distance[-5:]) + d_est) / (len(distance[-5:]) + 1) if idx > 5 else d_est

                velocity = frame_id_ano_data['OwnSpeed'] + ((d_est - distance[-1]) / delta_t) * 3.6
                velocity = alpha * velocity_l[-1] + (1 - alpha) * velocity
                
                velocity_l.append(velocity)
            
            print(f"　　先行車推定速度: {velocity:.1f}", f" 正解速度:{frame_id_ano_data['TgtSpeed_ref']}"  "　　自車速度:", frame_id_ano_data['OwnSpeed'])
                

            distance.append(d_est)
            
        videoR.release()
        scene_id_dict[scene_id] = velocity_l

    with open('train_pred.json', 'w', encoding='utf-8') as f:
        json.dump(scene_id_dict, f, ensure_ascii=False)


def test():
    video_path = "./test_videos"
    ano_path = "./test_annotations"
    scene_id_dict = {}

    for scene_id in tqdm(sorted(os.listdir(video_path))):
        if scene_id == ".gitignore":
            continue
        # if scene_id != "044":
        #     continue
        
        scene_id_path = os.path.join(video_path, scene_id)
        videoR = cv2.VideoCapture(os.path.join(scene_id_path, 'Right.mp4'))
        delta_t = 1 / videoR.get(cv2.CAP_PROP_FPS)

        scene_id_json_path = os.path.join(ano_path, scene_id + '.json')
        with open(scene_id_json_path, 'r') as f:
            ano_data = json.load(f)

        frame_list = sorted(os.listdir(os.path.join(scene_id_path, "disparity")))
        frame_num = len(frame_list)

        past_own_velocity = []
        lost_flag = False
        past_bbox = 0
        velocity_l = []
        distance = []
        alpha = 0.5
        
        # trackerの定義
        # params = cv2.TrackerKCF_Params()
        # params.detect_thresh = 0.4
        # params.interp_factor = 0.02
        # tracker = cv2.TrackerKCF_create(params)
        tracker = cv2.TrackerCSRT_create()

        for idx, frame_id in enumerate(frame_list):
            print(f"scene_id: {scene_id}", f"Frame No.:{idx + 1}/{frame_num}")
            frame_id_disparity_path = os.path.join(scene_id_path, "disparity", frame_id)
            frame_id_ano_data = ano_data['sequence'][idx]

            # 距離画像
            distance_img = read_disparity_raw(frame_id_disparity_path, frame_id_ano_data['inf_DP'])
            # 静止画
            _, frame = videoR.read()
            # マスクの作成
            mask = np.zeros_like(frame)
            mask[:, (frame.shape[1] // 4):((frame.shape[1] // 3) * 2), :] = 1
            frame = frame * mask

            if idx == 0: # 最初のフレーム
                init_bbox = (
                    int(frame_id_ano_data['TgtXPos_LeftUp']),
                    int(frame_id_ano_data['TgtYPos_LeftUp']),
                    int(frame_id_ano_data['TgtWidth']),
                    int(frame_id_ano_data['TgtHeight'])
                )
                bbox = init_bbox
                tracker.init(frame, init_bbox)
            elif lost_flag:
                result = cv2.matchTemplate(frame, past_bbox, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                top_left = max_loc
                h, w, _ = past_bbox.shape
                bbox = (top_left[0], top_left[1], w, h)
                tracker.init(frame, bbox)
            else:   # 以降のフレーム
                ret, bbox = tracker.update(frame)
                bbox = list(bbox)
                for idx, positon in enumerate(bbox):
                    if positon < 0:
                        bbox[idx] = 0
                    else:
                        pass
                if not ret:
                    lost_flag = True
                else:
                    past_bbox = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2],:]
                
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color=(0,0,255))
            cv2.imwrite("test1.png", frame)

            resize_scale_box = [int(point / 4) for point in bbox]

            d_bbox = distance_img[resize_scale_box[1]:(resize_scale_box[1] + resize_scale_box[3]), resize_scale_box[0]:(resize_scale_box[0] + resize_scale_box[2])]

            try:
                d_est = dbscan(d_bbox, eps=3.0, min_samples=10)
                if idx == 0: 
                    distance.append(d_est)
                elif np.abs(d_est - distance[-1]) > 5:
                    d_est = distance[-1] 
                print(f"推定距離：{d_est:.3f}")
            except Exception as e:
                d_est = distance[-1] if distance else 0
                print(f"推定失敗：{d_est}, エラー：{e}")

            if idx == 0:
                velocity_l.append(0)
            elif idx == 1:
                velocity = (past_own_velocity[-1] +  frame_id_ano_data['OwnSpeed']) * 0.5 + ((d_est - distance[-1]) / delta_t) * 3.6
                velocity_l.append(velocity)
            else:
                # d_avg = (sum(distance[-5:]) + d_est) / (len(distance[-5:]) + 1) if idx > 5 else d_est
                velocity = (past_own_velocity[-1] +  frame_id_ano_data['OwnSpeed']) * 0.5 + ((d_est - distance[-1]) / delta_t) * 3.6
                velocity = alpha * velocity_l[-1] + (1 - alpha) * velocity
                if velocity < 0:
                    velocity = 0.
                print(f"　　先行車推定速度: {velocity:.1f}",  "　　自車速度:", frame_id_ano_data['OwnSpeed'])
                velocity_l.append(velocity)
                
            distance.append(d_est)
            past_own_velocity.append(frame_id_ano_data['OwnSpeed'])
            
            
        videoR.release()
        scene_id_dict[scene_id] = velocity_l

    with open('test_sub.json', 'w', encoding='utf-8') as f:
        json.dump(scene_id_dict, f, ensure_ascii=False)


if __name__ == "__main__":
    # pdb.set_trace()
    if sys.argv[1] == '--train':
        print("train")
        train()
    elif sys.argv[1] == '--test':
        print("test")
        test()
    else:
        print("No setting comannd line!")
