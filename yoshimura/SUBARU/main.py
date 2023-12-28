import os
import json
import pdb

import cv2
import numpy as np

from read_disparity import read_disparity_raw
from processing import pre_processing
from metrics import frame_error


def main():
    train_video_path = "./train_videos"
    train_ano_path = "./train_annotations"
    # train_disparity_path = os.path.join(train_video_path, )
    if not os.path.exists(train_video_path):
        raise FileExistsError(f"Not exists {train_video_path}")
    # scene_id_list = os.listdir(train_disparity_path)
    for scene_id in os.listdir(train_video_path):
        scene_id_path = os.path.join(train_video_path, scene_id)
        scene_id_disparity_path = os.path.join(scene_id_path, "disparity")
        
        ''' アノテーションデータの読み取り '''
        scene_id_json_path = os.path.join(train_ano_path, scene_id+'.json') # anotation .json
        with open(scene_id_json_path, 'r') as file:
            ano_data = json.load(file)
        ''' 右動画ファイルの読み取り'''
        scene_id_videoR_path = os.path.join(scene_id_path, 'Right.mp4') 
        cap_file = cv2.VideoCapture(scene_id_videoR_path)
        delta_t = 1 / (cap_file.get(cv2.CAP_PROP_FPS) * 3600.) # delta hour
        # pdb.set_trace()
        if not cap_file.isOpened():
            raise FileNotFoundError(f"Not found {scene_id_videoR_path}")
    
        # 過去ディレクトリ
        past_distance = []
        past_velocity = []
        for idx, frame_id in enumerate(sorted(os.listdir(scene_id_disparity_path))):
            
            frame_id_path = os.path.join(scene_id_disparity_path, frame_id)
            frame_id_ano_data = ano_data['sequence'][idx]
            
            distance_img = read_disparity_raw(frame_id_path, frame_id_ano_data['inf_DP']) # 105x256  original.shape // 4
            _, frame = cap_file.read()
            # pre_processing(frame, frame_id_ano_data)
            
            ''' 先行車までの距離を計算 '''
            x_crop, y_crop = (int(frame_id_ano_data['TgtXPos_LeftUp'] / 4), int(frame_id_ano_data['TgtYPos_LeftUp'] / 4))
            width_crop, height_crop = (int(frame_id_ano_data['TgtWidth'] / 4), int(frame_id_ano_data['TgtHeight'] / 4))
            distance = np.mean(distance_img[y_crop:y_crop+height_crop, x_crop:x_crop+width_crop])
            # tmp = cv2.rectangle(distance_img, (x_crop, y_crop), (x_crop+width_crop, y_crop+height_crop), (255, 0, 0))
            # print("distance score:", frame_id_ano_data['Distance_ref'])
            # print("avg distance image:", np.mean(distance_img[y_crop:y_crop+height_crop, x_crop:x_crop+width_crop]))
            
            ''' 最初のフレームだけは計測しない '''
            if idx == 0:
                past_distance.append(distance)
                past_velocity.append(frame_id_ano_data['TgtSpeed_ref'])
                continue
            
            ''' 先行車の速度 '''
            velocity = past_velocity[-1] + ((distance - past_distance[-1]) / delta_t)
            print("Error:", frame_error(velocity, frame_id_ano_data['TgtSpeed_ref']))
            
            ''' 過去の情報を保存 '''
            past_distance.append(distance)
            past_velocity.append(velocity)
            


if __name__ == "__main__":
    main()