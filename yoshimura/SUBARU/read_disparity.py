import pdb
import json

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def read_disparity_raw(rawfile_path: str, inf_DP: float):
    '''------------------------------------------------------------------------------ 
    disparity_image_width	# 視差画像横サイズ 	256
    right_image_height	# 右画像縦サイズ  	420
    inf_DP			# 補正パラメータ Frame毎のSequenceデータから読み込み
    
    right_i         # 右画像の横(i)座標
    right_j         # 右画像の縦(j)座標
    disparity_i		# 視差画像の横(i)座標	
    disparity_j 	# 視差画像の縦(j)座標
    disparity    	# 視差情報
    distance    	# 距離情報
    -------------------------------------------------------------------------------'''
    # hyperpara
    disparity_image_width = 256
    right_image_height = 420
    
    
    # rawファイルを開く
    with open(rawfile_path, 'rb') as f:
        disparity_image = f.read()
    
    distance = np.zeros((105, 256))
    for right_j in range(420):
        for right_i in range(1000):
            # 右画像座標位置に対応する視差画像座標を求める
            disparity_j = int((right_image_height - right_j - 1) / 4) 	# 縦座標　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　				#視差画像と右画像は原点が左下と左上で違うため上下反転
            disparity_i = int(right_i / 4)  		# 横座標
            # pdb.set_trace()
            # 視差を読み込む   
            disparity =  disparity_image[(disparity_j * disparity_image_width + disparity_i) * 2]                       # 整数視差読み込み
            # disparity += disparity_image[(disparity_j * disparity_image_width + disparity_i) * 2 + 1] / 256     # 小数視差読み込み
            # pdb.set_trace()
            # 視差を距離へ変換
            if disparity > 0:			 # disparity =0 は距離情報がない
                # pdb.set_trace()
                d =  560 / (disparity - inf_DP)
                # outlier
                if (d < 0) or (d > 120):
                    continue
                distance[int(right_j / 4), int(right_i / 4)] = d
                
    
    # fig = go.Figure()
    # fig.add_trace(go.Box(y=distance.flatten()))
    # fig.write_image("box.jpg")
    
    # plt.imshow(distance)
    # plt.colorbar()
    # plt.savefig("test.png")
    # pdb.set_trace()
    
    return distance


