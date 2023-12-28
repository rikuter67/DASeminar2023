import pdb

import cv2
import numpy as np

def pre_processing(frame: np.ndarray, sequence: dict) -> np.ndarray:
    X, Y = (sequence['TgtXPos_LeftUp'], sequence['TgtYPos_LeftUp'])
    width_Bbox, height_Bbox = (sequence['TgtWidth'], sequence['TgtHeight'])
    # pdb.set_trace()
    bbox_img = cv2.rectangle(frame, (int(X), int(Y)), (int(X+width_Bbox), int(Y+height_Bbox)), (255, 0, 0), 3)
    cv2.imwrite("bbox.png", bbox_img)