import os
import cv2
from pathlib import Path
from tqdm import tqdm

def stereo_rectify(img_L, img_R, camera_param_left, camera_param_right):
    """Rectifies stereo image pairs. 

    Args:
        img_L: left cv2 image.
        img_R: right cv2 image.
        camera_param_left: dictionary of left camera intrinsics.
        camera_param_right: dictionary of right camera intrinsics.

    Returns:
        Rectified left and right images.

    """

    K1 = camera_param_left['intrinsics']
    D1 = camera_param_left['distortion']
    R1 = camera_param_left['rectification']
    P1 = camera_param_left['projection']
    
    K2 = camera_param_right['intrinsics']
    D2 = camera_param_right['distortion']
    R2 = camera_param_right['rectification']
    P2 = camera_param_right['projection']
    
    height, width, channels = img_L.shape
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    
    left_rectified = cv2.remap(img_L, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    right_rectified = cv2.remap(img_R, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)    
    return left_rectified, right_rectified

def rectify_folder(data_path, camera_param_left, camera_param_right):
    """Rectifies all stereo image pairs in specified data path. 

    Args:
        data_path: path to folder with LEFT and RIGHT raw stereo images.
        camera_param_left: dictionary of left camera intrinsics.
        camera_param_right: dictionary of right camera intrinsics.

    Returns:
        Saves rectified images to data_path/LEFT_RECT and RIGHT_RECT

    """
    
    Path(os.path.join(data_path, 'LEFT_RECT')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_path, 'RIGHT_RECT')).mkdir(parents=True, exist_ok=True)

    img_paths_L = []
    for root, dirs, files in os.walk(os.path.join(data_path, 'LEFT')):
        for file in files:
            if file.endswith('.jpg'):                
                img_paths_L.append(os.path.join(root, file))

    img_paths_R = []
    for root, dirs, files in os.walk(os.path.join(data_path, 'RIGHT')):
        for file in files:
            if file.endswith('.jpg'):                
                img_paths_R.append(os.path.join(root, file))

    imgs_L_rect, imgs_R_rect = [], []
    for img_L_path, img_R_path in tqdm(list(zip(img_paths_L, img_paths_R))):
        img_L = cv2.imread(img_L_path)
        img_R = cv2.imread(img_R_path)
        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
        img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB)
        img_L_rect, img_R_rect = stereo_rectify(img_L, img_R, camera_param_left, camera_param_right)
        cv2.imwrite(img_L_path.replace('.jpg', '_rect.jpg').replace('/LEFT', '/LEFT_RECT'), cv2.cvtColor(img_L_rect, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img_R_path.replace('.jpg', '_rect.jpg').replace('/RIGHT', '/RIGHT_RECT'), cv2.cvtColor(img_R_rect, cv2.COLOR_RGB2BGR))    