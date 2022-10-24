import numpy as np
import pickle 

def parse_ini(ini_path):
    """Parses camera intrinsic paramter config file of INI format. 

    See http://wiki.ros.org/camera_calibration_parsers for sample file format.

    Args:
        ini_path: path to INI file with camera intrinsics.

    Returns:
        Dictionary of camera intrinsic parameters.

    """
    with open(ini_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        lines[i] = line.rstrip('\n')
    
    cam_params = {}
    cam_params['width'] = int(lines[lines.index('width')+1])
    cam_params['height'] = int(lines[lines.index('height')+1])

    idx = lines.index('camera matrix')
    intrinsics = np.zeros((3,3), dtype=float)
    for i in range(3):
        row = np.fromstring(lines[idx+i+1], dtype=float, sep=' ')
        intrinsics[i] = row
    cam_params['intrinsics'] = intrinsics

    idx = lines.index('distortion')
    distortion = np.fromstring(lines[idx+1], dtype=float, sep=' ')
    cam_params['distortion'] = distortion

    idx = lines.index('rectification')
    rectification = np.zeros((3,3), dtype=float)
    for i in range(3):
        row = np.fromstring(lines[idx+i+1], dtype=float, sep=' ')
        rectification[i] = row
    cam_params['rectification'] = rectification

    idx = lines.index('projection')
    projection = np.zeros((3,4), dtype=float)
    for i in range(3):
        row = np.fromstring(lines[idx+i+1], dtype=float, sep=' ')
        projection[i] = row
    cam_params['projection'] = projection

    return cam_params

def parse_pkl(pkl_path):
    """Parses camera intrinsic paramter config file of PKL format. 

    Args:
        pkl_path: path to PKL file with camera intrinsics.

    Returns:
        Dictionary of camera intrinsic parameters.

    """
    with open(pkl_path, 'rb') as f:
        cam_params = pickle.load(f)
        lines = f.readlines()

    cam_params['intrinsics'] = np.array(cam_params['intrinsics']).reshape(3,3)
    cam_params['distortion'] = np.array(cam_params['distortion'])
    cam_params['rectification'] = np.array(cam_params['rectification']).reshape(3,3)
    cam_params['projection'] = np.array(cam_params['projection']).reshape(3,4)
    return cam_params

    