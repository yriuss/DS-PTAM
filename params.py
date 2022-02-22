import cv2

import numpy as np
import logging
import torch
import h5py
import os
from hloc import extractors
from hloc.utils.base_model import dynamic_load
from hloc.utils.tools import map_tensor
from hloc.utils.parsers import parse_image_lists
from hloc.utils.io import read_image, list_h5_names
from types import SimpleNamespace
import tqdm
conf = {
    'output': 'feats-superpoint-n4096-r1024',
    'model': {
        'name': 'superpoint',
        'nms_radius': 3,
        'max_keypoints': 4096,
    },
    'preprocessing': {
        'grayscale': False,
        'resize_max': 1024,
    },
}


class DeepExtractor():
    def __init__(self, conf):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = dynamic_load(extractors, conf['model']['name'])
        self.model = Model(conf['model']).eval().to(self.device)

    def get_data(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        image = image / 255.

        data = torch.tensor([[image]]).to(self.device)
        return data
    def compute(self, image):
        data = self.get_data(image)

        keypoints, scores, descriptors = self.model(data)

        kp = []
        for kpi in keypoints:
            kp.append(cv2.KeyPoint(kpi[0], kpi[1], 0))

        des = descriptors.T
        return kp, scores, des

class Matcher():
    def __init__(self, FLANN_INDEX_KDTREE):
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    def match(self, des1,des2, k=2):
        matches = self.flann.knnMatch(des1,des2,k=k)
        good = []
        for m,n in matches:
            if m.distance < n.distance:
                good.append(m)
                
        return good


class Params(object):
    def __init__(self):
        
        self.pnp_min_measurements = 10
        self.pnp_max_iterations = 10
        self.init_min_points = 10

        self.local_window_size = 10
        self.ba_max_iterations = 10

        self.min_tracked_points_ratio = 0.5

        self.lc_min_inbetween_frames = 10   # frames
        self.lc_max_inbetween_distance = 3  # meters
        self.lc_embedding_distance = 22.0
        self.lc_inliers_threshold = 15
        self.lc_inliers_ratio = 0.5
        self.lc_distance_threshold = 2      # meters
        self.lc_max_iterations = 20

        self.ground = False

        self.view_camera_size = 1



class ParamsEuroc(Params):
    
    def __init__(self, config='GFTT-BRIEF'):
        super().__init__()

        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=15.0, 
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        elif config == 'ORB-BRIEF':
            self.feature_detector = cv2.ORB_create(
                nfeatures=200, scaleFactor=1.2, nlevels=1, edgeThreshold=31)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
            
        else:
            raise NotImplementedError

        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 2
        self.matching_distance = 25

        self.frustum_near = 0.1  # meters
        self.frustum_far = 50.0

        self.lc_max_inbetween_distance = 4   # meters
        self.lc_distance_threshold = 1.5
        self.lc_embedding_distance = 22.0

        self.view_image_width = 400
        self.view_image_height = 250
        self.view_camera_width = 0.1
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000

    
#GFTT-BRIEF

class ParamsKITTI(Params):
    def __init__(self, config='deep'):
        super().__init__()
        self.deep = False
        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=12.0, 
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        elif config == 'GFTT-BRISK':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=2000, minDistance=15.0, 
                qualityLevel=0.01, useHarrisDetector=False)

            self.descriptor_extractor = cv2.BRISK_create()

        elif config == 'ORB-ORB':
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=1, edgeThreshold=31)
            self.descriptor_extractor = self.feature_detector
        elif config == 'deep':
            self.deep = True
            self.extractor = DeepExtractor(conf)
        else:
            raise NotImplementedError
        
        if(self.deep):
            self.descriptor_matcher = Matcher(1)
        else:
            self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 3
        self.matching_distance = 30

        self.frustum_near = 0.1    # meters
        self.frustum_far = 1000.0

        self.ground = True

        self.lc_max_inbetween_distance = 50
        self.lc_distance_threshold = 15
        self.lc_embedding_distance = 20.0

        self.view_image_width = 400
        self.view_image_height = 130
        self.view_camera_width = 0.75
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -500   # -10
        self.view_viewpoint_z = -100   # -0.1
        self.view_viewpoint_f = 2000