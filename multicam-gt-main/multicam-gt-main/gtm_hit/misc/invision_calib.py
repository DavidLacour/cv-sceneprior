import cv2 as cv
import numpy as np
import os
import glob
import json
import sys
import os.path as osp
import numpy as np
from ipdb import set_trace

class CameraParams:

    intrinsics = None
    extrinsics = None
    name=""
    def __init__(self, name=None, intrinsics=None, extrinsics=None):
        self.name = name
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
    
    class Intrinsics:
        cameraMatrix = None
        distCoeffs = None
        Rmat = None
        newCameraMatrix = None
        size = None
        def __init__(self, cameraMatrix=None, distCoeffs=None, Rmat=None, newCameraMatrix=None,size=None):
            self.cameraMatrix = cameraMatrix
            self.distCoeffs = distCoeffs
            self.Rmat = Rmat
            self.newCameraMatrix = newCameraMatrix
            self.size = size
    class Extrinsics:
        R = None
        T = None
        def __init__(self, R=None, T=None):
            self.R = R
            self.T = T
        def get_R_vec(self):
            return cv.Rodrigues(self.R)[0]
    

    def __repr__(self):
        s = self.name +'\nINTRINSICS:\ncameraMatrix:\n' + str(self.intrinsics.cameraMatrix) +\
            '\ndistCoeffs:\n' + str(self.intrinsics.distCoeffs) +\
            '\nRint:\n' + str(self.intrinsics.Rmat) +\
            '\nnewCameraMatrix:\n' + str(self.intrinsics.newCameraMatrix) +\
            '\nsize:\n' + str(self.intrinsics.size)
        if self.extrinsics is not None:
            s += '\nEXTRINSICS:\n'
            s += 'R:\n' + str(self.extrinsics.R) +\
                '\nT:\n' + str(self.extrinsics.T) + '\n'
        return s
    def read_from_xml(self, filename):
        self.intrinsics = CameraParams.Intrinsics()

        node = cv.FileStorage(filename, cv.FileStorage_READ)
        self.intrinsics.cameraMatrix = node.getNode('cameraMatrix').mat()

        distCoeffsNode = node.getNode('distCoeffs')
        self.intrinsics.distCoeffs = np.zeros((1, distCoeffsNode.size()))
        for i in range(distCoeffsNode.size()):
            self.intrinsics.distCoeffs[0,i] = distCoeffsNode.at(i).real()

        self.intrinsics.Rmat = node.getNode('R').mat() #Rotation matrix to correct for the slant s.t. bboxes are upright
        self.intrinsics.newCameraMatrix = node.getNode('newCameraMatrix').mat()

        sizeNode = node.getNode('size')
        # self.size = np.zeros((1, sizeNode.size()))
        # for i in range(sizeNode.size()):
        #     self.size[0,i] = sizeNode.at(i).real()
        self.intrinsics.size = (int(sizeNode.at(0).real()), int(sizeNode.at(1).real()))
        print("Read success.")
        self.set_default()

    def getParams(self):
        return self.cameraMatrix, self.distCoeffs, self.Rmat, self.newCameraMatrix, self.size
    def getMaps(self):
        return cv.initUndistortRectifyMap(*self.getParams(), cv.CV_32FC1)
    def read_from_json(self,filename,extrinsics=True):
        f = open(filename)
        data = json.load(f)
        f.close()
        if extrinsics:
            cext=data["extrinsics"]
            self.extrinsics = CameraParams.Extrinsics()
            self.extrinsics.T = np.array(cext["coordinatesOrigin"])
            self.extrinsics.R = np.vstack([cext["gridXVec"],cext["gridZVec"],cext["upVec"]]).T
        #TODO: add intrinsics
        print("JSON read success.")
        self.set_default()

    def set_default(self):
        if self.intrinsics is not None:
            self.K = self.intrinsics.cameraMatrix
            self.D = self.intrinsics.distCoeffs
        if self.extrinsics is not None:
            self.R = self.extrinsics.R
            self.T = self.extrinsics.T.reshape(-1,1)

    def set_id(self,id):
        self.id = id



 

def load_invision_calib(params_dir):
    cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    cam_id_keys = [f"cam_{cam_id[0]}_{cam_id[1]}" for cam_id in cam_id_mat]
    cam_id_keys_to_idx = dict([cam_id_keys[i],i] for i in range(len(cam_id_keys)))
    videos_captures = {}
    output_data = {}
    cam_params = {}
    for i,cam_id in enumerate(cam_id_mat):
        
        cam_id = tuple(cam_id)
        cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"

        campam = CameraParams(cam_id_key)
        campam.read_from_xml(osp.join(params_dir,f'undistort_params_cam_{cam_id[0]}_{cam_id[1]}.xml'))
        campam.read_from_json(osp.join(params_dir,f'cam_{cam_id[0]}_{cam_id[1]}_calib.json'))
        campam.set_id(cam_id)
        cam_params[i] = campam
        print("Loaded camera parameters for camera",cam_id,".")
    return cam_params