import frameinfo_pb2 as frameinfo__pb2

from google.protobuf import timestamp_pb2
from google.protobuf.json_format import MessageToDict
import json

import cv2
import os
from ipdb import set_trace
import cv2
import numpy as np
from dateutil.parser import parse
import matplotlib.pyplot as plt
# Import required modules
import cv2 as cv
import numpy as np
import os
import json
import argparse


class CameraParams:
    cameraMatrix = None
    distCoeffs = None
    Rmat = None
    newCameraMatrix = None
    size = None

    extrinsics = None
    
    class Extrinsics:
        R = None
        T = None
        def __init__(self, R=None, T=None):
            self.R = R
            self.T = T
        def get_R_vec(self):
            return cv.Rodrigues(self.R)[0]
    

    def __repr__(self):
        s = 'INTRINSICS:\ncameraMatrix:\n' + str(self.cameraMatrix) +\
            '\ndistCoeffs:\n' + str(self.distCoeffs) +\
            '\nR:\n' + str(self.Rmat) +\
            '\nnewCameraMatrix:\n' + str(self.newCameraMatrix) +\
            '\nsize:\n' + str(self.size)
        if self.extrinsics is not None:
            s += '\nEXTRINSICS:\n'
            s += 'R:\n' + str(self.extrinsics.R) +\
                '\nT:\n' + str(self.extrinsics.T) + '\n'
        return s
    def read_from_xml(self, filename):
        node = cv.FileStorage(filename, cv.FileStorage_READ)
        self.cameraMatrix = node.getNode('cameraMatrix').mat()

        distCoeffsNode = node.getNode('distCoeffs')
        self.distCoeffs = np.zeros((1, distCoeffsNode.size()))
        for i in range(distCoeffsNode.size()):
            self.distCoeffs[0,i] = distCoeffsNode.at(i).real()

        self.Rmat = node.getNode('R').mat()
        self.newCameraMatrix = node.getNode('newCameraMatrix').mat()

        sizeNode = node.getNode('size')
        # self.size = np.zeros((1, sizeNode.size()))
        # for i in range(sizeNode.size()):
        #     self.size[0,i] = sizeNode.at(i).real()
        self.size = (int(sizeNode.at(0).real()), int(sizeNode.at(1).real()))
        print("Read success.")
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
    def set_id(self,id):
        self.id = id


def imshow(img,img2=None,figsize=(10,10),save=None):
    if img2 is None:
        plt.figure(figsize=figsize)
        plt.imshow(img[:,:,::-1])
        if save:
            plt.savefig(save)
        plt.show()
    else:
        plt.figure(figsize=figsize)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(img[:,:,::-1])
        axarr[1].imshow(img2[:,:,::-1])
        
        plt.show()

def imshow_dict(imgdict,key_to_idx,figsize=(10,10)):
    plt.figure()
    f, axarr = plt.subplots(1,len(key_to_idx))
    #if imgdict is None or len(imgdict) == 0:
    for (k,img) in imgdict.items():
        axarr[key_to_idx[k]].imshow(img[:,:,::-1])
        axarr[key_to_idx[k]].set_title(k)
        axarr[key_to_idx[k]].title.set_fontsize(10)
    plt.rcParams["figure.figsize"]= figsize
    plt.rc('axes.spines',top=False,bottom=False,left=False,right=False)
    plt.rc('axes',facecolor=(1,1,1,0),edgecolor=(1,1,1,0))
    plt.rc(('xtick','ytick'),color=(1,1,1,0))
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.show()
    return


class CameraFrame:
    def __init__(self, frame = None, output: dict = None , camera_params: CameraParams = None, is_distorted: bool = True, apply_undistortion: bool = True):
        if isinstance(frame, np.ndarray):
            self.frame = frame
        elif isinstance(frame, str):
            self.frame = cv.imread(frame,cv.COLOR_RGB2BGR)
        self.frameId = output.get('frameId',None)
        self.timestamp = output.get('timestamp',None)
        self.network_detections = output.get('network_detections',None)
        self.tracked_detections = output.get('tracked_locations',None)
        self.camera_params = camera_params
        self.is_distorted = is_distorted

        if is_distorted and apply_undistortion:
            self.undistort()
            self.is_distorted = False
            
        self.load_detections()
        
            
    
    def __repr__(self) -> str:
        return f"CameraFrame\
                (timestamp={self.timestamp},\
                frameId={self.frameId}, \
                is_distorted={self.is_distorted}, \
                no_network_detections={len(self.network_detections)}, no_tracked_detections={len(self.tracked_detections)})"
    def undistort(self):
        mapx, mapy = self.camera_params.getMaps()
        self.frame = cv.remap(self.frame, mapx, mapy, cv.INTER_LINEAR)
    def load_detections(self):
        if self.network_detections is not None:
            self.network_detections = [NetworkDetection(det) for det in self.network_detections]
        if self.tracked_detections is not None:
            self.tracked_detections = [TrackedDetection(det) for det in self.tracked_detections]
            self.tracked_detections_dict = dict([(det.track_id, det) for det in self.tracked_detections])

    def get_frame(self, display_network_detections = False, display_tracked_detections = True):
        frame_copy = self.frame.copy()
        if self.is_distorted:
            if display_network_detections and self.network_detections is not None:
                for ndet in self.network_detections:
                    pass
                    #todo finish
            if display_tracked_detections and self.tracked_detections is not None:
                for tdet in self.tracked_detections:
                    
                    cuboid_points3d = tdet.get_cuboid_world_vertices()
                    cuboid_points2d = self.get_projected_points(cuboid_points3d)
                    
                    plot_cuboid(frame_copy, cuboid_points2d)

                    p1,p2 = get_bounding_box(cuboid_points2d)
                    plot_bounding_box(frame_copy, p1,p2,RED,name=f"ID:{tdet.track_id}") #plot cuboid bb
                    


        else: #undistorted
            if display_tracked_detections and self.tracked_detections is not None:
                for tdet in self.tracked_detections:

                    cuboid_points2d = np.array(tdet.cuboid).reshape(-1,2)
                    cuboid_points2d = [tuple(p) for p in cuboid_points2d]
                    cuboid_points2d = cuboid_points2d[:4]+ [cuboid_points2d[6],cuboid_points2d[4],cuboid_points2d[5],cuboid_points2d[7]] #correct the order for plotting
                    
                    plot_cuboid(frame_copy, cuboid_points2d,PURPLE)
                    
                    p1,p2 = get_bounding_box(cuboid_points2d) #compute cuboid bbox
                    plot_bounding_box(frame_copy, p1,p2,PINK,name=f"ID:{tdet.track_id}") #plot cuboid bbox


                    #NEW: get projected with undistortion
                    cuboid_points3d = tdet.get_cuboid_world_vertices()
                    cuboid_points2d = self.get_projected_points(cuboid_points3d,undistort=True)
                    
                    plot_cuboid(frame_copy, cuboid_points2d,GREEN)
                    
                    p1,p2 = get_bounding_box(cuboid_points2d) #compute cuboid bbox
                    plot_bounding_box(frame_copy, p1,p2,RED,name=f"ID:{tdet.track_id}") #plot cuboid bbox

                    # p1,p2 = tdet.get_bbox_points()
                    # plot_bounding_box(frame_copy, p1,p2,PINK,name=f"ID:{tdet.track_id}") #plot precomputed bbox

                    
        return frame_copy
    def get_tracked_person(self, track_id,return_points=False):
        frame_copy = self.frame.copy()
        if self.tracked_detections is not None:
            if track_id in self.tracked_detections_dict:
                tdet = self.tracked_detections_dict[track_id]
                if self.is_distorted:
                    cuboid_points3d = tdet.get_cuboid_world_vertices()
                    cuboid_points2d = self.get_projected_points(cuboid_points3d)

                    p1,p2 = get_bounding_box(cuboid_points2d)
                            
                else: #undistorted
                    cuboid_points2d = np.array(tdet.cuboid).reshape(-1,2)
                    cuboid_points2d = [tuple(p) for p in cuboid_points2d]

                    p1,p2 = get_bounding_box(cuboid_points2d) #compute cuboid bbox
                    # p1,p2 = tdet.get_bbox_points() #precomputed bbox points
            else:
                return None
        else:
            return None
        
        if return_points:
            return p1,p2
        
        x1,y1 = p1
        x2,y2 = p2

        return frame_copy[y1:y2,x1:x2]
    

    ##delete?
    def get_tracked_all(self,return_points=False):
        frame_copy = self.frame.copy()
        bboxes = []

        if self.tracked_detections is not None:
            for tdet in self.tracked_detections:
                if self.is_distorted:
                    cuboid_points3d = tdet.get_cuboid_world_vertices()
                    cuboid_points2d = self.get_projected_points(cuboid_points3d)

                    p1,p2 = get_bounding_box(cuboid_points2d)
                    
                else: #undistorted
                    cuboid_points2d = np.array(tdet.cuboid).reshape(-1,2)
                    cuboid_points2d = [tuple(p) for p in cuboid_points2d]

                    #p1,p2 = get_bounding_box(cuboid_points2d) #compute cuboid bbox
                    p1,p2 = tdet.get_bbox_points() #precomputed bbox points


                if return_points:
                    bboxes.append((p1,p2))
                else:
                    frame_copy = self.frame.copy()
                    x1,y1 = p1
                    x2,y2 = p2
                    bboxes.append(frame_copy[y1:y2,x1:x2])
        else:
            return None
        
        return bboxes
    
    def show_frame(self, display_network_detections = False, display_tracked_detections = True):
        frame_copy = self.get_frame(display_network_detections, display_tracked_detections)
        if frame_copy is None:
            return None
        imshow(frame_copy,figsize=(20,20))
        
    def show_tracked_person(self, track_id):
        frame_copy = self.get_tracked_person(track_id)
        imshow(frame_copy,figsize=(20,20))
        
    def get_projected_points(self,points3d,undistort=False):
        points3d = np.array(points3d).reshape(-1,3)
        Rvec = self.camera_params.extrinsics.get_R_vec() #cv.Rodrigues
        Tvec = self.camera_params.extrinsics.T
        points2d,_ = cv.projectPoints(points3d,Rvec,Tvec,self.camera_params.cameraMatrix,self.camera_params.distCoeffs)
        if undistort:
            #points3d_homogenous = np.hstack([points3d,np.ones((points3d.shape[0],1))])
            points3d_cam = self.camera_params.extrinsics.R @ points3d.T + self.camera_params.extrinsics.T.reshape(-1,1)
            points3d_cam_rectified = self.camera_params.Rmat @ points3d_cam #correct the slant of the camera
            points2d = self.camera_params.newCameraMatrix @ points3d_cam_rectified

            points2d = points2d[:2,:]/points2d[2,:]
            points2d = points2d.T
        points2d = np.squeeze(points2d)
        points2d = [tuple(p) for p in points2d]
        return points2d

def plot_cuboid(img, vertices, color=(255, 255, 0), scale=1):
    vertices = vertices.copy()
    for idx,point in enumerate(vertices):
        if point:
            point = tuple([int(p*scale) for p in point])
    line_1 = [0, 1, 5, 4, 4, 0, 1, 5, 7, 3, 2, 6]
    line_2 = [1, 5, 4, 0, 7, 3, 2, 6, 3, 2, 6, 7]
    for p1, p2 in zip(line_1, line_2):
        if vertices[p1] and vertices[p2]:
            _p1 = tuple([int(p * scale) for p in vertices[p1]])
            _p2 = tuple([int(p * scale) for p in vertices[p2]])
            img = cv.line(img, _p1, _p2, color=color, thickness=1)

def get_bounding_box(points):
    points = points.copy()
    points = np.array(points,dtype=np.int32).reshape(-1,1,2)
    x,y,w,h = cv.boundingRect(points)
    return (x,y),(x+w,y+h)

def plot_bounding_box(img, p1,p2,color=(0, 0, 255),thickness=2, name=None):
    if name is not None:
        # print(p1,p2)
        # print(name)
        textLoc = (p1[0], p1[1]-20)
        draw_text(img, name, textLoc)  
    cv.rectangle(img, p1, p2, color, thickness)

def draw_text(img, text,
          pos=(0, 0),
          font=cv.FONT_HERSHEY_PLAIN,
          font_scale=1,
          text_color=(255, 255, 255),
          font_thickness=1,
          text_color_bg=(0, 0, 0)
          ):
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness, cv.LINE_AA)
    return text_size

class Detection:
    def __init__(self, bbox, class_name, cuboid):
        self.bbox = bbox
        self.class_name = class_name
        self.cuboid = cuboid
    
    def __init__(self, detection_dict:dict):
        self.bbox = detection_dict["bbox"]
        self.class_name = detection_dict["class"]
        self.cuboid=detection_dict["cuboid"]
    
    def get_bbox_points(self):
        x1,y1,x2,y2 = [int(x) for x in self.bbox]
        return (x1,y1),(x2,y2)

class NetworkDetection(Detection):
    def __init__(self,bbox, class_name, cuboid, score):
        super().__init__(bbox, class_name, cuboid)
        self.score = score
    def __init__(self, detection_dict):
        self.bbox = detection_dict["bbox"]
        self.class_name = detection_dict["class"]
        self.cuboid=detection_dict["cuboid"]
        self.score = detection_dict["ml_score"]

class TrackedDetection(Detection):
    def __init__(self,bbox, class_name, cuboid, score, cuboid_to_world_transform,detection_associated_idx,object_size, track_id, uncertainty_ellipse_m2):
        super().__init__(bbox, class_name, cuboid, score)
        self.cuboid_to_world_transform = cuboid_to_world_transform
        self.detected_associated_idx = detection_associated_idx
        self.object_size = object_size
        self.track_id = track_id
        self.uncertainty_ellipse_m2 = uncertainty_ellipse_m2
        self.compute_cuboid()
    def __repr__(self) -> str:
        return f"TrackedDetection({self.__dict__})"
    def __init__(self, tracked_location_dict):
        super().__init__(tracked_location_dict)
        self.cuboid_to_world_transform = tracked_location_dict["cuboidToWorldTransform"]
        self.detected_associated_idx = tracked_location_dict["detection_associated_idx"]
        self.object_size = tracked_location_dict["objectSize"]
        self.track_id = tracked_location_dict["trackId"]
        self.uncertainty_ellipse_m2 = tracked_location_dict["uncertainty_ellipse_m2"]
        self.compute_cuboid()

    def compute_cuboid(self):
        self.cuboid_obj = Cuboid(self.object_size)
        self._cuboid_world_vertices = self.cuboid_obj.get_world_vertices(self.cuboid_to_world_transform)
    def get_cuboid_world_vertices(self):
        return self._cuboid_world_vertices

class Cuboid:
    def __init__(self, object_size):
        self.object_size = object_size
        self.vertices = np.zeros((CUBOID_VERTEX_COUNT, 3))
        self.gen_vertices()
    def __repr__(self) -> str:
        return "Cuboid({})".format(self.vertices)
    def gen_vertices(self):
        width, length, height = self.object_size
        self.vertices[CuboidVertexEnum.FrontTopRight] = [width / 2, length / 2, height]
        self.vertices[CuboidVertexEnum.FrontTopLeft] = [-width / 2, length / 2, height]
        self.vertices[CuboidVertexEnum.FrontBottomLeft] = [-width / 2, -length / 2, height]
        self.vertices[CuboidVertexEnum.FrontBottomRight] = [width / 2, -length / 2, height]
        self.vertices[CuboidVertexEnum.RearTopRight] = [width / 2, length / 2, 0]
        self.vertices[CuboidVertexEnum.RearTopLeft] = [-width / 2, length / 2, 0]
        self.vertices[CuboidVertexEnum.RearBottomLeft] = [-width / 2, -length / 2, 0]
        self.vertices[CuboidVertexEnum.RearBottomRight] = [width / 2, -length / 2, 0]
        self.base = [0, 0, 0]
    
    def get_world_vertices(self, cuboid_to_world_transform):
        vertices = np.hstack([self.vertices,np.ones((8,1))])
        return (cuboid_to_world_transform @ vertices.T).T[:,:3]

PINK = (250,190,203)
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
YELLOW = (0,255,255)
ORANGE = (0,140,255)
PURPLE = (255,0,255)

from enum import IntEnum

class CuboidVertexEnum(IntEnum):
    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    Base = 8
CUBOID_VERTEX_COUNT = 8
        

def load_frame_info_chunks(file_path):
    frame_info_list = []

    with open(file_path, "rb") as file:
        while True:
            frame_info = frameinfo__pb2.FrameInfo()
            data = file.read(4)
            if not data:
                break
            size = int.from_bytes(data, byteorder="little")
            frame_info.ParseFromString(file.read(size))
            frame_info_list.append(frame_info)

    return frame_info_list

def frame_info_to_readable_format(frame_info_list):
    readable_frame_info_list = []

    for frame_info in frame_info_list:
        #frame_info_dict = MessageToDict(frame_info)
        #frame_info_dict['timestamp'] = frame_info.timestamp.ToJsonString()
        readable_frame_info_list.append(frame_info.timestamp.ToJsonString())

    return readable_frame_info_list

def save_frames_from_video(cap, frame_info_list, cam_id, cam_data,cam_param,output_directory, start_time, end_time,target_fps=2,apply_undistortion=True,output_format="jpg"):
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    #set_trace()
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = video_fps // target_fps
    frame_indices = set(range(0, total_frames, frame_interval))
    
    for frame_id,timestamp in enumerate(frame_info_list):
        timestamp = parse(timestamp)
        if (start_time <= timestamp <= end_time) and (frame_id in frame_indices):
            print(f"{cam_id} frame_id: {frame_id:5d} timestamp: {timestamp}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                print("stopping, frame_id: ", frame_id)

            camera_frame = CameraFrame(frame, {},cam_param, is_distorted=True,apply_undistortion=apply_undistortion)
            framesave = camera_frame.get_frame(display_network_detections = False, display_tracked_detections = False)

            output_file = f"{output_directory}/{frame_id:08d}.{output_format}"
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            cv2.imwrite(output_file, framesave)
    print("Done. Frames saved to: ", output_directory,"last frame_id:", frame_id)
    cap.release()

def process_multiple_cameras(input_dir, output_base_dir, start_time, end_time, target_fps=2,apply_undistortion=True,output_format="jpg"):
    """
    input_dir: directory containing the video_output folder
    output_base_dir: directory where the frames will be saved
    start_time: datetime object
    end_time: datetime object
    target_fps: frames per second to save

    The following structure is assumed:
    <input_dir>
        json_output
            *.json
        example_footage
            *.mp4
            *.chunks
        calib
            *.json
            *.xml
    """
    cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    cam_id_keys = [f"cam_{cam_id[0]}_{cam_id[1]}" for cam_id in cam_id_mat]
    cam_id_keys_to_idx = dict([cam_id_keys[i],i] for i in range(len(cam_id_keys)))
    videos_captures = {}
    output_data = {}
    cam_params = {}
    cams_chunks = {}
    for i,cam_id in enumerate(cam_id_mat):
        
        cam_id = tuple(cam_id)
        cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"
        cap = cv2.VideoCapture(f"{input_dir}/example_footage/cam_{cam_id[0]}_{cam_id[1]}.mp4")
        if (cap.isOpened() == False): 
            print("Unable to read camera feed for camera",cam_id,". Skipping...")
            continue
        videos_captures[cam_id_key] = cap
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print("Loaded video capture for camera",cam_id, "with", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "frames.")


        file_path = f"{input_dir}/example_footage/cam_{cam_id[0]}_{cam_id[1]}.mp4-frameInfo.chunks"  
        frame_info_list = load_frame_info_chunks(file_path)
        readable_frame_info_list = frame_info_to_readable_format(frame_info_list)
        cams_chunks[cam_id_key] = readable_frame_info_list
        print("Loaded frame info chunks for camera",cam_id, "for", len(readable_frame_info_list), "frames.")
        
        f = open(f'{input_dir}/json_output/cam_{cam_id[0]}_{cam_id[1]}.json')
        data = json.load(f)
        data = dict([(frame["frameId"],frame) for frame in data])
        data["id"]=cam_id
        output_data[cam_id_key] = data
        print( "Loaded json output data for camera",cam_id, "for", len(data), "frames.")

        campam = CameraParams()
        campam.read_from_xml(f'{input_dir}/calib/undistort_params_cam_{cam_id[0]}_{cam_id[1]}.xml')
        campam.read_from_json(f'calib/cam_{cam_id[0]}_{cam_id[1]}_calib.json')
        campam.set_id(cam_id)
        cam_params[cam_id_key] = campam
        print("Loaded camera parameters for camera",cam_id,".")
        print("="*60)

    for i,cam_id in enumerate(cam_id_mat):
        cam_id = tuple(cam_id)
        cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"
        
        cam_data = None #output_data[cam_id_key]
        cam_param = cam_params[cam_id_key]
        video_cap = videos_captures[cam_id_key]
        cam_chunks = cams_chunks[cam_id_key]
        
        print(f"Processing camera {cam_id_key}...")
        cam_name = f"cam{i}"
        frames_path = "undistorted_frames" if apply_undistortion else "frames"
        output_directory = os.path.join(output_base_dir,frames_path,cam_name)
        os.makedirs(output_directory, exist_ok=True)

        save_frames_from_video(video_cap, cam_chunks, cam_name,cam_data, cam_param, output_directory, start_time, end_time, target_fps=target_fps, apply_undistortion = apply_undistortion,output_format=output_format)


def main():
    #python invison_frames_process.py your/input/path gtm_hit/static/gtm_hit/dset/myDataset 2022-06-10T16:00:00Z 2022-06-10T16:04:00Z
    parser = argparse.ArgumentParser(description="Save frames from video within a specified time interval.")
    parser.add_argument("input_path", help="Input path. Must contain the structure as described in the README and process_multiple_cameras function")
    parser.add_argument("output_path", help="Directory to save the extracted frames. To be used as the dataset name in the annotation tool.")
    parser.add_argument("start_time", help="Start time of the interval (ISO format, e.g., 2022-06-10T16:00:00Z).")
    parser.add_argument("end_time", help="End time of the interval (ISO format, e.g., 2022-06-10T16:00:30Z).")
    parser.add_argument("--target_fps", type=int, default=2, help="Target frames per second (default: 2).")
    parser.add_argument("--output_format", default="jpg", help="Output format (default: jpg).")
    #add argument for undistortion (bool)
    parser.add_argument("--apply_undistortion", type=bool, default=True, help="Apply undistortion (default: True).")
    args = parser.parse_args()

    start_time = parse(args.start_time)
    end_time = parse(args.end_time)
    process_multiple_cameras(args.input_path, args.output_path, start_time, end_time, target_fps=args.target_fps,apply_undistortion=args.apply_undistortion,output_format=args.output_format)

if __name__ == "__main__":
    main()