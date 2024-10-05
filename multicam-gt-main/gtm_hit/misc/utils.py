
import os
import json
import numpy as np
from PIL import Image
from ipdb import set_trace
import re
from gtm_hit.misc.geometry import Calibration
from django.conf import settings
def request_to_dict(request):
    #set_trace()
    retdict = {}
    pattern = r"\[([^\]]*)\]|(\w+)"
    #set_trace()
    for k in request.POST.keys():
        matches = re.findall(pattern, k)
        nested_dict = retdict
        if matches[-1][0]==matches[-1][1]=="":
            matches = matches[:-1]
            #set_trace()
        for i,match in enumerate(matches[:-1]):
            if match[0]==match[1]=="": continue
            p = 1 if i==0 else 0
            if match[p] not in nested_dict:
                nested_dict[match[p]] = {}
            nested_dict = nested_dict[match[p]]
        val = request.POST.getlist(k)
        p = 1 if len(matches) == 1 else 0
        last_match = matches[-1]

        if len(val)==1:
            try:
                nested_dict[last_match[p]]= float(val[0])
            except ValueError:
                nested_dict[last_match[p]]= val[0]
        else:
            nested_dict[last_match[p]]= [float(v) for v in val]
    return retdict

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def process_action(obj):
    if "action" in obj:
        action_dict = obj["action"]
        if "changeSize" in action_dict:
            size_dict = action_dict["changeSize"]
            for param,val in size_dict.items():
                if val=="increase":
                    sign =1
                elif val=="decrease":
                    sign =-1
                else:
                    continue
                if param=="height":
                    obj["object_size"][0] += settings.SIZE_CHANGE_STEP*sign
                if param=="width":
                    obj["object_size"][1] += settings.SIZE_CHANGE_STEP*sign
                if param=="length":
                    obj["object_size"][2] += settings.SIZE_CHANGE_STEP*sign
        if "rotate" in action_dict:
            rotation_direction = action_dict["rotate"]

            rotation_theta = obj.get("rotation_theta",0)
            if rotation_direction=="cw":
                rotation_theta+= settings.ROTATION_THETA
            elif rotation_direction=="ccw":
                rotation_theta-= settings.ROTATION_THETA
            obj["rotation_theta"] = rotation_theta

        if "move" in action_dict:
            move_direction = action_dict["move"]
            step = settings.MOVE_STEP
            if "stepMultiplier" in action_dict:
                step*=action_dict["stepMultiplier"]
            if move_direction=="left":
                obj["Xw"] -= step
            elif move_direction=="right":
                obj["Xw"] += step
            elif move_direction=="up":
                obj["Yw"] += step
            elif move_direction=="down":
                obj["Yw"] -= step
            elif move_direction=="forward":
                obj["Zw"] += step
            elif move_direction=="backward":
                obj["Zw"] -= step
    return obj

def convert_rect_to_dict(rect_tuple,cuboid, cam_id, rect_id, world_point,object_size,rotation_theta):
    # if cam_id=="7" or cam_id==7:
    #      set_trace()
    #set_trace()
    if rect_tuple[0] is None:
        x1 = x2 = y1 = y2 = ratio = 0
    else:
        x1 = rect_tuple[0]
        y1 = rect_tuple[1]
        x2 = rect_tuple[2]
        y2 = rect_tuple[3]
        ratio = float(((y2-y1)/(x2 - x1 +1e-6))*0.1)
        if ratio==np.inf:
            ratio=0
    return {'rectangleID': rect_id,
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'cuboid': cuboid,
            'object_size': object_size,
            'rotation_theta': rotation_theta,
            'cameraID': cam_id,
            'ratio': ratio,
            'xMid': int((x1 + x2) / 2),
            "Xw":float(world_point[0]),
            "Yw":float(world_point[1]),
            "Zw":float(world_point[2])
            }


def read_calibs(calib_filepath, camera_names):

        # log.debug(calib_filepath)
        with open(os.path.abspath(calib_filepath)) as f:    
            calibration_json = json.load(f)

        calibs = list()
        dists = list()

        for i, cname in enumerate(camera_names):
            curr_calib = Calibration(K=np.array(calibration_json[cname]["K"]), R=np.array(calibration_json[cname]["R"]), T=np.array(calibration_json[cname]["t"])[..., np.newaxis], view_id=i)
            curr_dist = np.array(calibration_json[cname]["dist"])

            calibs.append(curr_calib)
            dists.append(curr_dist)

        return calibs 
    
def get_frame_size(dset, cams, start_frame):
    sizes = list()
    for cam in cams:
        frame_path = "./gtm_hit/static/gtm_hit/dset/"+dset+"/frames/" + cam + "/" +str(start_frame).zfill(8) + ".jpg" 
        img = Image.open(frame_path)
        sizes.append(img.width)
        sizes.append(img.height)

    return sizes