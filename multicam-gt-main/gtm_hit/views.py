# -*- coding: utf-8 -*-
from curses.textpad import rectangle
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib.auth import authenticate, login, logout
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect, HttpResponse, HttpResponseNotFound
from django.core import serializers
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.conf import settings
from .models import Worker, ValidationCode, MultiViewFrame, View, Annotation, Annotation2DView, Person, Dataset
from django.template import RequestContext
from django.http import JsonResponse
from django.db import transaction
from django.views.decorators.csrf import csrf_exempt
import re
import json
import os
import random as rand
from threading import Thread
from ipdb import set_trace
import random
import numpy as np
from gtm_hit.misc import geometry
from gtm_hit.misc.db import *
from gtm_hit.misc.serializer import *
from gtm_hit.misc.utils import convert_rect_to_dict, request_to_dict, process_action
from pprint import pprint
import uuid
from gtm_hit.misc.invision.create_video import create_video as create_video_invision

def requestID(request):
    
    context = RequestContext(request).flatten()
    if request.method == "POST":
        if 'wID' in request.POST:
            workerID = request.POST['wID']
            pattern = re.compile("^[A-Z0-9]+$")
            if pattern.match(workerID):
                if 'datasetName' in request.POST:
                    dataset_name = request.POST['datasetName']
                    return redirect(f"/gtm_hit/{dataset_name}/{workerID}/processInit")
    return render(request, 'gtm_hit/requestID.html', context)

def processInit(request, dataset_name, workerID):
    
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == -1:
            w.state = 0
            w.save()
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def index(request, workerID,dataset_name):
    
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state != 0:
            return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
        return render(request, 'gtm_hit/index.html', {'workerID': workerID, **context, 'dset_name': dataset_name})

    except Worker.DoesNotExist:
        
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def processIndex(request, workerID,dataset_name):
    
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == 0:
            w.state = 3
            w.save()
    except Worker.DoesNotExist:
        
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def dispatch(request, dataset_name,workerID):
    
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk=workerID)
        try:
            code = ValidationCode.objects.get(worker_id=w)
            stop = False
            i = 2
            while not stop:
                try:
                    w2 = Worker.objects.get(pk=workerID+str(i))
                    c2 = ValidationCode.objects.get(worker_id=w2)
                    i = i + 1
                except Worker.DoesNotExist:
                    stop = True
                except ValidationCode.DoesNotExist:
                    return redirect(f"/gtm_hit/{dataset_name}/{workerID+str(i)}")
            return redirect(f"/gtm_hit/{dataset_name}/{workerID+str(i)}")
        except ValidationCode.DoesNotExist:
            pass
    except Worker.DoesNotExist:
        w = registerWorker(workerID)
    
    dataset,_ = Dataset.objects.get_or_create(name=dataset_name)

    urlpath = "/gtm_hit/"+dataset_name+"/"+workerID+"/"
    #urlpath = ""
    state = w.state
    if state == 0:
        return redirect(urlpath+'index')
        # return render(request, 'gtm_hit/frame.html',{'frame_number': frame_number, 'workerID' : workerID},context)
    elif state == 1:
        return redirect(urlpath+'frame')
#        return render(request, 'gtm_hit/finish.html',{'workerID' : workerID, 'validation_code' : validation_code},context)
    elif state == 2:
        return redirect(urlpath+'finish')
#        return render(request, 'gtm_hit/finish.html',{'workerID' : workerID, 'validation_code' : validation_code},context)
    elif state == 3:
        return redirect(urlpath+'tuto')
    elif state == -1:
        return redirect(urlpath+'processInit')
        # return render(request, 'gtm_hit/index.html',{'workerID' : workerID},context)
    else:
        return redirect(urlpath+'index')
        # return render(request, 'gtm_hit/index.html',{'workerID' : workerID},context)

def frame(request, dataset_name,workerID):
    
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state != 1:
            return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
        if w.frameNB < 0:
            w.frameNB = settings.STARTFRAME
            w.save()
        frame_number = w.frameNB
        nblabeled = w.frame_labeled

        try:
            dataset,_ = Dataset.objects.get_or_create(name=dataset_name)
        except Dataset.DoesNotExist:
            return HttpResponseNotFound("Dataset not found")

        return render(request, 'gtm_hit/frame.html', {'dset_name': dataset.name, 'frame_number': frame_number, 'frame_inc': settings.INCREMENT, 'workerID': workerID, 'cams': settings.CAMS, 'frame_size': settings.FRAME_SIZES, 'nb_cams': settings.NB_CAMS, 'nblabeled': nblabeled, **context, "undistort": settings.UNDISTORTED_FRAMES})
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")

def processFrame(request, workerID,dataset_name):
    
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == 1 and w.frame_labeled >= 500:
            w.state = 2
            timelist = w.getTimeList()
            timelist.append(timezone.now().isoformat())
            w.setTimeList(timelist)
            w.save()
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def finish(request, workerID,dataset_name):
    
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == 2:
            validation_code = generate_code(w)
            startframe = w.frameNB - (w.frame_labeled*5)
            try:
                settings.UNLABELED.remove(startframe)
            except ValueError:
                pass
            return render(request, 'gtm_hit/finish.html', {'workerID': workerID, 'validation_code': validation_code, **context})
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'


def get_cuboids_2d(world_point, obj,new=False):
    rectangles = list()
    rect_id = str(int(world_point[0])) + "_" + str(int(world_point[1])
                                                   ) + "_" + uuid.uuid1().__str__().split("-")[0]
    # 

    if "object_size" in obj:
        object_size = obj["object_size"]
    else:
        object_size = [settings.HEIGHT, settings.RADIUS, settings.RADIUS]

    for cam_id in range(settings.NB_CAMS):
        # 
        try:
            cuboid = geometry.get_cuboid_from_ground_world(
                world_point, settings.CALIBS[cam_id], *object_size, obj.get("rotation_theta", 0))
            p1, p2 = geometry.get_bounding_box(cuboid)
        except ValueError:
            cuboid = []
            p1 = [-1, -1]
            p2 = [-1, -1]
        rectangle_as_dict = convert_rect_to_dict(
            (*p1, *p2), cuboid, cam_id, rect_id, world_point, object_size, obj.get("rotation_theta", 0))
        if "person_id" in obj:
            rectangle_as_dict["personID"] = obj["person_id"]
        rectangles.append(rectangle_as_dict)

    return rectangles


def click(request):
    if is_ajax(request):
        # try:
        x = int(float(request.POST['x']))
        y = int(float(request.POST['y']))
        obj = request_to_dict(request)
        cam = request.POST['canv']
        cam = int(re.findall('\d+', cam)[0]) - 1
        #
        worker_id = request.POST['workerID']
        dataset_name = request.POST['datasetName']
        if 0 <= cam < settings.NB_CAMS:
            feet2d_h = np.array([[x], [y], [1]])
            # 
            world_point = geometry.reproject_to_world_ground(
                feet2d_h, settings.CALIBS[cam], undistort=settings.UNDISTORTED_FRAMES)
            if "person_id" not in obj:
                obj["person_id"] = get_next_available_id(worker_id=worker_id,dataset_name=dataset_name)
            rectangles = get_cuboids_2d(world_point, obj)

            rect_json = json.dumps(rectangles)
            #
            return HttpResponse(rect_json, content_type="application/json")

        return HttpResponse("OK")


def move(request):
    if is_ajax(request):
        try:
            obj = request_to_dict(request)

            Xw = obj["Xw"]
            Yw = obj["Yw"]
            Zw = obj["Zw"]

            world_point = np.array([[Xw], [Yw], [Zw]])
            if request.POST['data[dir]'] == "down":
                world_point = world_point + \
                    np.array([[0], [-settings.STEPL], [0]])

            elif request.POST['data[dir]'] == "up":
                world_point = world_point + \
                    np.array([[0], [settings.STEPL], [0]])

            elif request.POST['data[dir]'] == "right":
                world_point = world_point + \
                    np.array([[settings.STEPL], [0], [0]])

            elif request.POST['data[dir]'] == "left":
                world_point = world_point + \
                    np.array([[-settings.STEPL], [0], [0]])

            else:
                return HttpResponse("Error")

            # 
            next_rect = get_cuboids_2d(world_point, obj)

            next_rect_json = json.dumps(next_rect)
            # 
            return HttpResponse(next_rect_json, content_type="application/json")

        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")


def action(request):
    # 
    #
    if is_ajax(request):
        try:

            obj = json.loads(request.POST["data"])

            obj = process_action(obj)
            Xw = obj["Xw"]
            Yw = obj["Yw"]
            Zw = obj["Zw"]

            world_point = np.array([[Xw], [Yw], [Zw]])
            world_point = geometry.move_with_mesh_intersection(world_point)
            next_rect = get_cuboids_2d(world_point, obj)

            next_rect_json = json.dumps(next_rect)
            # 
            return HttpResponse(next_rect_json, content_type="application/json")
        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")


def save(request):
    return save_db(request)
    # 
    if is_ajax(request):
        try:
            # 
            data = json.loads(request.POST['data'])
            frameID = request.POST['ID']
            wid = request.POST['workerID']
            # annotations = []
            # cols = ["rectID","personID","modified","Xw","Yw","Zw"]#,"a1","b1","c1","d1","a2","b2","c2","d2","a3","b3","c3","d3","a4","b4","c4","d4","a5","b5","c5","d5","a6","b6","c6","d6","a7","b7","c7","d7"]
            # for i in range(settings.NB_CAMS):
            #     cols += [f"a{i+1}", f"b{i+1}", f"c{i+1}", f"d{i+1}"]
            # annotations.append(cols)
            # for r in data:
            #     row = data[r]
            #     row.insert(0,r)
            #     annotations.append(row)
            # 
            if not os.path.exists("./gtm_hit/labels/"+settings.DSETNAME+"/" + wid + "/"):
                os.makedirs("./gtm_hit/labels/" +
                            settings.DSETNAME+"/" + wid + "/")
            with open("./gtm_hit/labels/"+settings.DSETNAME+"/" + wid + "/" + wid + "_" + frameID + '.json', 'w') as outFile:
                # create dir if not exist
                json.dump(data, outFile, sort_keys=True,
                          indent=4, separators=(',', ': '))

            if not os.path.exists("./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/" + wid + "/"):
                os.makedirs("./gtm_hit/static/gtm_hit/dset/" +
                            settings.DSETNAME+"/labels/" + wid + "/")
            with open("./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/" + wid + "/" + wid + "_" + frameID + '.json', 'w') as outFile:
                json.dump(data, outFile, sort_keys=True,
                          indent=4, separators=(',', ': '))
            return HttpResponse("Saved")
        except KeyError:
            return HttpResponse("Error")
    else:
        return("Error")


def load(request):
    return load_db(request)
    if is_ajax(request):
        try:
            frameID = request.POST['ID']
            wid = request.POST['workerID']
            rect_json = read_save(frameID, wid)
            return HttpResponse(rect_json, content_type="application/json")
        except (FileNotFoundError, KeyError):
            return HttpResponse("Error")
    return HttpResponse("Error")


def load_previous(request):
    if is_ajax(request):
        try:

            frameID = request.POST['ID']
            wid = request.POST['workerID']
            current_frame = int(frameID)
            closest = float('inf')
            diff = float('inf')

            for f in os.listdir("./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/" + wid + "/"):
                if f.endswith(".json"):
                    nb_frame = int((f.split('.')[0]).split('_')[1])
                    if nb_frame < current_frame:
                        if current_frame - nb_frame < diff:
                            diff = current_frame - nb_frame
                            closest = nb_frame
            if closest != float('inf'):
                frame = "0" * (8 - len(str(closest))) + str(closest)
                rect_json = read_save(frame, wid)
                return HttpResponse(rect_json, content_type="application/json")
        except (FileNotFoundError, KeyError):
            return HttpResponse("Error")
    return HttpResponse("Error")


def read_save(frameID, workerID):
    # 
    filename = "./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME + \
        "/labels/" + workerID + "/" + workerID + "_" + frameID + '.json'
    with open(filename, 'r') as loadFile:
        annotations = json.load(loadFile)
    return json.dumps(annotations)


def changeframe(request):
    context = RequestContext(request)
    #
    if is_ajax(request):
        frame = 0
        try:
            wID = request.POST['workerID']
            order = request.POST['order']
            frame_number = request.POST['frameID']
            increment = request.POST['incr']

            worker = Worker.objects.get(pk=wID)

            #   
            #worker.increaseFrame(1)
            
            timelist = worker.getTimeList()
            timelist.append(timezone.now().isoformat())
            worker.setTimeList(timelist)
            #validation_code = generate_code()
            # return render(request, 'gtm_hit/finish.html',{'workerID' : wID, 'validation_code': validation_code},context)
            if order == "next":
                inc = int(increment)
            elif order == "prev" and (int(frame_number) - int(increment)) >= 0:
                inc = -int(increment)
            else:
                return HttpResponse("Requested frame not existing")
            frame = int(frame_number) + inc
            worker.frame_labeled = frame
            worker.save()
            frame = "0" * (8 - len(str(frame))) + str(frame)
            response = {}
            response['frame'] = frame
            response['nblabeled'] = worker.frame_labeled
            worker.frameNB = frame
            worker.save()
            return HttpResponse(json.dumps(response))
        except KeyError:
            return HttpResponse("Error")
    else:
        return HttpResponse("Error")


def get_rect(closest):
    rects = []
    for i in range(settings.NB_CAMS):
        rdic = {}
        rdic['rectangleID'] = closest
        if closest in settings.RECT[i]:
            a, b, c, d, ratio = settings.RECT[i][closest]
        else:
            a, b, c, d, ratio = 0, 0, 0, 0, 0
        rdic['x1'] = a
        rdic['y1'] = b
        rdic['x2'] = c
        rdic['y2'] = d
        rdic['cameraID'] = i
        rdic['ratio'] = ratio
        rdic['xMid'] = (a + c) // 2
        rects.append(rdic)
    return rects


def registerWorker(workerID):
    w = Worker()
    w.workerID = workerID
    w.frameNB = settings.STARTFRAME % settings.NBFRAMES
    settings.STARTFRAME = settings.STARTFRAME + 100*settings.INCREMENT
    w.save()
    return w


def updateWorker(workerID, state):
    w = Worker.objects.get(pk=workerID)


def generate_code(worker):
    try:
        code = ValidationCode.objects.get(worker_id=worker)
    except ValidationCode.DoesNotExist:
        random_code = int(16777215 * rand.random())
        random_code = "{0:0>8}".format(random_code)
        while(random_code in settings.VALIDATIONCODES):
            random_code = int(16777215 * rand.random())
            random_code = "{0:0>8}".format(random_code)
        settings.VALIDATIONCODES.append(random_code)
        code = ValidationCode()
        code.validationCode = random_code
        code.worker = worker
        code.save()
    return code.validationCode


def tuto(request, workerID,dataset_name):
    context = RequestContext(request).flatten()
    
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state != 3:
            return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
        return render(request, 'gtm_hit/tuto.html', {'workerID': workerID, 'dset_name':dataset_name, **context})

    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def processTuto(request, workerID,dataset_name):
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == 3:
            w.state = 1
            timelist = [timezone.now().isoformat()]
            w.setTimeList(timelist)
            w.save()
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def processFinish(request):
    context = RequestContext(request)
    if request.is_ajax():
        try:
            wID = request.POST['workerID']

            w = Worker.objects.get(pk=wID)
            startframe = w.frameNB - w.frame_labeled
            # delete_and_load(startframe)
            return HttpResponse("ok")
        except KeyError:
            return HttpResponse("Error")
    else:
        return HttpResponse("Error")


def delete_and_load(startframe):
    toload = settings.LASTLOADED + 10
    # 1. remove frames
    sframe = startframe
    # 2. copy next frames
    for i in range(10):
        rm_frame = "0" * (8 - len(str(sframe))) + str(sframe)
        cp_frame = "0" * (8 - len(str(toload))) + str(toload)
        sframe = sframe + 1
        toload = toload + 1
        for j in range(settings.NB_CAMS):
            command = os.system(
                "rm gtm_hit/static/gtm_hit/frames/" + settings.CAMS[j] + "/" + rm_frame + ".png")
            command = os.system("cp gtm_hit/static/gtm_hit/day_2/annotation_final/" +
                                settings.CAMS[j] + "/begin/" + cp_frame + ".png gtm_hit/static/gtm_hit/frames/" + settings.CAMS[j] + "/")

    settings.LASTLOADED = settings.LASTLOADED + 10


def save_db(request):
    #set_trace()
    
    if is_ajax(request) and request.method == 'POST':
        try:
            data = json.loads(request.POST['data'])
            frame_id = request.POST['ID']
            worker_id = request.POST['workerID']
            # Check if the frame exists or create a new frame object
            worker, _ = Worker.objects.get_or_create(workerID=worker_id)
            dataset_name = request.POST['datasetName']
            #set_trace()
            dataset,_ = Dataset.objects.get_or_create(name=dataset_name)
            frame, created = MultiViewFrame.objects.get_or_create(
                frame_id=frame_id, worker=worker,undistorted=settings.UNDISTORTED_FRAMES,dataset=dataset)
            
            #delete all annotations for this frame (if not single person save)
            if len(data)>1:
                Annotation.objects.filter(frame=frame).delete()
            # 
            # Iterate through each annotation in the data and create an annotation object for it
            for annotation_data in data:
                person, _ = Person.objects.get_or_create(
                    person_id=annotation_data['personID'],worker=worker,dataset=dataset)
                # Create a new annotation object for the given person and frame
                try:
                    annotation = Annotation.objects.get(
                        person=person, frame=frame)
                    annotation.person = person
                    annotation.frame = frame
                    annotation.rectangle_id = annotation_data['rectangleID']
                    annotation.rotation_theta = annotation_data['rotation_theta']
                    annotation.Xw = annotation_data['Xw']
                    annotation.Yw = annotation_data['Yw']
                    annotation.Zw = annotation_data['Zw']
                    annotation.object_size_x = annotation_data['object_size'][0]
                    annotation.object_size_y = annotation_data['object_size'][1]
                    annotation.object_size_z = annotation_data['object_size'][2]
                except Annotation.DoesNotExist:
                    annotation = Annotation(
                        person=person,
                        frame=frame,
                        rectangle_id=annotation_data['rectangleID'],
                        rotation_theta=annotation_data['rotation_theta'],
                        Xw=annotation_data['Xw'],
                        Yw=annotation_data['Yw'],
                        Zw=annotation_data['Zw'],
                        object_size_x=annotation_data['object_size'][0],
                        object_size_y=annotation_data['object_size'][1],
                        object_size_z=annotation_data['object_size'][2],
                    )

                # Save the annotation object to the database
                annotation.save()
                save_2d_views(annotation)

            # # Create the directory for the labels if it doesn't exist
            # labels_directory = os.path.join('./gtm_hit/static/gtm_hit/dset/', settings.DSETNAME, 'labels', worker_id)
            # os.makedirs(labels_directory, exist_ok=True)

            # # Serialize the views associated with the frame and save them to a JSON file
            # views = View.objects.filter(frame=frame).prefetch_related('annotation_set__twod_views')
            # serialized_views = serializers.serialize('json', views)
            # with open(os.path.join(labels_directory, f'{worker_id}_{frame_id}.json'), 'w') as f:
            #     f.write(serialized_views)

            return HttpResponse("Saved")

        except KeyError:
            return HttpResponse("Error")

    else:
        return HttpResponse("Error")


def load_db(request):
    #
    if is_ajax(request):
        try:
            frame_id = int(request.POST['ID'])
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']

            frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)
            # 
            retjson = []
            camviews = View.objects.all()
            for camview in camviews:
                #
                a2l = serialize_annotation2dviews(
                    Annotation2DView.objects.filter(annotation__frame=frame, view=camview))
                retjson.append(a2l)
            #a2l = list(Annotation2DView.objects.filter(annotation__frame=frame, view=View.objects.get(view_id=0)).values())
            return HttpResponse(json.dumps(retjson), content_type="application/json")

            # Read the serialized views from the JSON file and deserialize them
            labels_directory = os.path.join(
                './gtm_hit/static/gtm_hit/dset/', settings.DSETNAME, 'labels', worker_id)
            with open(os.path.join(labels_directory, f'{worker_id}_{frame_id}.json'), 'r') as rect_json:
                # Deserialize the views from the serialized data
                views = serializers.deserialize('json', rect_json)

            # Iterate through each view and prefetch the related annotation and annotation2dview objects
            for view in views:
                view.object = view.object.prefetch_related(
                    'annotation_set__twod_views').select_related('frame')

            # Serialize the views and send them as a response
            serialized_views = serializers.serialize('json', views)
            return HttpResponse(serialized_views, content_type="application/json")

        except (Person.DoesNotExist, MultiViewFrame.DoesNotExist, FileNotFoundError, KeyError):
            return HttpResponse("Error")

    return HttpResponse("Error")

def change_id(request):
    #set_trace()
    if is_ajax(request):
        try:
            person_id = int(float(request.POST['personID']))
            new_person_id = int(float(request.POST['newPersonID']))
            frame_id = int(float(request.POST['frameID']))
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']

            frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)

            options = json.loads(request.POST['options'])
            success = change_annotation_id_propagate(person_id, new_person_id,frame,options)
            if success:
                return HttpResponse(JsonResponse({"message": "ID changed.","options":options}))
            else:
                return HttpResponse("Error")
        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")

def person_action(request):
    
    if is_ajax(request):
        try:
            person_id = int(float(request.POST['personID']))
            worker_id = request.POST['workerID']
            options = json.loads(request.POST['options'])
            dataset_name = request.POST['datasetName']

            person = Person.objects.get(person_id=person_id,worker_id=worker_id,dataset__name=dataset_name)
            #
            try:
                if "mark" in options:
                    person.annotation_complete = options["mark"]
                    person.save()
                    return HttpResponse(JsonResponse({"message": "Person annotation complete."}))
                if "delete" in options:
                    if "delete" in options and options["delete"]:
                        person.delete()
                    return HttpResponse(JsonResponse({"message": "Person deleted."}))
            except Person.DoesNotExist:
                return HttpResponse("Error")
        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")

def tracklet(request):
    
    if is_ajax(request):
        try:
            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']
            try:
                frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)
                person = Person.objects.get(person_id=person_id,worker_id=worker_id,dataset__name = dataset_name)
            except ValueError:
                HttpResponse("Error")
            multiview_tracklet = get_annotation2dviews_for_frame_and_person(
                frame, person)
            #
            return HttpResponse(json.dumps(multiview_tracklet), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")

def interpolate(request):
    if is_ajax(request):
        try:
            #
            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']
            #
            try:
                frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)
                person = Person.objects.get(person_id=person_id,worker_id=worker_id, dataset__name=dataset_name)
                message = interpolate_until_next_annotation(frame=frame, person=person)
            except ValueError:
                message = "Error while interpolating"
                return HttpResponse("Error", status=500)
            return HttpResponse(json.dumps({"message":message}), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")

def cp_prev_or_next_annotation(request):
    #set_trace()
    if is_ajax(request):
        try:
            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']
            try:
                frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)
                person = Person.objects.get(person_id=person_id,worker_id=worker_id, dataset__name=dataset_name)
                #delete person from current frame
                try:
                    annotation = Annotation.objects.get(person=person,frame=frame)
                    annotation.delete()
                except Annotation.DoesNotExist:
                    pass
                annotation = find_closest_annotations_to(person,frame,bidirectional=False)
                success = copy_annotation_to_frame(annotation, frame)
                if success:
                    return HttpResponse(JsonResponse({"message": "Annotation copied."}))
                else:
                    return HttpResponse("Error", status=500)
            except ValueError:
                HttpResponse("Error", status=500)
        except KeyError:
            return HttpResponse("Error",status=500)
def timeview(request):
    
    if is_ajax(request):
        try:
            #
            worker_id = request.POST['workerID']
            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))
            view_id = int(float(request.POST['viewID']))
            dataset_name=request.POST['datasetName']
            # Calculate the range of frame_ids for 5 frames before and 5 frames after the given frame

            FRAMES_NO = 35
            frame_id_start = 3150 #max(1, frame_id - FRAMES_NO)
            frame_id_end =  4425 #frame_id + FRAMES_NO
            
            # Filter the Annotation2DView objects using the calculated frame range and the Person object
            annotation2dviews = Annotation2DView.objects.filter(
                annotation__frame__frame_id__gte=frame_id_start,
                annotation__frame__frame_id__lte=frame_id_end,
                annotation__frame__worker__workerID=worker_id,
                annotation__frame__dataset__name=dataset_name,
                annotation__person__person_id=person_id,
                view__view_id = view_id,
                annotation__frame__undistorted = settings.UNDISTORTED_FRAMES
            ).order_by('annotation__frame__frame_id')
            timeviews =  serialize_annotation2dviews(annotation2dviews)
            # 
            return HttpResponse(json.dumps(timeviews), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")

import numpy as np

def reset_ac_flag(request):
    set_trace()
    if is_ajax(request):
        try:
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']
            for person in Person.objects.filter(worker_id=worker_id,dataset__name=dataset_name):
                person.annotation_complete = False
                person.save()
            return HttpResponse(json.dumps({"message":"ok"}), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")

def create_video(request):
    if is_ajax(request):
        try:
            dataset_name = request.POST['datasetName']
            worker_id = request.POST['workerID']
            create_video_invision(f"{worker_id}.mp4",15,dataset_name,worker_id)
            return HttpResponse(json.dumps({"message":"ok"}), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")
        
