import cv2
import numpy as np
from IPython.display import clear_output
import time
import json
import uuid
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person,Dataset
from django.conf import settings
from gtm_hit.misc.db import save_2d_views
import os
from ipdb import set_trace
import os.path as osp
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def transfer_complete_annotations(dataset_name,worker_from="SYNC17APR0908",worker_to="IVANA"):
    dataset= Dataset.objects.get(name=dataset_name)
    worker = Worker.objects.get(workerID=worker_from)
    persons_complete = Person.objects.filter(dataset=dataset, annotation_complete=True, worker=worker)

    for person_complete in persons_complete:
        person_complete_worker = person_complete.worker
        other_worker = Worker.objects.get(workerID=worker_to)
        person_complete_annotations = Annotation.objects.filter(person=person_complete)

        #print (f"Syncing {person_complete.person_id} from {person_complete.worker.workerID}...")
        print("***")
        try:
            person_other_worker = Person.objects.get(person_id=person_complete.person_id, worker=other_worker, dataset=dataset)
        except Person.DoesNotExist:
            continue
        print(f"Syncing {person_complete.person_id} from {person_complete.worker.workerID} to {other_worker.workerID}...")
        for complete_annotation in person_complete_annotations:
            try:
                target_annotation = Annotation.objects.get(person=person_other_worker, frame__frame_id=complete_annotation.frame.frame_id, frame__undistorted=complete_annotation.frame.undistorted, frame__worker=other_worker, frame__dataset=dataset)
            except Annotation.DoesNotExist:
                target_annotation = Annotation()
                frame,_ = MultiViewFrame.objects.get_or_create(frame_id=complete_annotation.frame.frame_id, undistorted=complete_annotation.frame.undistorted, worker=other_worker, dataset=dataset)
                target_annotation.frame = frame
                target_annotation.person = person_other_worker

            target_annotation.rectangle_id = complete_annotation.rectangle_id
            target_annotation.creation_method = f"sync_{person_complete_worker.workerID}__{complete_annotation.creation_method}"
            target_annotation.validated = complete_annotation.validated
            target_annotation.rotation_theta = complete_annotation.rotation_theta
            target_annotation.Xw = complete_annotation.Xw
            target_annotation.Yw = complete_annotation.Yw
            target_annotation.Zw = complete_annotation.Zw
            target_annotation.object_size_x = complete_annotation.object_size_x
            target_annotation.object_size_y = complete_annotation.object_size_y
            target_annotation.object_size_z = complete_annotation.object_size_z
            target_annotation.save()
            save_2d_views(target_annotation)
        person_other_worker.annotation_complete = True
        person_other_worker.save()


def sync_annotations(dataset_name):
    dataset= Dataset.objects.get(name=dataset_name)
    workers = Worker.objects.all()
    persons_complete = Person.objects.filter(dataset=dataset, annotation_complete=True)

    for person_complete in persons_complete:
        person_complete_worker = person_complete.worker
        other_workers = [worker for worker in workers if worker != person_complete_worker]
        person_complete_annotations = Annotation.objects.filter(person=person_complete)

        #print (f"Syncing {person_complete.person_id} from {person_complete.worker.workerID}...")
        print("***")
        for other_worker in other_workers:
            try:
                person_other_worker = Person.objects.get(person_id=person_complete.person_id, worker=other_worker, dataset=dataset)
                if person_other_worker.annotation_complete:
                    continue
            except Person.DoesNotExist:
                continue
            print(f"Syncing {person_complete.person_id} from {person_complete.worker.workerID} to {other_worker.workerID}...")
            for complete_annotation in person_complete_annotations:
                try:
                    target_annotation = Annotation.objects.get(person=person_other_worker, frame__frame_id=complete_annotation.frame.frame_id, frame__undistorted=complete_annotation.frame.undistorted, frame__worker=other_worker, frame__dataset=dataset)
                except Annotation.DoesNotExist:
                    target_annotation = Annotation()
                    frame,_ = MultiViewFrame.objects.get_or_create(frame_id=complete_annotation.frame.frame_id, undistorted=complete_annotation.frame.undistorted, worker=other_worker, dataset=dataset)
                    target_annotation.frame = frame
                    target_annotation.person = person_other_worker

                target_annotation.rectangle_id = complete_annotation.rectangle_id
                target_annotation.creation_method = f"sync_{person_complete_worker.workerID}__{complete_annotation.creation_method}"
                target_annotation.validated = complete_annotation.validated
                target_annotation.rotation_theta = complete_annotation.rotation_theta
                target_annotation.Xw = complete_annotation.Xw
                target_annotation.Yw = complete_annotation.Yw
                target_annotation.Zw = complete_annotation.Zw
                target_annotation.object_size_x = complete_annotation.object_size_x
                target_annotation.object_size_y = complete_annotation.object_size_y
                target_annotation.object_size_z = complete_annotation.object_size_z
                target_annotation.save()
                save_2d_views(target_annotation)
            person_other_worker.annotation_complete = True
            person_other_worker.save()

def preprocess_invision_data(path,worker_id,dataset_name,range_start=0,range_end=5000):
    #set_trace()


    #temp fix
    fd_matrix_path = "gtm_hit/static/gtm_hit/labels/framcnt.pkl"
    
    fd_matrix = np.load(fd_matrix_path,allow_pickle=True)
    fd_matrix = np.vstack((np.zeros(fd_matrix.shape[0]),fd_matrix.T)).astype(int)
    # tempfix end
    cam_id_mat = np.mgrid[1:3,1:5].reshape(2,-1).T
    cam_id_keys = [f"cam_{cam_id[0]}_{cam_id[1]}" for cam_id in cam_id_mat]
    cam_id_keys_to_idx = dict([cam_id_keys[i],i] for i in range(len(cam_id_keys)))
    output_data = {}

    for cam_id in cam_id_mat:
        cam_id = tuple(cam_id)
        cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"
        f = open(osp.join(path,f'cam_{cam_id[0]}_{cam_id[1]}.json'))
        data = json.load(f)
        data = dict([(frame["frameId"],frame) for frame in data])
        data["id"]=cam_id
        output_data[cam_id_key] = data
        print( "Loaded json output data for camera",cam_id, "for", len(data), "frames.")

    frame_id=0
    
    print("\n\n","*"*50)
    print("Working with worker_id:",worker_id)
    print("Working with dataset_name:",dataset_name)
    print("Working with undistorted frames:",settings.UNDISTORTED_FRAMES)
    print("Processing data...")
    print("*"*50,"\n\n")
    worker, _ = Worker.objects.get_or_create(workerID=worker_id)
    dataset,_ = Dataset.objects.get_or_create(name=dataset_name)

    for frame_idx in range(range_start,range_end,settings.INCREMENT):
        print(frame_idx,fd_matrix[frame_idx])
        #clear_output(wait=True)
        obj_set = set()
        for i,cam_id in enumerate(cam_id_mat):
            frame_id = fd_matrix[frame_idx][i]
            cam_id = tuple(cam_id)
            cam_id_key = f"cam_{cam_id[0]}_{cam_id[1]}"
            cam_data = output_data[cam_id_key]
            if frame_id in cam_data:
                if "tracked_locations" not in cam_data[frame_id] or cam_data[frame_id]["tracked_locations"] is None:
                    continue
                for tdet in cam_data[frame_id]["tracked_locations"]:
                    track_id = tdet["trackId"]
                    #add metadata to the tracked location
                    tdet["cam_id"] = cam_id_key
                    tdet["frame_id"] = frame_id
                    if track_id not in obj_set:
                        obj_set.add(track_id)
                        process_tracked_location(tdet,worker,frame_idx,dataset)
        cls()

def process_tracked_location(tdet,worker,frame_id,dataset):
    world_coords = tdet["cuboidToWorldTransform"] @ np.array([0,0,0,1]).reshape(-1,1)
    world_coords = world_coords[:3,0].T
    rect_id = uuid.uuid4().__str__().split("-")[-1]
    #assumes that cuboid only rotates around z axis: ([[np.cos(theta),-np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0,0,1]
    rotation_theta = np.arccos(tdet["cuboidToWorldTransform"][0][0])

    annotation_data= {"Xw":world_coords[0],
            "Yw":world_coords[1],
            "Zw":world_coords[2],
            "object_size":tdet["objectSize"][::-1],
            "personID":tdet["trackId"],
            "rotation_theta":rotation_theta,
            "rectangleID":rect_id
            }
    
            
    frame, created = MultiViewFrame.objects.get_or_create(
        frame_id=frame_id, worker=worker,undistorted=settings.UNDISTORTED_FRAMES,dataset=dataset)
    
    person, _ = Person.objects.get_or_create(
        person_id=annotation_data['personID'],worker=worker,dataset=dataset)
    if person.annotation_complete==True:
        print("Person",person.person_id, "marked as complete. Skipping...")
        return
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
        annotation.creation_method = f"imported_{tdet['cam_id']}_frame_{tdet['frame_id']}"
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
            creation_method = f"imported_{tdet['cam_id']}_frame_{tdet['frame_id']}"
        )

    # Save the annotation object to the database
    try:
        annotation.save()
        try:
            save_2d_views(annotation)
        except Exception as e:
            print("Error saving 2d views:", e)
    except Exception as e:
        print("Error saving annotation:", e)
    
    print("Saved annotation for person", person.person_id, "in frame", frame.frame_id)