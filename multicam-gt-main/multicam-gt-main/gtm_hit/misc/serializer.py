def serialize_annotation2dviews(queryset):
    serialized_data = []
    for atdv in queryset:
        cuboid = None
        if atdv.cuboid_points:
            cuboid = [atdv.cuboid_points[0:2],
                        atdv.cuboid_points[2:4],
                        atdv.cuboid_points[4:6],
                        atdv.cuboid_points[6:8],
                        atdv.cuboid_points[8:10],
                        atdv.cuboid_points[10:12],
                        atdv.cuboid_points[12:14],
                        atdv.cuboid_points[14:16],
                        atdv.cuboid_points[16:18],
                        atdv.cuboid_points[18:20],
                        ]
        ann = atdv.annotation
        serialized_view = {
            'rectangleID': ann.rectangle_id,
            'cameraID': atdv.view.view_id,
            'person_id': ann.person.person_id,  # Include the person_id
            'annotation_complete': ann.person.annotation_complete,
            'validated': ann.validated,
            'creation_method': ann.creation_method,
            'object_size': ann.object_size,
            'rotation_theta': ann.rotation_theta,
            'Xw': ann.Xw,
            'Yw': ann.Yw,
            'Zw': ann.Zw,
            'x1': atdv.x1,
            'y1': atdv.y1,
            'x2': atdv.x2,
            'xMid': atdv.x1+(atdv.x2-atdv.x1)/2,
            'y2': atdv.y2,
            'cuboid': cuboid,
            'frameID': ann.frame.frame_id,
        }
        serialized_data.append(serialized_view)
    return serialized_data