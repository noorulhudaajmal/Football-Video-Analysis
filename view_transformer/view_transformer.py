import numpy as np
import cv2 as cv


class ViewTransformer:
    """
    Class for the transformation of tracked positions 
    from camera coordinates to real-world coordinates.
    """
    def __init__(self):
        
        # physical dimensions of the court
        court_width = 68
        court_length = 23.32
        
        # coordinates of four key points in the camera view 
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275], 
            [910, 260],
            [1640, 915]
        ])
        
        # corresponding four points in real-world coordinates
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])
        
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        
        # maps each point in pixel_vertices to the corresponding point in target_vertices
        self.perspective_transformer = cv.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
        
        
    def assign_transformed_position_to_tracks(self, tracks):
        """
        Iterates over tracking data to assign 
        transformed positions for each tracked object.
        """
        tracks = tracks.copy()
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, obj_track in track.items():
                    position = np.array(obj_track["position"])
                    trasformed_position = self.transform_point(position)                    
                    if trasformed_position is not None:
                        tracks[object][frame_num][track_id]["position_transformed"] = trasformed_position.squeeze().tolist()
                    else:
                        tracks[object][frame_num][track_id]["position_transformed"] = None
                        
        return tracks
                    
    
    def transform_point(self, position):
        """
        Transforms a single point from pixel 
        space to real-world court space.
        """
        point = (int(position[0]), int(position[1]))
        
        is_inside = cv.pointPolygonTest(contour=self.pixel_vertices, pt=point, measureDist=False) >= 0
        
        if is_inside:
            reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
            transformed_point = cv.perspectiveTransform(reshaped_point, self.perspective_transformer)
            
            return transformed_point.reshape(-1,2)
        else:
            return None