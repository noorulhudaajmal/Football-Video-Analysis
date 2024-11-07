import os
import pickle
import cv2 as cv
import numpy as np
from utils.bbox_util import measure_distance


class CameraMovementEstimator:
    """
    Estimates and compensates for camera 
    movement in a sequence of video frames.
    """
    
    def __init__(self, frame):
        """
        Initializes the camera movement estimator with 
        feature detection parameters and optical flow settings.
        """

        grayscaled_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        features_mask = np.zeros_like(grayscaled_frame)
        
        features_mask[:, :20] = 1
        features_mask[:, -20:] = 1
        
        self.features = dict(
            maxCorners = 100, # maximum number of corners/features to detect.
            qualityLevel = 0.3, # A threshold value to filter weak features.
            minDistance = 3, # The minimum distance between detected features.
            blockSize = 7, # size of the neighborhood considered for each pixel in the detection process.
            mask = features_mask # mask used to restrict feature detection to specific areas of the frame.
        )
        
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 10, 0.03)
        )
        
        self.minimum_distance = 5
        
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Calculates the camera movement between 
        consecutive frames using optical flow.
        """
        
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        camera_movement = [[0, 0] for _ in range(len(frames))]

        
        old_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
        old_features = cv.goodFeaturesToTrack(image=old_gray, **self.features)
        
        for frame_num in range(1, len(frames)):
            frame_gray = cv.cvtColor(frames[frame_num], cv.COLOR_BGR2GRAY)
            new_features, _, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)
            
            max_distance = 0
            x, y = 0, 0
            for i, (new_feats, old_feats) in enumerate(zip(new_features, old_features)):
                new_features_pts = new_feats.ravel()
                old_features_pts = old_feats.ravel()
                
                distance = measure_distance(new_features_pts, old_features_pts)
                
                if distance > max_distance:
                    max_distance = distance
                    x = new_features_pts[0] - old_features_pts[0]
                    y = old_features_pts[1] - old_features_pts[1]
                    
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [x, y]
                
                old_features = cv.goodFeaturesToTrack(frame_gray, **self.features)    
                
            old_gray = frame_gray.copy()
            
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
            
        return camera_movement  
    
    
    def add_ajusted_position_to_tracks(self, tracks, camera_movements):
        """
        Adjusts object positions by compensating for 
        camera movement based on estimated movements.
        """

        tracks = tracks.copy()
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, obj_track in track.items():
                    position = obj_track["position"]
                    camera_movement_per_frame = camera_movements[frame_num]
                    adjusted_position = (position[0]-camera_movement_per_frame[0], position[1]-camera_movement_per_frame[1])
                    
                    tracks[object][frame_num][track_id]["position_adj"] = adjusted_position  
        return tracks
    
    
    def annotate_camera_movement(self, frames, camera_movements):
        """
        Annotates frames with arrows and stats 
        indicating the camera movement in each frame.
        """

        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Check if there is camera movement data for the current frame
            if camera_movements[frame_num]:
                x, y = camera_movements[frame_num]

                # Draw an arrow indicating camera movement direction
                arrow_start = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
                arrow_end = (arrow_start[0] + int(x), arrow_start[1] + int(y))
                cv.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), 3, tipLength=0.3)


                overlay = frame.copy()
                rect_x, rect_y, rect_width, rect_height = 10, frame.shape[0] - 120, 550, 60
                cv.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)

                alpha = 0.75
                frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Prepare text to display the camera movement
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_color = (255, 255, 255)
                text_movement = f"Camera Movement: x = {x:.2f}, y = {y:.2f}"
                text_position = (rect_x + 10, rect_y + 35)

                #text on the frame
                cv.putText(frame, text_movement, text_position, font, font_scale, text_color, font_thickness, cv.LINE_AA)

            output_frames.append(frame)

        return output_frames
        
            