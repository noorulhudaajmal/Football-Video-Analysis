import sys
import cv2 as cv
import numpy as np
sys.path.append("../")
from utils.bbox_util import measure_distance, get_bbox_bottom_center


class SpeedAndDistanceEstimator:
    """
    This class is responsible for estimating the speed and 
    distance traveled by players in a given set of frames.    
    """
    
    def __init__(self):
        """
        Initializes the SpeedAndDistanceEstimator with default values.
        """
        self.frame_window = 5 # the number of frames to calculate speed/distance
        self.fps = 24 # frame rate
        
        
    def assign_speed_and_distance_to_tracks(self, tracks):
        """
        Calculates and assigns the speed (in km/h) and 
        total distance (in meters) to each player track 
        in the given frames.
        """
        tracks = tracks.copy()
        players_total_distance = {}
        
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referee":
                continue
            
            total_frames = len(object_tracks)
            for frame_num in range(0, total_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window, total_frames)
                
                for track_id, _ in object_tracks[frame_num].items():
                    if last_frame >= total_frames:
                        continue
                    
                    # if obj/player not in all frames in current frame-window
                    if track_id not in object_tracks[last_frame]:
                        continue
                    
                    start_position = object_tracks[frame_num][track_id]["position_transformed"]
                    end_position = object_tracks[last_frame][track_id]["position_transformed"]
                    
                    if start_position and end_position:
                        distance_covered = measure_distance(start_position, end_position)
                        time_elapsed = (last_frame - frame_num) / self.fps
                        
                        speed_mps = distance_covered / time_elapsed
                        speed_kph = speed_mps * 3.6
                        
                        
                        if object not in players_total_distance:
                            players_total_distance[object] = {}
                            
                        if track_id not in players_total_distance[object]:
                            players_total_distance[object][track_id] = 0
                        
                        players_total_distance[object][track_id] += distance_covered
                        
                        for frame_batch_indx in range(frame_num, last_frame):
                            if track_id not in tracks[object][frame_batch_indx]:
                                continue
                            
                            tracks[object][frame_batch_indx][track_id]["speed"] = speed_kph
                            tracks[object][frame_batch_indx][track_id]["distance"] = players_total_distance[object][track_id]
       
        return tracks
                            
                            
    def annotate_speed_and_distance(self, frames, tracks):
        """
        Annotates the given frames with the 
        calculated and distance for each player. 
        """
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            curr_frame = frame.copy()
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referee":
                    continue
                                        
                for track_id, track in object_tracks[frame_num].items():
                    if "speed" not in track:
                        continue
                    
                    speed = track["speed"]
                    distance = track["distance"]
                    bbox = track["bbox"]
                    bbox_bottom = get_bbox_bottom_center(bbox=bbox)
                    
                    
                    text_position_x = bbox_bottom[0] - 60
                    text_position_y = bbox_bottom[1] + 40 
                    
                    # Annotating speed and distance
                    speed_text = f"Speed = {speed:.1f} km/h"
                    dist_text = f"Distance = {distance:.1f} m"
                    
                    curr_frame = cv.putText(curr_frame, speed_text, (text_position_x, text_position_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)                    
                    curr_frame = cv.putText(curr_frame, dist_text, (text_position_x, text_position_y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                    
            output_frames.append(curr_frame)
                        
        return output_frames

                            
                                                
                    