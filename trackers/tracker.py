import os
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from utils.bbox_util import get_bbox_center, get_bbox_bottom_center


class Tracker:
    """
    Handles the tracking of objects in video 
    frames using a YOLO model and ByteTrack.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the tracker with a YOLO model.
        """
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()
    
    
    def detect_frames(self, frames: list) -> list:
        """
        Detect objects in each batch of 
        frame and return the bounding boxes.
        """
        batch_size = 5
        detection_results = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i: i+batch_size]
            batch_detection = self.model.predict(batch_frames, conf=0.1)
            
            detection_results += batch_detection
            # break
            
        return detection_results
    
    
    def get_object_tracks(self, frames: str, read_from_stub=False, stub_path=None) -> dict:
        """
        Tracks objects across frames, saving/loading 
        tracks from a stub if specified.
        """
        
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self.detect_frames(frames=frames)
        # ball: 0, goalkeeper:1, player: 2, refree: 3
        
        #tracking-store
        tracks = {
            "ball": [],
            "player": [],
            "referee": []
        }
        
        for frame_num, detection in enumerate(detections):
            classes = detection.names
            class_names_dict = dict((v, k) for k, v in classes.items())
            
            # yolo detection to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # replacing goalkeepr with player in tracking
            for cls_idx, cls_id in enumerate(detection_supervision.class_id):
                if classes[cls_id] == "goalkeeper":
                    detection_supervision.class_id[cls_idx] = class_names_dict["player"]
            
            # track objects
            tracks_detection = self.tracker.update_with_detections(detections=detection_supervision) # assigns tracking id to each bounding box and stores in variable tracker_id in Detection object
            
            tracks["ball"].append({})
            tracks["player"].append({})
            tracks["referee"].append({})
            
            for frame_detection in tracks_detection:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]
                class_name = classes[class_id]
                    
                tracks[f"{class_name}"][frame_num][track_id] = {"bbox": bounding_box}
                
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            
        return tracks
    
    
    def assign_position_to_tracks(self, tracks):
        """
        Assigns the position of each object in 
        the track based on its bounding box.
        """
        tracks = tracks.copy()
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, obj_track in track.items():
                    bbox = obj_track["bbox"]
                    if object == "ball":
                        position = get_bbox_center(bbox)
                    else:
                        position = get_bbox_bottom_center(bbox)
                    
                    tracks[object][frame_num][track_id]["position"] = position
        return tracks
                 
    
    
    def annotate_line_and_id(self, frame, bbox, color, track_id=None):
        """
        Draws a bottom line on the bounding box
        and annotates the track ID on the frame.
        """

        # bbox -> [x1, y1, x2, y2]
        # Bottom-left corner
        pt1 = (int(bbox[0]-8), int(bbox[3]))
        # Bottom-right corner
        pt2 = (int(bbox[2]+8), int(bbox[3]))

        # Draw the bottom line of the bounding box
        cv.line(
            img=frame,
            pt1=pt1,
            pt2=pt2,
            color=color,
            thickness=2,
            lineType=cv.LINE_4
        )
        
        if track_id:
            # Calculate the midpoint of the bottom line for the rectangle position
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = pt1[1] + 10  # The y-coordinate is the same as the bottom line
            
            # Define the rectangle size
            rect_width, rect_height = 40, 20  # Adjust these values as needed
            
            # Top-left and bottom-right points of the rectangle
            rect_tl = (mid_x - rect_width // 2, mid_y - rect_height // 2)
            rect_br = (mid_x + rect_width // 2, mid_y + rect_height // 2)
            
            # Draw a filled rectangle at the midpoint
            cv.rectangle(
                img=frame,
                pt1=rect_tl,
                pt2=rect_br,
                color=color,
                thickness=cv.FILLED
            )
            
            # Put the track_id text in the center of the rectangle
            font = cv.FONT_HERSHEY_SIMPLEX
            text_size = cv.getTextSize(str(track_id), font, 0.5, 1)[0]
            text_x = mid_x - text_size[0] // 2
            text_y = mid_y + text_size[1] // 2

            cv.putText(
                img=frame,
                text=str(track_id),
                org=(text_x, text_y),
                fontFace=font,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv.LINE_AA
            )
        
        return frame
    
        
    def annotate_pointer(self, frame, bbox, color):
        """
        Draws an inverted triangle above the 
        bounding box to point the object.
        """

        # bbox -> [x1, y1, x2, y2]
        # Calculate the top-center of the bounding box
        top_center_x = int((bbox[0] + bbox[2]) / 2)
        top_center_y = int(bbox[1])

        # Define the size of the triangle
        triangle_height = 15  # Height of the inverted triangle
        triangle_width = 20   # Width of the base of the inverted triangle

        # Define the three points of the inverted triangle
        pt1 = (top_center_x, top_center_y)  # Bottom point of the triangle (base of the bounding box)
        pt2 = (top_center_x - triangle_width // 2, top_center_y - triangle_height)  # Top-left point
        pt3 = (top_center_x + triangle_width // 2, top_center_y - triangle_height)  # Top-right point

        # Create a list of triangle points
        triangle_cnt = np.array([pt1, pt2, pt3])

        # Draw the inverted triangle using fillPoly
        cv.fillPoly(frame, [triangle_cnt], color)

        return frame
    
    
    def draw_team_ball_control(self, frame, frame_num, ball_controls):
        """
        Displays the ball control percentages for teams.
        """
        
        controls_till_frame = ball_controls[:frame_num+1]
        
        # Calculate ball control percentages
        team_1_control = controls_till_frame.count(1)/len(controls_till_frame) * 100
        team_2_control = controls_till_frame.count(2)/len(controls_till_frame) * 100
        
        # position and size of the semi-transparent rectangle
        overlay = frame.copy()
        rect_x, rect_y, rect_width, rect_height = 10, 10, 550, 120
        cv.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)
        
        alpha = 0.6  # Transparency factor
        frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # adding the ball control stats text
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        
        # Text for Team 1 and Team 2 control percentages
        text_team_1 = f"Team 1 Ball Control: {team_1_control:.2f}%"
        text_team_2 = f"Team 2 Ball Control: {team_2_control:.2f}%"
        
        # text positions
        text1_position = (rect_x + 20, rect_y + 50)
        text2_position = (rect_x + 20, rect_y + 85)
        
        # Adding text to the frame
        cv.putText(frame, text_team_1, text1_position, font, font_scale, text_color, font_thickness, cv.LINE_AA)
        cv.putText(frame, text_team_2, text2_position, font, font_scale, text_color, font_thickness, cv.LINE_AA)

        return frame
            
        
        
     
    def annotate_objects(self, frames, tracks, ball_controls):
        """
        Annotates the frames with tracked objects,
        and their IDs, ball control stats.
        """

        output_frames = []
        
        for frame_idx, frame in enumerate(frames):
            curr_frame = frame.copy()
            
            ball_tracks = tracks["ball"][frame_idx]
            player_tracks = tracks["player"][frame_idx]
            referee_tracks = tracks["referee"][frame_idx]
            
            for track_id, player in player_tracks.items():
                curr_frame = self.annotate_line_and_id(frame=curr_frame, bbox=player["bbox"], color=player.get("team_color", (0,0,255)), track_id=track_id)
                if player.get("has_ball", False):
                    curr_frame = self.annotate_pointer(frame=curr_frame, bbox=player["bbox"], color=(0,0,255))
            
            for track_id, refree in referee_tracks.items():
                curr_frame = self.annotate_line_and_id(frame=curr_frame, bbox=refree["bbox"], color=(0,225,255), track_id=None)
    
            for _, ball in ball_tracks.items():
                curr_frame = self.annotate_pointer(frame=curr_frame, bbox=ball["bbox"], color=(255,255,0))
            
            curr_frame = self.draw_team_ball_control(frame=curr_frame, frame_num=frame_idx, ball_controls=ball_controls)
            output_frames.append(curr_frame)
        
        return output_frames
    
    
    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions 
        between frames for smoother tracking.
        """
        
        tracking_id = next((list(d.keys())[0] for d in ball_positions if d), None)

        ball_positions = [x.get(tracking_id, {}).get("bbox", []) for x in ball_positions]
        ball_pos_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        ball_pos_df = ball_pos_df.interpolate()
        ball_pos_df = ball_pos_df.bfill()
        
        ball_positions = [{tracking_id: {"bbox": row.to_numpy().tolist()}} for _, row in ball_pos_df.iterrows()]
        
        return ball_positions
    
    
# Supervision Format: (xyxy=array(bounding-boxes), mask, confidence=array(conf-scores), class_id=array(classes-ids), tracker_id, data=dict("class_name":array(classes-names)))
