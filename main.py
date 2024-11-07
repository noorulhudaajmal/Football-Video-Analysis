from utils.video_utils import read_video, save_video
from ultralytics import YOLO
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import BallAssigner
from camera_momement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_calculator import SpeedAndDistanceEstimator


def main():
    # Read Video
    video_frames = read_video(file_path="input_videos/vid_1.mp4")
    
    tracker = Tracker(model_path="models/best.pt")
    
    tracks = tracker.get_object_tracks(frames=video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")
    
    # Assigning object positions
    tracks = tracker.assign_position_to_tracks(tracks=tracks)
    
    # Camera Movement Estimation
    cm_movement_estimator = CameraMovementEstimator(frame=video_frames[0])
    cm_movement_per_frame = cm_movement_estimator.get_camera_movement(frames=video_frames, read_from_stub=True, stub_path="stubs/camera_movement_stubs.pkl")
    tracks = cm_movement_estimator.add_ajusted_position_to_tracks(tracks=tracks, camera_movements=cm_movement_per_frame)
    
    # View Transforming
    view_transformer = ViewTransformer()
    tracks = view_transformer.assign_transformed_position_to_tracks(tracks=tracks)
    
    # Interpolate Ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(ball_positions=tracks["ball"])
    
    # Speed and Distance Estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    tracks = speed_and_distance_estimator.assign_speed_and_distance_to_tracks(tracks=tracks)
    
    # Assign Team Colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame=video_frames[0], player_detections=tracks["player"][0])
    
    for frame_num, player_track in enumerate(tracks["player"]):
        for id, track in player_track.items():
            team = team_assigner.get_player_team(frame=video_frames[frame_num],
                                                 player_bbox=track["bbox"],
                                                 player_id=id)
            tracks["player"][frame_num][id]["team"] = team
            tracks["player"][frame_num][id]["team_color"] = team_assigner.team_colors[team]
        
    # Assign ball to player
    ball_assigner = BallAssigner()
    ball_tracking_id = next((list(d.keys())[0] for d in tracks["ball"] if d), None)
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["player"]):
        ball_bbox = tracks["ball"][frame_num][ball_tracking_id]["bbox"]
        assigned_player = ball_assigner.assign_ball_to_player(players=player_track, ball_bbox=ball_bbox)
        
        if assigned_player != -1:
            tracks["player"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["player"][frame_num][assigned_player]["team"])    
        else:
            team_ball_control.append(team_ball_control[-1])  
        
    output_frames = tracker.annotate_objects(video_frames, tracks=tracks, ball_controls=team_ball_control)
    
    # Annotate Camera Movement
    output_frames = cm_movement_estimator.annotate_camera_movement(frames=output_frames, camera_movements=cm_movement_per_frame)
    
    
    # Annotate Speed and Distance
    output = speed_and_distance_estimator.annotate_speed_and_distance(frames=output_frames, tracks=tracks)
    
    #Save Video
    save_video(output, "output_videos/vid_1.avi")
    
    
if __name__ == "__main__":
    main()