import os
import cv2 as cv


def read_video(file_path):
    cap = cv.VideoCapture(filename=file_path)
    frames = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
        
    return frames


def save_video(video_frames, output_file_path):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Check that there are frames to save
    if not video_frames:
        print("No frames to save.")
        return

    # VideoWriter setup
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    frame_height, frame_width = video_frames[0].shape[:2]
    out = cv.VideoWriter(output_file_path, fourcc, 24, (frame_width, frame_height))

    # Write frames to video
    for frame in video_frames:
        # Ensure each frame is in color format (3 channels)
        if len(frame.shape) == 2:  # grayscale
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        elif frame.shape[2] != 3:  # ensure 3-channel color
            raise ValueError("Frame format not supported for color video output.")
        
        out.write(frame)

    out.release()
    print(f"Video saved successfully to {output_file_path}")

