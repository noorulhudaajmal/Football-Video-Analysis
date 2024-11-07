import os
import random
import cv2 as cv
from ultralytics import YOLO
os.path.join("../")


def main():
    # Load the YOLO model
    model = YOLO("models/best.pt")

    # Load the image
    image = cv.imread("img1.png")
    if image is None:
        raise ValueError("Image not found or path is incorrect.")

    # Perform object detection
    detections = model.predict(image)
    
    # Retrieve bounding boxes
    bboxes = []
    for result in detections[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Convert bbox coordinates to integers
        bboxes.append((x1, y1, x2, y2))

    # Select 2-3 random bounding boxes
    selected_bboxes = random.sample(bboxes, min(3, len(bboxes)))

    # Crop and save each selected bounding box
    cropped_images = []
    for i, (x1, y1, x2, y2) in enumerate(selected_bboxes):
        cropped_img = image[y1:y2, x1:x2]
        cropped_images.append(cropped_img)
        
        # Optionally save the cropped image
        cv.imwrite(f"cropped_box_{i+1}.png", cropped_img)
    
    
    
if __name__ == "__main__":
    main()
    
