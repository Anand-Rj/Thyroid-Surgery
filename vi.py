import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, measure
from skimage.color import label2rgb
from scipy import ndimage as nd

# Video file path
video_path = r'C:\Users\rjana\OneDrive\Desktop\ag.mp4'

# Open the video file
video = cv2.VideoCapture(video_path)
print(1)
# Initialize frame counter
i = 0

# Iterate over each frame of the video
while True:
    ret, frame = video.read()
    print(2)
    if not ret:
        break

    # Convert frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply green segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 90, 90), (75, 255, 255))
    print(3)
    area = cv2.countNonZero(mask)
    print("area of segmented area: ", area)
    if area > 40000:
        # # Perform morphological operations
        # closed_mask = nd.binary_closing(mask, np.ones((7, 7)))
        #
        # # Label connected components
        # label_image = measure.label(closed_mask)
        #
        # # Overlay labels on the original frame
        # image_label_overlay = label2rgb(label_image, image=frame_rgb)
        #
        # # Convert image_label_overlay to 8-bit depth
        # image_label_overlay_8bit = (image_label_overlay * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box around each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box in red

        # Save segmented image as JPG
        cv2.imwrite('image' + str(i) + '.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(4)
        # Increment frame counter
        i += 1

    else:
        print("Area is less than 10000. No need to save the image.")
print(5)
# Release the video object
video.release()