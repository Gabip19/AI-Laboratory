import cv2
import numpy as np

def apply_sepia_filter(image):
    # Sepia filter matrix
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])

    # Apply the sepia filter
    filtered_image = cv2.transform(image, sepia_filter)

    # Clip pixel values to the valid range (0-255)
    filtered_image = np.clip(filtered_image, 0, 255)

    return filtered_image.astype(np.uint8)

# Path to the input image
image_paths = []
for i in range(1, 51):
    image_path = f"data/default/image-{i}.jpg"
    image_paths.append(image_path)

# Load the image

for i in range(1, 51):
    image = cv2.imread(image_paths[i-1])

# Apply the sepia filter
    sepia_image = apply_sepia_filter(image)

# Save the filtered image
    output_path = f"data/sepia/image-{i}.jpg"
    cv2.imwrite(output_path, sepia_image)