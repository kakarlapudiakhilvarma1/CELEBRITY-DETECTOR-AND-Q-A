import cv2
from io import BytesIO
import numpy as np

def process_image(image_file):
    ## Temporary cache memory to save the input
    in_memory_file = BytesIO()  

    ## Save the input
    image_file.save(in_memory_file)  

    image_bytes = in_memory_file.getvalue()

    # Convert byte data into numpy arrays
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Convert the numpy array to actual open cv format image so it will usable by OpenCV (decode)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert image(bgr format) to greyscale - Because grey scale is best for the face detections.
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Load pre-trained model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect all the faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)


    ## Now there will be 2 cases here - 1. Either there os 'Zero' faces / 2. There will be one or more faces in the same image

    # To check if there is any face in the given image or not
    if len(faces) == 0:
        return image_bytes, None
    
    # Detect the one image even if we have multiple faces in the image
    largest_face = max(faces, key=lambda r:r[2] * r[3])

    (x,y,w,h) = largest_face

    # Draw a rectangle on detected face (original image)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255,0), 3)

    # Encode our image
    is_sucess, buffer = cv2.imencode(".jpg", img)

    return buffer.tobytes(), largest_face

