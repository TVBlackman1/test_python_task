import cv2
import numpy as np
from keras.models import load_model

MODEL = load_model("model.h5")
image_no = 9

for image_no in range(1, 10):
    input_img = cv2.imread('test_images/{}.png'.format(image_no))

    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)

    image_resize = cv2.resize(gray,(28,28))
    image_resize = image_resize/255
    image_reshape = np.reshape(image_resize,[1,28,28])

    prediction = MODEL.predict(image_reshape)
    print(f"The Predicted digit is {np.argmax(prediction)}")