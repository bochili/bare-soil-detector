import os
from ultralytics import YOLO

model = YOLO("train_result/weights/best.pt")

# results = model("test_imgs/2.jpg",save=True, conf=0.5)

# read images files from test_imgs/ipc folder
for file in os.listdir("test_imgs/"):
    # judge if the file is a image file
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
        results = model("test_imgs/" + file,save=True, conf=0.1)
