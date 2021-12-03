import functools
import math
import airsim
import numpy as np
import cv2
import time
import os
import sys
import threading

MIN_DEPTH_METERS = 0
MAX_DEPTH_METERS = 230
HEIGHT=200
SPEED=4
IMAGE_STOP_FLAG = True
FINISH_FLAG = False

# # Africa, Landscape
# xmin, xmax, ymin, ymax = (-600, 600, -300, 300)
# NH, MSBuild2018
xmin, xmax, ymin, ymax = (-200, 200, -200, 200)
x, y = xmin, ymin

start_id = 0

output_path = ''
if len(sys.argv) < 2:
    print("\033[31m Not generating images now.\033[0m")
else:
    output_path = sys.argv[1]
    os.makedirs(output_path, exist_ok = True)
    os.makedirs(os.path.join(output_path, 'rgb'), exist_ok = True)
    os.makedirs(os.path.join(output_path, 'depth'), exist_ok = True)


def getVel(client):
    curr_acc = client.getImuData().linear_acceleration
    vel = math.sqrt(curr_acc.x_val**2 + curr_acc.y_val**2)
    return vel

def goPosition(client, x, y):
    global IMAGE_STOP_FLAG
    IMAGE_STOP_FLAG = True
    fly_task = client.moveToPositionAsync(x, y, -HEIGHT, SPEED)
    time.sleep(2)
    IMAGE_STOP_FLAG = False
    fly_task.join()
    time.sleep(2)

class ImageThread(threading.Thread):
    def __init__(self, start_id=0, delay=-1):
        super().__init__()
        self.client = airsim.MultirotorClient()
        if delay > 0:
            self.delay = delay
        else:
            self.delay = 2. / SPEED
        self.id = start_id
        self.running = False
    def run(self):
        while True:
            global IMAGE_STOP_FLAG
            global FINISH_FLAG
            time.sleep(self.delay)
            if FINISH_FLAG:
                return
            if IMAGE_STOP_FLAG:
                if self.running:
                    print(f'Photo stopped. Next id is {self.id}')
                    self.running = False
                continue
            self.running = True
            responses = self.client.simGetImages([
                airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("high_res", airsim.ImageType.DepthPlanar, True,
                                    False),
            ])
            for response in responses:
                if response.height < 1 or response.width < 0:
                    print(f'Photo {self.id} error')
                    continue
                if response.pixels_as_float:
                    # get numpy array
                    img1d = np.array(response.image_data_float, dtype=float)
                    img_depth = img1d.reshape(response.height, response.width, 1)
                    img_depth = np.flipud(img_depth)
                    # to visualize
                    img_depth = np.clip(img_depth, MIN_DEPTH_METERS, MAX_DEPTH_METERS)
                    img_depth = (img_depth * 255).astype(np.uint16)
                    path = os.path.join(output_path, f'depth/{self.id:04}.png')
                    airsim.write_png(os.path.normpath(path), img_depth)
                else:
                    # get numpy array
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    # reshape array to 4 channel image array H X W X 4
                    img_rgb = img1d.reshape(response.height, response.width, 3)
                    # original image is fliped vertically
                    img_rgb = np.flipud(img_rgb)
                    # write to png
                    path = os.path.join(output_path, f'rgb/{self.id:04}.png')
                    airsim.write_png(os.path.normpath(path), img_rgb)
            self.id += 1

class ShowThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.client = airsim.MultirotorClient()
        cv2.namedWindow('view', 0)
    def run(self):
        while True:
            response = self.client.simGetImages([
                airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)
            ])[0]
            # get numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            # reshape array to 4 channel image array H X W X 4
            img_rgb = img1d.reshape(response.height, response.width, 3)
            # original image is fliped vertically
            img_rgb = np.flipud(img_rgb)
            cv2.imshow('view', img_rgb)
            cv2.waitKey(50)


client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)

# prepare
client.takeoffAsync().join()
fly_task = client.moveToPositionAsync(0, 0, -HEIGHT, 10)
print(f"Fly to height: {HEIGHT}")
if len(output_path)  == 0:
    image_thread = ShowThread()
    image_thread.start()
else:
    image_thread = ImageThread(start_id=start_id)
    image_thread.start()

fly_task.join()
time.sleep(2)

go = functools.partial(goPosition, client)

# # Check range
# import ipdb
# SPEED=10
# ipdb.set_trace()

print(f'fly to start point ({x}, {y})')
fly_task = client.moveToPositionAsync(x, y, -HEIGHT, 10)
fly_task.join()
time.sleep(10)

# fly serpentine
while x < xmax:
    if y == ymin:
        y = ymax
    else:
        y = ymin
    print(f'fly to ({x}, {y})')
    go(x, y)
    x += HEIGHT
    print(f'fly to ({x}, {y})')
    go(x, y)
IMAGE_STOP_FLAG = True
FINISH_FLAG = True
print('finish')
exit(0)


"""
Settings of camera
{
    "ultra_res": {
        "CaptureSettings" : [
            {
                "ImageType" : 0,
                "Width" : 4320,
                "Height" : 2160
            }
        ],
        "X": 0.00, "Y": 0.00, "Z": 0.10,
        "Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0
    },
    "high_res": {
        "CaptureSettings" : [
            {
                "ImageType" : 0,
                "Width" : 1920,
                "Height" : 1080
            }
        ],
        "X": 0.00, "Y": 0.00, "Z": 0.10,
        "Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0
    },
    "low_res": {
        "CaptureSettings" : [
            {
                "ImageType" : 0,
                "Width" : 256,
                "Height" : 144
            }
        ],
        "X": 0.00, "Y": 0.00, "Z": 0.10,
        "Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0
    }
}
"""
