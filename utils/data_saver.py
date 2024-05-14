import os
import pickle

import cv2
import json
from datetime import datetime
import threading
import numpy as np


class DataSaver:
    def __init__(self, root_path):
        self.root_path = root_path
        self.dirs = ['radar', 'cam0', 'cam1']
        for dir in self.dirs:
            os.makedirs(os.path.join(root_path, dir), exist_ok=True)

    def save_data(self, data_dict, image0, image1):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
        # Save data_dict in 'radar' directory

        # print(data_dict['numDetectedPoints'])
        with open(os.path.join(self.root_path, 'radar', f'{timestamp}.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)
        # Save images in 'cam0' and 'cam1' directories
        if image0 is not None:
            cv2.imwrite(os.path.join(self.root_path, 'cam0', f'{timestamp}.jpg'), image0)
        if image1 is not None:
            cv2.imwrite(os.path.join(self.root_path, 'cam1', f'{timestamp}.jpg'), image1)


# class DataSaver:
#     def __init__(self, root_path):
#         self.root_path = root_path
#         self.dirs = ['radar', 'cam0', 'cam1']
#         for dir in self.dirs:
#             os.makedirs(os.path.join(root_path, dir), exist_ok=True)
#
#     def save_data(self, data_dict, image0, image1):
#         # Start a new thread to save the data
#         threading.Thread(target=self._save_data, args=(data_dict, image0, image1)).start()
#
#     def _save_data(self, data_dict, image0, image1):
#         timestamp = datetime.now().strftime('%H%M%S.%f')[:-3]
#         # Save data_dict in 'radar' directory
#         with open(os.path.join(self.root_path, 'radar', f'{timestamp}.pkl'), 'wb') as f:
#             pickle.dump(data_dict, f)
#         # Save images in 'cam0' and 'cam1' directories
#         if image0 is not None:
#             cv2.imwrite(os.path.join(self.root_path, 'cam0', f'{timestamp}.jpg'), image0)
#         if image1 is not None:
#             cv2.imwrite(os.path.join(self.root_path, 'cam1', f'{timestamp}.jpg'), image1)

if __name__ == '__main__':
    # Usage example
    saver = DataSaver('data/my_data/0')
    data_dict = {'key': 'value'}  # Replace with your actual data
    image = np.zeros((480, 640, 3))  # Replace with your actual image
    saver.save_data(data_dict, image, image)
