import math
import os
import sys
import threading

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtGui import QFont
import pickle
from matplotlib import pyplot as plt

import time
import serial
import serial.tools.list_ports
from sklearn.cluster import DBSCAN

from gui_parser import uartParser
from gui_common import *
from tracker import Sort
from utils.camera import Camera
from utils.data_saver import DataSaver

DEMO_NAME_OOB = 'SDK Out of Box Demo'

from parseFrame import parseStandardFrame


class RadarDataReader:
    def __init__(self, directory='data/r1f_t3_30fps_vwalk', raw=False):
        self.directory = directory
        self.frame_per_file = 1
        self.files = [f for f in os.listdir(directory) if f.startswith('pHistBytes_') and f.endswith('.bin')]
        self.files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        self.gen = self.generator()
        self.raw = raw

    def generator(self):
        while True:
            for file in self.files:
                try:
                    with open(os.path.join(self.directory, file), 'rb') as f:
                        if self.raw:
                            yield f.read()
                        else:
                            data = parseStandardFrame(f.read())
                            # print(data['frameNum'])
                            yield data
                except:
                    pass

    def next(self):
        return next(self.gen)


class mmGateDataReader:
    def __init__(self, directory='data/people-gait/room1/1/fixed_route/77Ghz_radar'):
        self.directory = directory
        self.frame_per_file = 1
        self.files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        self.files.sort()
        self.column_names = ['Frame #', '# Obj', 'X', 'Y', 'Z', 'Doppler', 'Intensity', 'y', 'm', 'd', 'h', 'm', 's']
        self.gen = self.generator()

    def generator(self):
        while True:
            for file in self.files:
                try:
                    df = pd.read_csv(os.path.join(self.directory, file))
                    frame_numbers = df['Frame #'].unique()
                    for frame in frame_numbers:
                        df_frame = df[df['Frame #'] == frame]
                        # dict_keys(['error', 'frameNum', 'pointCloud', 'numDetectedPoints', 'rangeProfile'])
                        frame_data = {}
                        frame_data['error'] = None
                        frame_data['frameNum'] = frame
                        frame_data['pointCloud'] = df_frame[['X', 'Y', 'Z']].to_numpy()
                        frame_data['numDetectedPoints'] = df_frame['# Obj'].to_numpy()[0]
                        frame_data['rangeProfile'] = None
                        yield frame_data

                except:
                    pass

    def next(self):
        return next(self.gen)

# =======================================================

class clustering:
    def __init__(self):
        self.alpha = 0.25
        self.dbscan = DBSCAN(eps=0.25, min_samples=1, metric=self.metric)

    def metric(self, p1, p2):
        squared_distance = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + self.alpha * (p1[2] - p2[2]) ** 2
        distance = np.sqrt(squared_distance)
        return distance

    def predict(self, data):
        labels = self.dbscan.fit_predict(data)
        cluster_data = []
        for i in np.unique(labels):
            if i == -1: continue
            cluster_data.append(data[np.where(labels == i)[0]])
        return cluster_data


class Tracker:
    # TODO: UPDATE WITH KALMAN FILTER
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.current_frame = -1
        self.current_frame_obj = []
        self.previous_frame_obj = []
        self.match_dist = 2

    def track(self, detected_objs):
        self.current_frame+=1
        self.current_frame_obj = detected_objs
        if self.current_frame <= 2:
            for pt in self.current_frame_obj:
                for pt2 in self.previous_frame_obj:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1], pt2[2] - pt[2])

                    if distance < self.match_dist:
                        self.tracks[self.next_id] = pt
                        self.next_id += 1
        else:
            tracks_copy = self.tracks.copy()
            current_frame_obj_copy = self.current_frame_obj.copy()

            for object_id, pt2 in tracks_copy.items():
                object_exists = False
                for pt in current_frame_obj_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1], pt2[2] - pt[2])

                    # Update IDs position
                    if distance < self.match_dist:
                        self.tracks[object_id] = pt
                        object_exists = True
                        if pt in self.current_frame_obj:
                            self.current_frame_obj.remove(pt)
                        continue

                # Remove IDs lost
                if not object_exists:
                    self.tracks.pop(object_id)

            # Add new IDs found
            for pt in self.current_frame_obj:
                self.tracks[self.next_id] = pt
                self.next_id += 1
        return self.tracks



def merge_dicts(dict_list):
    # Initialize an empty dictionary to store the merged data
    merged_dict = dict(key_list=['error', 'frameNum', 'pointCloud', 'numDetectedPoints', 'rangeProfile'])
    pointCloud = np.concatenate([dict_list[i]['pointCloud'] for i in range(-1, len(dict_list)*-1-1, -1)], axis=0)
    numDetectedPoints = sum([dict_list[i]['numDetectedPoints'] for i in range(-1, len(dict_list)*-1-1, -1)])
    rangeProfile = dict_list[-1]['rangeProfile']

    merged_dict['error'] = dict_list[-1]['error']
    merged_dict['frameNum'] = dict_list[-1]['frameNum']
    merged_dict['error'] = dict_list[-1]['error']
    merged_dict['pointCloud'] = pointCloud
    merged_dict['numDetectedPoints'] = numDetectedPoints
    merged_dict['rangeProfile'] = rangeProfile

    return merged_dict


def rotate_point_cloud(points, angles_degrees):
    # Unpack the angles
    angle_x, angle_y, angle_z = angles_degrees

    # Convert angles from degrees to radians
    angle_x_rad = np.deg2rad(angle_x)
    angle_y_rad = np.deg2rad(angle_y)
    angle_z_rad = np.deg2rad(angle_z)

    # Rotation matrix for the X-axis
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
        [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]
    ])

    # Rotation matrix for the Y-axis
    ry = np.array([
        [np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
        [0, 1, 0],
        [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]
    ])

    # Rotation matrix for the Z-axis
    rz = np.array([
        [np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
        [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    rotation_matrix = rz @ ry @ rx

    # Apply the combined rotation matrix to the points
    rotated_points = points @ rotation_matrix.T

    return rotated_points

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, radar_parser, parent=None):
        super(MainWindow, self).__init__(parent)
        self.active_threads = 0
        self.ROOM_SIZE = [2.75, 2.75, 2.5]
        self.RADAR_POS = [1.5, 0.5, 1]
        self.RADAR_ANGLE = [0, 0, 0]

        self.ROOM_SIZE = [20, 20, 2.5]
        self.RADAR_POS = [0, 0, 1]
        self.radar_parser = radar_parser

        # Create a GLViewWidget
        self.glview = gl.GLViewWidget()
        self.glview.show()

        self.add_grid(area=self.ROOM_SIZE, scale=1.0, translate=False)
        self.add_axes(size=self.ROOM_SIZE)
        self.add_radar(position=self.RADAR_POS, angle=self.RADAR_ANGLE)

        # Set the GLViewWidget as the central widget of the main window
        self.setCentralWidget(self.glview)

        # Set the initial size of the window
        self.resize(800, 600)  # Width, Height

        # Create a timer to update the scatter plot
        self.data_reader = RadarDataReader()
        # self.data_reader = mmGateDataReader()

        self.plot_history = []
        self.radar_data_history = []
        self.max_iterations = 1

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        # self.timer.timeout.connect(self.dummy_update())
        self.timer.start(100)  # Update every 100 ms

        self.clustering = clustering()
        cmap = plt.get_cmap('gist_rainbow')
        self.colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        self.history = {'scatter_plot': [], 'bbox': [], 'center_point': []}
        self.tracker = Sort()

        self.save = True
        self.cam0 = Camera()
        self.cam1 = None
        self.save_path = self.create_folder('data/my_data')
        self.data_saver = DataSaver(self.save_path)

        # Timer to stop the application after 1 minute
        self.stop_timer = QtCore.QTimer()
        self.stop_timer.singleShot(1000 * 10, sys.exit)  # Close
        self.c = 0
        self.save_threads = []


    def create_folder(self, path):
        # If the path doesn't exist, create the first folder
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, '0'))
            return os.path.join(path, '0')

        # Get the list of existing folders
        folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

        # If there are no folders, create the first one
        if not folders:
            os.makedirs(os.path.join(path, '0'))
            return os.path.join(path, '0')

        # Find the maximum folder number
        max_folder = max(int(folder) for folder in folders if folder.isdigit())

        # Create a new folder with a number one higher than the maximum
        new_folder = str(max_folder + 1)
        os.makedirs(os.path.join(path, new_folder))

        return os.path.join(path, new_folder)


    def add_radar(self, position=None, angle=None):
        # Create a GLBoxItem
        if position is None:
            position = [0, 0, 1]
        if angle is None:
            position = [0, 0, 0]
        box = gl.GLBoxItem()

        box.setSize(x=0.07, y=0.01, z=0.08)
        box.setColor(QtGui.QColor(255, 0, 0))  # Red color
        box.translate(position[0], position[1], position[2])
        box.rotate(angle[0], 1, 0, 0)
        box.rotate(angle[1], 0, 1, 0)
        box.rotate(angle[2], 0, 0, 1)

        self.glview.addItem(box)

        # box = gl.GLBoxItem()
        # box.setSize(x=1, y=1, z=2)
        # box.setColor(QtGui.QColor(255, 255, 0))  # Red color
        # box.translate(1, 1, 1)
        # self.glview.addItem(box)

    def add_grid(self, area=None, scale=1.0, translate=True):
        if area is None:
            area = [10, 10, 10]
        grid = gl.GLGridItem()
        grid.setSize(x=area[0], y=area[1], z=area[2])
        grid.setSpacing(x=scale, y=scale, z=scale)
        if translate:
            grid.translate(area[0]/2, area[1]/2, 0)
        self.glview.addItem(grid)

    def add_axes(self, size=None):

        # Create custom axes with labels
        if size is None:
            size = [6, 8.5, 8]
        x_axis = gl.GLAxisItem()
        x_axis.setSize(size[0], 0, 0)
        self.glview.addItem(x_axis)
        y_axis = gl.GLAxisItem()
        y_axis.setSize(0, size[1], 0)
        self.glview.addItem(y_axis)
        z_axis = gl.GLAxisItem()
        z_axis.setSize(0, 0, size[2])
        self.glview.addItem(z_axis)

        font = QFont()
        font.setPointSize(5)  # Set the font size to 10

        for i in np.arange(0, size[0]+0.5, 0.5):
            x_label = gl.GLTextItem(pos=(i, 0, 0))
            x_label.setData(text=str(i), font=font)
            self.glview.addItem(x_label)

        for i in np.arange(0, size[1]+0.5, 0.5):
            y_label = gl.GLTextItem(pos=(0, i, 0))
            y_label.setData(text=str(i), font=font)
            self.glview.addItem(y_label)

    def update(self):
        for i in self.history['scatter_plot']:
            self.glview.removeItem(i)
        self.history['scatter_plot'] = []
        for i in self.history['bbox']:
            self.glview.removeItem(i)
        self.history['bbox'] = []
        for i in self.history['center_point']:
            self.glview.removeItem(i)
        self.history['center_point'] = []


        # Get the next data point from the dummyData instance
        radarOutputDict = self.data_reader.next()
        # radarOutputDict = self.radar_parser.readAndParseUartDoubleCOMPort()
        # dict_keys(['error', 'frameNum', 'pointCloud', 'numDetectedPoints', 'rangeProfile'])
        # point cloud : Each point has the following: X, Y, Z, Doppler, SNR, Noise, Track index

        if not radarOutputDict:
            return
        self.c+=1
        print(self.c)
        # SAVE DATA
        if self.save:
            frame_0, frame_1 = None, None
            if self.cam0 is not None:
                frame_0 = self.cam0.next()
            if self.cam1 is not None:
                frame_1 = self.cam1.next()
            self.data_saver.save_data(radarOutputDict, frame_0, frame_1)


        # self.radar_data_history.append(radarOutputDict)
        # if len(self.radar_data_history) > 3:
        #     self.radar_data_history.pop(0)
        # radarOutputDict = merge_dicts(self.radar_data_history)
        #
        # frame_number = radarOutputDict['frameNum']
        # point_cloud_data = radarOutputDict['pointCloud']
        # data = point_cloud_data[:, :3]
        #
        # data += np.array(self.RADAR_POS)
        # data = rotate_point_cloud(data, self.RADAR_ANGLE)
        # condition = np.all((data >= 0) & (data <= self.ROOM_SIZE), axis=1)
        # # data = data[condition]
        # data = self.clustering.predict(data) # (N, n, 3)
        # det_obj = []
        #
        # for i, cluster in enumerate(data):
        #     scatter = gl.GLScatterPlotItem(pos=cluster, color=(0, 1, 0, 1), size=5)
        #     self.glview.addItem(scatter)
        #     self.history['scatter_plot'].append(scatter)
        #
        #     center = np.mean(cluster, axis=0)
        #     x_min, y_min, z_min = center[0]-0.375, center[1]-0.375, 0
        #     x_max, y_max, z_max = center[0]+0.375, center[1]+0.375, 2
        #     result = [x_min, y_min, x_max, y_max]
        #     det_obj.append(result)
        # det_obj = np.array(det_obj)
        # tracks = self.tracker.update(det_obj)
        # for track in tracks:
        #     x_min, y_min, x_max, y_max,  id = track
        #     center = np.array([(x_min+x_max)/2, (y_min+y_max)/2, 1])
        #     size = np.array([0.75, 0.75, 2.0])
        #     c = tuple(int(x * 255) for x in self.colors[int(id)%len(self.colors)][:3])
        #     box = gl.GLBoxItem(size=QtGui.QVector3D(*size), color=QtGui.QColor(*c))
        #     box.setSize(x=size[0], y=size[1], z=size[2])
        #     box.translate(*[center[0]-0.375, center[1]-0.375, 0])
        #     self.glview.addItem(box)
        #     self.history['bbox'].append(box)
        #
        #     center_point = gl.GLScatterPlotItem(pos=center,
        #                                         color=self.colors[int(id)%len(self.colors)], size=5)
        #     self.glview.addItem(center_point)
        #     self.history['center_point'].append(center_point)


if __name__ == "__main__":
    parser = uartParser(type=DEMO_NAME_OOB)
    # 1. Find all Com Ports
    serialPorts = list(serial.tools.list_ports.comports())
    cli_port, data_port = None, None
    for port in serialPorts:
        if (CLI_XDS_SERIAL_PORT_NAME in port.description or CLI_SIL_SERIAL_PORT_NAME in port.description):
            print(f'CLI COM Port found: {port.device}')
            cli_port = port.device

        elif (DATA_XDS_SERIAL_PORT_NAME in port.description or DATA_SIL_SERIAL_PORT_NAME in port.description):
            print(f'Data COM Port found: {port.device}')
            data_port = port.device

    if cli_port is None or data_port is None:
        print("PORTS NOT FOUND")
        exit(-1)
    parser.connectComPorts(cli_port, data_port)

    # file_path = "xwr18xx_AOP_profile_2024_03_18T05_58_07_151.cfg"
    # file = open(file_path, 'r')
    # parser.sendCfg(file)
    # print("CFG SEND")

    # for _ in range(5):
    #     radarOutputDict = parser.readAndParseUartDoubleCOMPort()
    #     print(len(radarOutputDict))


    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(radar_parser=parser)
    window.show()
    sys.exit(app.exec_())
