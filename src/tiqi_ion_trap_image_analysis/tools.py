"""This file contains all the needed functions"""
import time
import fileinput
import re
import os
import sys
import json
import logging
from threading import Thread
import warnings
from multiprocessing import shared_memory
import influxdb
from influxdb_client import InfluxDBClient, Point
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from scipy.signal import peak_widths
from sympy import symbols, nsolve
from scipy.ndimage import center_of_mass
from PIL import Image
import settings
import zmq.green as zmq
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from datetime import datetime
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS

if settings.SYSTEM == "Windows":
    from semaphore_win_ctypes import Semaphore, SemaphoreWaitTimeoutException
elif settings.SYSTEM == "Linux":
    from posix_ipc import Semaphore, BusyError

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://*:{settings.ZMQ_PORT}")

log = logging.getLogger(__name__)


class Main(Thread):
    """
    Main class. Does the processing of the images and stores all the outputs.
    """

    def __init__(self, cam, sem):
        """
        Connect to influxdb. Instantiate all the needed classes and variables.
        """
        Thread.__init__(self)

        # Connect to influxdb
        # self.db = influxdb.InfluxDBClient(
        #     host=settings.DB_HOST,
        #     port=settings.DB_PORT,
        #     username=settings.DB_USER,
        #     password=settings.DB_PWD,
        #     database=settings.DB_DATASET,
        #     ssl=True,
        #     verify_ssl=True
        # )
        self.client = InfluxDBClient(
            # url="http://avalon.qchub.ch:8086",
            url="http://influxdb.qchub.ch:8086",
            # token="1SnC41qsbuiLtHbd5C1vkNRIYtqL5qvTME1agVYO_3R9NtcBt-ZC29khymkaoYuGJ46VTp2hGw1MXcjLzQE-ng==",
            token="O-iKMjUeuROGkGeRdpeX0YwqLAqh7fmBnkit5w2ehWKqTrqmR9jGY1Tm32pRWc8vPDGfNeP_gKpuYDJfEnkBbA==",
            org="qchub",
        )
        self.api = self.client.buckets_api()
        self.bucket = "qchub" # self.api.find_bucket_by_name("qchub")
        self.org = "qchub"

        # Connect to camera
        self.cam = cam
        self.fetch_from_camera = True

        # Semaphore
        self.sem = sem

        # Instantiate classes
        self.memory = Memory()
        self.count_avg = MovingAverage()
        self.fit_avg = FitAverage()
        # Start threads
        self.count_avg.daemon = True
        self.count_avg.start()
        self.fit_avg.daemon = True
        self.fit_avg.start()

        # Define class attributes
        self.init_time = time.time()
        self.output = {}

        # Add toggle for ion detection
        self.ion_detection = True

    def run(self):
        try:
            # Run processing
            while True:
                self._background_thread()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            log.exception(f"Error: {e}; File: {fname}; Line: {exc_tb.tb_lineno}")
            os._exit(1)


    def _background_thread(self):
        """
        Apply the processing on the images.
        """
        dist_factor = read_factor()
        image = load_image(self.cam, self.fetch_from_camera)

        if self.ion_detection is True:
            ion_cloud = detect_ion_cloud(image)

            if ion_cloud:
                output = {
                    'image': image,
                    'coord': np.array([]),
                    'com_coord': np.array([]),
                    'num_ions': None,
                    'bright_ions': None,
                    'dim_ions': None,
                    'dark_ions': None,
                    'linearity': 'cloud',
                    'max_int': None,
                    'num_pixels': None,
                    'reorder_avg': None,
                    'reorder_flag': None,
                    'max_spot_size': None,
                    'roundness': None,
                    'fit_error': None,
                    'fit_avg': None,
                    'ion_cloud': 1
                }

            elif not ion_cloud:
                output = self._processing(image, dist_factor)

                # Send data to GRAFANA
                self._save_in_memory(output)

        elif self.ion_detection is False:
            output = {
                'image': image,
                'coord': np.array([]),
                'com_coord': np.array([]),
                'num_ions': None,
                'bright_ions': None,
                'dim_ions': None,
                'dark_ions': None,
                'linearity': None,
                'max_int': None,
                'num_pixels': None,
                'reorder_avg': None,
                'reorder_flag': None,
                'max_spot_size': None,
                'roundness': None,
                'fit_error': None,
                'fit_avg': None,
                'ion_cloud': 0
            }

        # Delete unnecessary variables for html and put output in queue
        del output['num_ions']
        del output['reorder_flag']
        del output['ion_cloud']
        sem = acquire_semaphore_write(self.sem, num_sem=10)
        self.output = output
        release_semaphore_write(sem, num_sem=10)

        if len(self.output['com_coord']) > 0:
            # Save coord values in shared memory
            socket.send_string(f"Coordinates {round(self.output['com_coord'][0], 2)} "
                               f"{round(self.output['com_coord'][1], 2)}")
        else:
            socket.send_string(f"Coordinates {0} {0}")

    def _processing(self, image, dist_factor):
        img = image.copy()
        coord, max_spot_size, com_coord = ion_positions(img)
        if len(coord) > 0:
            roundness = ion_roundness(img, coord[0][:2])
            img = compensate_gradient(img, coord)
            max_int, num_pix = max_ion_intensity(img, coord)
            coord = detect_dim_ions(img, coord, max_int)
            linearity = ion_linearity(coord)

            if len(coord) > 1 and dist_factor is not None:
                coord, fit_error = dark_ions(img, coord, dist_factor)
                self.fit_avg.update_value(fit_error)

            else:
                fit_error = 0
        else:
            roundness = 0
            fit_error = 0
            max_int, num_pix = (0, 0)
            linearity = "No ions"

        old_coord = self.memory.get_value()
        self.memory.save_value(coord)
        coord = self.memory.get_value()

        same_coord = _compare_coord(old_coord, coord)

        # If idx changes, it means that ion reordered
        reorder_flag = 0
        if not same_coord:
            reorder_flag = 1
            self.count_avg.update_count()

        num_ions = len(coord)
        num_bright_ions = np.sum(coord[:, 2] == 0).item()
        num_dim_ions = np.sum(coord[:, 2] == 1).item()
        num_dark_ions = np.sum(coord[:, 2] == 2).item()

        return {
            'image': image,
            'coord': coord,
            'com_coord': com_coord,
            'num_ions': num_ions,
            'bright_ions': num_bright_ions,
            'dim_ions': num_dim_ions,
            'dark_ions': num_dark_ions,
            'linearity': linearity,
            'max_int': int(max_int),
            'num_pixels': int(num_pix),
            'reorder_avg': self.count_avg.moving_avg,
            'reorder_flag': reorder_flag,
            'max_spot_size': int(max_spot_size),
            'roundness': int(roundness),
            'fit_error': int(fit_error),
            'fit_avg': int(self.fit_avg.moving_avg),
            'ion_cloud': 0
        }

    def write_data(self, data, write_option=SYNCHRONOUS):
        write_api = self.client.write_api(write_option)
        write_api.write(bucket=self.bucket, record=data, org=self.org)

    def _save_in_memory(self, output):
        """
        Save measured variables in shared memory
        """

        if output['num_ions'] > 0 and len(output['com_coord']) > 0 and len(output['coord']) > 0:
            try:
                grafana_data = {
                    'measurement': 'ion_image',
                    'fields': {
                        'num_ions': output['num_ions'],
                        'num_bright_ions': output['bright_ions'],
                        'num_dim_ions': output['dim_ions'],
                        'num_dark_ions': output['dark_ions'],
                        'reorder_rate': output['reorder_avg'],
                        'max_int': output['max_int'],
                        'max_spot_size': output['max_spot_size'],
                        'ion_roundness': output['roundness'],
                        'ion_x_pos': output['coord'][:, 1][0], # we swap x and y to make it more readable (like coordinates)
                        'ion_y_pos': output['coord'][:, 0][0],
                        'com_x_coord': output['com_coord'][1], # we swap x and y to make it more readable (like coordinates)
                        'com_y_coord': output['com_coord'][0],
                        'ion_cloud': output['ion_cloud']
                    }
                }
                influxdb_data = Point("ion_image") \
                        .field("num_ions", output['num_ions']) \
                        .field("num_bright_ions", output['bright_ions']) \
                        .field("num_dim_ions", output['dim_ions']) \
                        .field("num_dark_ions", output['dark_ions']) \
                        .field("reorder_rate", output['reorder_avg']) \
                        .field("max_int", output['max_int']) \
                        .field("max_spot_size", output['max_spot_size']) \
                        .field("ion_roundness", output['roundness']) \
                        .field("ion_x_pos", output['coord'][:, 1][0]) \
                        .field("ion_y_pos", output['coord'][:, 0][0]) \
                        .field("com_x_coord", output['com_coord'][1]) \
                        .field("com_y_coord", output['com_coord'][0]) \
                        .field("ion_cloud", output['ion_cloud']) \
                # p = Point("ion_image").tag("location", "RT").field("temperature", 25.3)
                self.write_data(influxdb_data)
                # self.write_data(grafana_data)
                # print(grafana_data)
                # print(datetime.now(), 'num_ions', output['num_ions'])
            except (ConnectionError, influxdb.exceptions.InfluxDBClientError, influxdb.exceptions.InfluxDBServerError):
                warnings.warn("ConnectionError: Can't connect to Grafana")


class Memory:
    """
    Stores values in memory and counts how many times each value was stored.
    """

    def __init__(self):
        self.i = 0
        self.max_idx = 0
        self.memory = []
        self.count = []

    def save_value(self, value):
        """
        Saves new value in memory. If the value already existed in memory, sum 1 to count.
        """
        # If memory is empty, append new value
        if len(self.memory) == 0:
            self.memory.append(value)
            self.count.append([0])
            self.i += 1
            return

        # If value has the same ion types as any element in memory, append abs(value - element)
        diff = []
        for i, element in enumerate(self.memory):
            if len(element) == len(value):
                if all(value[:, 2] == element[:, 2]):
                    diff_pos = np.abs(value[:, :2] - element[:, :2])
                    diff.append([i, diff_pos.sum()])

        diff = np.array(diff)
        # If sum(abs(value - element)) is less than 10, consider as same distribution
        try:
            if 0 <= diff[:, 1].min() < 30:
                idx = diff[np.argmin(diff[:, 1])][0]
                self.memory[idx] = value
                self.count[idx].append(self.i)
                self.i += 1
            else:
                self.memory.append(value)
                self.count.append([self.i])
                self.i += 1
        except IndexError:
            self.memory.append(value)
            self.count.append([self.i])
            self.i += 1

        # When we reach NUM_ITERATIONS, we remove the 0 value and
        # substract 1 for each value inside count list.

        if self.i == settings.NUM_ITERATIONS:
            list_len = []
            for i, val in enumerate(self.count):
                if 0 in val:
                    val.remove(0)
                if len(val) == 0:
                    del self.memory[i]
                else:
                    val = [val[i] - 1 for i in range(len(val))]
                    self.count[i] = val
                    list_len.append(len(val))
            self.count = [i for i in self.count if len(i) > 0]
            self.i -= 1
            self.max_idx = np.argmax(list_len)

    def get_value(self):
        """
        Returns the value stored in memory with the highest count, together with the index
        where this value was stored.
        """
        try:
            max_value = self.memory[self.max_idx]
        except IndexError:
            max_value = np.array([[-1, -1, -1]])

        return max_value


class FitAverage(Thread):
    """
    Compute moving average of fit error
    """

    def __init__(self):
        Thread.__init__(self)
        self.value = 0
        self.value_history = [0]
        self.moving_avg = 0

    def run(self):
        """
        Every second update history and compute moving average.
        """
        try:
            while True:
                time.sleep(1)
                self.value_history.append(self.value)
                if len(self.value_history) > 60:
                    self.value_history.pop(0)
                self.moving_avg = np.mean(self.value_history)
                min_val = min(self.value_history)
                self.value_history = [i - min_val for i in self.value_history]
                self.value -= min_val

        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            log.exception(f"Error: {e}; File: {fname}; Line: {exc_tb.tb_lineno}")
            os._exit(1)

    def update_value(self, value):
        """
        Update value.
        """
        self.value = value


class MovingAverage(Thread):
    """
    Keep record of number of reorders of the ions and return a moving average of
    number of reorders per minute.
    """

    def __init__(self):
        Thread.__init__(self)
        self.reorder_count = 0
        self.reorder_history = [0]
        self.moving_avg = 0


    def run(self):
        """
        Every second update history and compute moving average.
        """
        try:
            while True:
                time.sleep(1)
                self.reorder_history.append(self.reorder_count)
                if len(self.reorder_history) > 60:
                    self.reorder_history.pop(0)
                self.moving_avg = self.reorder_history[-1] - self.reorder_history[0]
                min_val = min(self.reorder_history)
                self.reorder_history = [i - min_val for i in self.reorder_history]
                self.reorder_count -= min_val

        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            log.exception(f"Error: {e}; File: {fname}; Line: {exc_tb.tb_lineno}")
            os._exit(1)

    def update_count(self):
        """
        Update count.
        """
        self.reorder_count += 1


def read_factor():
    """
    Reads distance factor from settings.py
    """
    dist_factor = settings.DIST_FACTOR

    return dist_factor


def load_image(cam, fetch=True):
    """
    Load image of the ion trap from camera.
    """
    if fetch:
        if cam is not None:
            cam.wait_for_first_image()
        image, _ = (cam.image(image_number=0xFFFFFFFF) if cam is not None else [np.array(Image.open('data/image.png'))[:, :, 0], None])
    else:
        image = np.zeros((2048, 2048), dtype=np.uint16)

    return image


def detect_ion_cloud(image):
    """
    Measure the area of the biggest contour of the image. If it is larger than a certain threshold
    consider an ion cloud.
    """
    file_path = os.path.dirname(os.path.abspath(__file__)) + "\\data\\region.json"
    with open(file_path, "r") as f:
        region = np.array(json.load(f)["ROI"])
    # image = np.array(image[region[0, 1]:region[1, 1], region[0, 0]:region[1, 0]])  # crop region of interest
    # image = gaussian(image, sigma=2)
    # _, binary = cv2.threshold(image, thresh=image.max() * 0.9, maxval=255, type=cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(np.uint8(binary), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # if len(contours) > 0:
    #     max_contour = max(contours, key=cv2.contourArea)
    #     is_ion_cloud = settings.AREA_THS < cv2.contourArea(max_contour) < np.size(image) / 2
    # else:
    #     is_ion_cloud = False

    is_ion_cloud = False
    
    return is_ion_cloud


def ion_positions(image):
    """
    Given an image, find the coordinates of ion centers.
    Filter out the coordinates that are not horizontally aligned with the brightest center.
    Order the coordinates and mark them as bright ions (by adding a column with zeros).
    Finally return the coordinates together with the maximum ion width.
    """
    # image = rotate(image, settings.RANGLE)
    coordinates, max_width, com_coord = _max_coord(image, width=True)  # measure local max
    # print("before", coordinates, max_width, com_coord)
    # delete local maxima that are not in the same axis as the brightest spot.
    try:
        # h_axis = coordinates[0, 0]
        # coordinates = coordinates[np.logical_and(coordinates[:, 0] < h_axis + 15,
        #                                          coordinates[:, 0] > h_axis - 15)]
        h_axis = coordinates[0, 1]
        # print("h_axis", h_axis)
        # print("coordinates[0, :]", coordinates[:, 1])
        coordinates = coordinates[np.logical_and(coordinates[:, 1] < h_axis + 15, coordinates[:, 1] > h_axis - 15)]
    except IndexError:
        pass
    # print("after", coordinates, max_width, com_coord)

    # x, y = zip(*coordinates)
    # plt.imshow(image)
    # plt.scatter(x, y)
    # plt.show()

    file_path = os.path.dirname(os.path.abspath(__file__)) + "\\data\\region.json"
    with open(file_path, "r") as f:
        region = np.array(json.load(f)["ROI"])
    coordinates += np.flip(region[0])  # change reference
    com_coord += np.flip(region[0])
    coordinates = coordinates[coordinates[:, 1].argsort()]  # order
    # add 3rd column indicating the ion's state (bright, dark, wrong isotope)
    coordinates = np.concatenate((coordinates, np.zeros((len(coordinates[:, 1]), 1))), axis=1)

    if len(coordinates) == 0:
        coordinates = np.empty((0, 3))


    return coordinates.astype(int), max_width, com_coord


def ion_roundness(image, center):
    """
    Compute PCA of brightest ion to check its circularity.
    Return 1 if the ion is circular, 0 if it is linear.
    """
    # Remove pixels farther than 15 pixels from the brightest ion center.
    image_copy = image.copy()
    mask = np.zeros(image_copy.shape)
    mask = cv2.circle(mask, tuple(center[::-1]), radius=20, color=(255, 255, 255),
                      thickness=-1) == 0
    image_copy[mask] = 0

    c_x = center[0]
    c_y = center[1]
    image_copy = image_copy[max(0, c_x - 25): c_x + 25, max(0, c_y - 25): c_y + 25]  # crop the image

    # get coordinates of ion pixels
    pca_input = np.swapaxes(np.stack(np.where(image_copy > image_copy.max() * 0.5)), 0, 1)

    if len(pca_input) > 1:
        pca = PCA(n_components=2)  # compute PCA
        pca.fit(pca_input)
        var1, var2 = pca.explained_variance_  # return variance of 2 principal components
        roundness = var2 / var1  # if the ion is a circle, var2 / var1 = 1
    else:
        roundness = 0

    return roundness * 100


def ion_linearity(points):
    """
    Fit the points into a line and return the sum of the squared residuals of the least-squares fit.
    We will use this value to determine the linearity of the ions.
    """
    # We use only bright ions
    points = points[points[:, 2] == 0]

    y = points[:, 0]
    x = points[:, 1]

    if len(x) == 0:
        linearity = 'No ions'
    else:
        _, residuals, _, _, _ = np.polyfit(x, y, deg=1, full=True)
        if residuals.size == 0 or (residuals.size > 0 and residuals < 4):
            linearity = 'linear'
        else:
            linearity = 'zigzag'

    return linearity


def compensate_gradient(image, coord):
    """
    Measure the intensity gradient of the ions along the x axis and correct it.
    Ignore any background intensity gradient.
    """
    if len(coord) > 1:
        image = image.copy()
        slope = _measure_gradient(image, coord)

        start = coord[0, 1] - 30
        end = coord[-1, 1] + 30

        for i in np.r_[start:end]:
            image[:, i] = np.clip(image[:, i] - (i - start) * slope, a_min=0, a_max=255)

    return image.astype(np.uint16)


def max_ion_intensity(image, centers):
    """
    Returns the maximum intensity of the pixels around the center of each ion,
    together with the number of pixels that it summed over.
    """
    if len(centers) > 0:
        intensity = []
        for center in centers:
            intensity.append(_ion_intensity(image, center[:-1]))
        max_intensity, num_pixels = intensity[(intensity == np.max(intensity)).nonzero()[0][0]]
    else:
        max_intensity, num_pixels = [0, 0]

    return max_intensity, num_pixels


def detect_dim_ions(image, centers, max_intensity):
    """
    Measure intensity for each ion. If the intensity is below INT_THS,
    we mark it as a wrong isotope.
    """
    for i, center in enumerate(centers):
        intensity = _ion_intensity(image, center[:-1])
        if intensity[0] <= settings.INT_THS * max_intensity and center[-1] != 2:
            centers[i, 2] = 1
    return centers


def dark_ions(image, coord, dist_factor):
    """
    Find dark ions. It imports a mask of position and distance values corresponding to different
    number of ions in the ion trap. For each combination of ions it calculates a fitting loss
    (only using the detected ions) and keeps the mask with lowest loss. If this mask contains more
    ions than the number detected, it adds them as "dark ions".

    Note: To avoid errors when the first/last ion is missing, we compute the fitting loss to the
    detected positions starting from the left AND right. Please note we will have 3 different
    fitting losses (positions from left, positions from right and relative distances).
    The algorithm takes the absolute minimum.
    """
    # Compute the positions starting from the left/right
    x_pos = np.stack((coord[:, 1] - coord[0, 1], np.sort(np.abs(coord[:, 1] - coord[-1, 1]))))
    x_dist = _ion_distance(coord)  # compute relative distances
    # Load the masks. TODO: Remove the mask options that end up outside the image!
    low_idx = max(0, len(x_pos[0]) - 2)
    high_idx = min(28, len(x_pos[0]) * 2)
    mask = []
    mask_dist = []
    # mask = (pd.read_csv(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), "mask.csv"), index_col=0).to_numpy() *
    #        dist_factor)[low_idx: high_idx, :]
    # mask_dist = (pd.read_csv(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), "mask_dist.csv"), index_col=0).to_numpy() *
    #             dist_factor)[low_idx: high_idx, :]
    if len(mask) > 0:
        # Compute min error between every value of the mask and every value of the measured distances
        comparison = np.full((3, mask.shape[0], mask.shape[1]), 100000, dtype=float)
        for i in range(mask.shape[0]):
            dist = x_dist.copy()
            for j in range(mask.shape[1]):
                val = mask[i, j]
                val_dist = mask_dist[i, j]
                if val >= 0:
                    comparison[:2, i, j] = np.min((x_pos - val) ** 2, axis=1)
                if val_dist >= 0:
                    diff = (dist - val_dist) ** 2
                    comparison[2, i, j] = np.min(diff)
                    dist[np.argmin(diff)] = 0

        comp = np.sort(comparison, axis=2)[:, :, :len(x_pos[1])]
        # Set to 0 the last column of the comparison of the distances (they have 1 value less)
        # This way the last column won't affect the loss
        comp[2, :, -1] = 0
        loss = np.sum(comp, axis=2) / len(x_pos)

        idx = np.where(loss == np.min(loss))

        if idx[0][0] == 1:  # if using "inverted positions", transform back
            new_x_pos = np.abs(mask[idx[1][0], :][mask[idx[1][0], :] >= 0] - coord[-1, 1])
        else:
            new_x_pos = mask[idx[1][0], :][mask[idx[1][0], :] >= 0] + coord[0, 1]

        diff_n = len(new_x_pos) - len(x_pos[0])
        for i in range(diff_n):
            mask2 = comparison[idx[0][0], idx[1][0], :] < 100000
            idx2 = np.argsort(comparison[idx[0][0], idx[1][0], :][mask2])[-(diff_n - i)]
            if idx2 != 0:
                # If we are using the distances' loss, then we have to add 1 to the index
                if idx[0][0] == 2:
                    idx2 += 1
                coord = np.append(coord, np.array([[np.mean(coord[:, 0]),
                                                    new_x_pos[idx2], 2]]), axis=0)

            elif idx2 == 0 and idx[0][0] == 2:
                coord = np.append(coord, np.array([[np.mean(coord[:, 0]), new_x_pos[0] -
                                                    comparison[2, idx[1][0], 0], 2]]), axis=0)

        coord = coord[coord[:, 1].argsort()].astype(int)
        coord = coord[coord[:, 1] < image.shape[1]]  # remove points outside the image
        # TODO: uncomment this!
        # if loss.min() > 20:
        #     warnings.warn("Loss is too high... "
        #     "Try calculating a new factor value using calibrate() function.")

    else:
        loss = np.array([-1])

    return coord, loss.min()


def init_semaphore(name: str = settings.SEM_NAME, count=1):
    sem = Semaphore(name)  # , flags=O_CREAT, initial_value=settings.sem_max)
    try:
        sem.close()
    except Exception as e:
        log.debug(f'Exception unlinking semaphore before initializing: {e}')
    # With flags set to O_CREAT, the module opens the semaphore if it exists
    # (in which case mode and initial value are ignored) or creates it if it doesn't.
    try:
        sem.create(maximum_count=count)
    except Exception as e:
        log.warning(f'Could not initialize semaphore {sem.name}; maybe it already exists? {e}')
    # sem.open() # opened upon creation
    return sem

if settings.SYSTEM == "Windows":
    def acquire_semaphore_read(sem, name: str = settings.SEM_NAME, timeout=1):
        """
        Acquire semaphore.
        """
        while True:
            try:
                sem.acquire(timeout_ms=timeout * 1000)
            except SemaphoreWaitTimeoutException:
                log.warning('semaphore {} blocked'.format(name))
            else:
                break
        return sem

    def acquire_semaphore_write(sem, name: str = settings.SEM_NAME, timeout=1, num_sem=10):
        """
        Acquire all semaphores to write into memory.
        """
        while True:
            for i in range(num_sem):
                try:
                    sem.acquire(timeout_ms=timeout * 1000)
                except SemaphoreWaitTimeoutException:
                    log.warning('semaphore {} blocked'.format(name))
                    break
            if i == num_sem - 1:
                break

        return sem

elif settings.SYSTEM == "Linux":
    def acquire_semaphore_read(sem, name: str = settings.SEM_NAME, timeout=1):
        """
        Acquire semaphore.
        """
        while True:
            try:
                sem.acquire(timeout=timeout)
            except BusyError:
                log.warning('semaphore {} blocked'.format(name))
            else:
                break
        return sem

    def acquire_semaphore_write(sem, timeout=1, num_sem=10):
        """
        Acquire all semaphores to write into memory.
        """
        while True:
            for i in range(num_sem):
                try:
                    sem.acquire(timeout=timeout)
                except BusyError:
                    break
            if i == num_sem - 1:
                break

        return sem


def release_semaphore_read(sem: Semaphore):
    """
    Release semaphore.
    """
    sem.release()


def release_semaphore_write(sem, num_sem=10):
    for _ in range(num_sem):
        sem.release()

def pos_solver(num):
    """
    Solve numerically the positions of N ions in the trap
    """
    start = np.linspace(-num / 3, num / 3, num=num)

    u = symbols(list('u' + str(i) for i in range(1, num + 1)))
    eqns = []
    for i, x in enumerate(u):
        eq = x
        for j, y in enumerate(u):
            if i > j:
                eq -= (x - y) ** (-2)
            elif i < j:
                eq += (x - y) ** (-2)
        eqns.append(eq)

    solution = nsolve(eqns, u, start, dict=True)
    print('The solution for N = {} is: \n {}'.format(num, solution[0]))
    return solution


def plot_detection(image, points, cross_hairs):
    """
    Use OpenCV to plot the detected ions on the image, following this color code:
        - Red: Bright Ion
        - Green: Bright Ion
        - Blue: Dark Ion
    Code the image into bytes and return it.
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()

    for point in points:
        if point[2] == 1:
            cv2.circle(overlay, tuple(np.flip(point[:2])), radius=settings.RADIUS_PLT,
                       color=(255, 0, 0), thickness=2)
        elif point[2] == 0:
            cv2.circle(overlay, tuple(np.flip(point[:2])), radius=settings.RADIUS_PLT,
                       color=(0, 255, 0), thickness=2)
        else:
            cv2.circle(overlay, tuple(np.flip(point[:2])), radius=settings.RADIUS_PLT,
                       color=(0, 0, 255), thickness=2)

    for i, cross_hair in enumerate(cross_hairs['cross_hairs']):
        color = cross_hairs['color'][i]
        bgr_color = color[::-1]
        cv2.line(overlay, (int(cross_hair[0]), 0), (int(cross_hair[0]), overlay.shape[0]), color=bgr_color, thickness=1)
        cv2.line(overlay, (0, int(cross_hair[1])), (overlay.shape[1], int(cross_hair[1])), color=bgr_color, thickness=1)

    alpha = settings.ALPHA
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


def _max_coord(image, width=False):
    """
    - Apply a band pass filter (difference of gaussians) for blob detection.
    - Detect local maxima.
    - If we have one ion, find the center of the ion by computing the center of mass 
    of the binary image containing the ion.
    - If width is True return also the max width of all the ions.
    """
    file_path = os.path.dirname(os.path.abspath(__file__)) + "\\data\\region.json"
    with open(file_path, "r") as f:
        region = np.array(json.load(f)["ROI"])
    image = image[region[0, 1]:region[1, 1], region[0, 0]:region[1, 0]]

    filt_image = _difference_of_gaussians(image, low_sigma=settings.BLOB_SIZE, preserve_range=True)
    coordinates = peak_local_max(filt_image, threshold_abs=settings.PEAK_THRESHOLD, min_distance=settings.ION_DIST)

    if len(coordinates) == 1:
        # We compute the center of mass of the binary image to get a more precise value
        _, binary = cv2.threshold(filt_image, thresh=filt_image.max() * settings.COM_THRESHOLD, maxval=255, type=cv2.THRESH_BINARY)
        # Check if binary image has any non-zero values before computing center of mass
        if np.sum(binary) > 0:
            com_coord = np.array(center_of_mass(binary))
        else:
            com_coord = coordinates[0]  # Fallback to peak position

    elif len(coordinates) > 1:
        com_coord = coordinates[0]

    else:
        com_coord = np.empty((0, 2))

    if width:
        try:
            max_width = _ion_max_spot_size(image[coordinates[0, 0]])
        except IndexError:
            max_width = 0

        return coordinates, max_width, com_coord

    return coordinates, com_coord


def _difference_of_gaussians(image, low_sigma, preserve_range=True):
    """
    Apply a bandpass filter for blob detection by computing two gaussian filters (with sigma
    values low_sigma and low_sigma*1.6) and substracting the more-blurred image with the
    less-blurred image. The more-blurred image has a sigma 1.6 times higher than the less-blurred
    image, which is optimal for blob detection of sizes similar to low_sigma.
    """
    im1 = gaussian(image, sigma=low_sigma, preserve_range=preserve_range)
    im2 = gaussian(image, sigma=low_sigma * 1.6, preserve_range=preserve_range)
    filt_image = (im1 - im2).clip(min=0)  # apply bandpass filter (difference of gaussians)

    return filt_image


def _ion_max_spot_size(y):
    """
    Return max width of peaks.
    """
    y = _smooth(y, window_len=15, window='hamming')
    width, _, _, _ = peak_widths(y, [np.argmax(y)], rel_height=0.5)
    return np.max(width)


def _smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def _measure_gradient(image, coord):
    """
    Fits the ion intensity into a polynomial function of degree one and returns
    the value of the slope.
    """
    y = []
    x = []
    for pos in coord:
        y.append(np.sum(image[pos[0] - 15: pos[0] + 15, pos[1]], axis=0))
        x.append(pos[1])
    m, _ = np.polyfit(x, y, deg=1)

    return m / 30


def _ion_intensity(image, center):
    """Returns the sum of the intensity of the pixels inside a circle around the given center."""
    mask = np.zeros(image.shape)
    mask = cv2.circle(mask, tuple(center[::-1]), radius=settings.RADIUS, color=(255, 255, 255),
                      thickness=-1) != 0
    num_pixels = np.sum(mask)
    intensity = np.sum(image[mask])

    return [intensity, num_pixels]


def _ion_distance(coord):
    """
    Compute the relative distance between ions.
    """
    dist = []
    for i in range(len(coord) - 1):
        dist.append(np.sqrt((coord[i, 0] - coord[i + 1, 0]) ** 2 +
                            (coord[i, 1] - coord[i + 1, 1]) ** 2).item())
    dist = np.array(dist)

    return dist


def _compare_coord(coord1, coord2):
    """
    Check if coord1 == coord2 up to some threshold (30 pixels)
    """
    same_coord = False

    if len(coord1) == len(coord2):
        if all(coord1[:, 2] == coord2[:, 2]):
            if np.abs(coord1[:, :2] - coord2[:, :2]).sum() < 30:
                same_coord = True

    return same_coord


def plot_hist(img, hmin: int = 0, hmax : int =65536):
    """
    Plot histogram of image
    """
    hist = cv2.calcHist([img], [0], None, [int(hmax - hmin)], [hmin, hmax])
    x = np.arange(hmin, hmax, 1)
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.plot(x, hist)
    width, height = fig.get_size_inches() * fig.get_dpi()

    canvas.draw()  # draw the canvas, cache the renderer

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)[:, :, 0]
    return image


def set_min_max(image, hmin, hmax):
    """
    Set the min and max values of the histogram of the image
    """
    image = image - hmin
    image[image < 0] = 0
    image = image*255/hmax
    image[image > 255] = 255
    image = image.astype(np.uint8)
    return image


def change_settings(name, value):
    """
    Change the value stored in settings.py
    """
    # temp_path = os.path.join(os.path.dirname(__file__), 'settings.py')
    temp_path = r"C:\Users\qchub-lab-user\Desktop\RT\hamamatsu-camera\src\settings.py"
    with fileinput.FileInput(temp_path, inplace=True) as file:
        for line in file:
            print(re.sub('{} =.*$'.format(name), '{} = {}'.format(name, value), line), end='')
    return
