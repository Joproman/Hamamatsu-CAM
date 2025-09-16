"""Calibrate the trap by calculating the distance factor."""
import fileinput
import re
import numpy as np
import pandas as pd
import time
from tools import load_image, ion_positions, dark_ions, read_factor
from PIL import Image
import HamamatsuCamera

# import tiqi_andor_camera.AndorCamera as andorCam


def calibrate(cam):
    """ Main function """
    # try:
    #     cam = andorCam.AndorCamera(gain= 200)
    #     cam.record(number_of_images=4, mode='ring buffer')
    #     time.sleep(0.5)
    # except ValueError:
    #     cam = None

    cam.record(number_of_images=4, mode='ring buffer')
    time.sleep(0.5)

    image = (load_image(cam) if cam is not None else np.array(Image.open('data/image.png'))[:, :, 0])
    pos, _, _ = ion_positions(image)
    pos = pos[:, 1]

    if len(pos) > 1:
        print('The detected ions are at positions: ', pos)

        pos = pos[pos.argsort()] # order
        pos = (pos - pos[0])[1:]
        mask = (pd.read_csv('data/mask.csv', index_col=0).to_numpy())[len(pos) - 1][1:len(pos)+1]

        new_dist_factor = round(np.mean(pos/mask), 1)

        with fileinput.FileInput('settings.py', inplace=True) as file:
            for line in file:
                print(re.sub('DIST_FACTOR =.*$',
                'DIST_FACTOR = {}'.format(new_dist_factor), line), end='')


        # conn.set(name='/image_analysis/distance_factor', value=new_dist_factor)

        print('The factor is {}'.format(new_dist_factor))
    else:
        print('The trap needs to have more than 1 ion to calibrate.')


if __name__ == "__main__":
    cam = HamamatsuCamera.HamamatsuCameraPlugin(exposureTime=0.2)
    calibrate(cam)
