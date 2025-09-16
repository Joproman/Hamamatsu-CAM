"""Main file"""
# import matplotlib
# matplotlib.use('tkagg')
import os.path
import time
import ast
import colorsys
import random
import json
import logging
import atexit
import numpy as np
import cv2
# import pco
from scipy.ndimage import rotate
from flask import Flask, render_template, Response, request
from .tools import plot_detection, Main, init_semaphore, acquire_semaphore_read
from .tools import plot_hist, set_min_max, change_settings, release_semaphore_read
# import settings

# import tiqi_andor_camera.AndorCamera as andorCam

def imageAnalysis(cam, settings):

    if settings.SYSTEM == "Linux":
        import threading

    app = Flask(__name__)
    log = logging.getLogger('werkzeug')  # get flask logger
    log.setLevel(logging.WARN)  # show only flask errors
    app.logger.setLevel(logging.WARN)  # show only app errors

    # Initiate semaphore
    if settings.SYSTEM == "Windows":
        SEM = init_semaphore(count=10)
    elif settings.SYSTEM == "Linux":
        SEM = threading.Semaphore(10)

    # # Initiate camera
    # if settings.SYSTEM == "Windows":
    #     try:
    #         cam = andorCam.AndorCamera(gain= 200)
    #         cam.record(number_of_images=4, mode='fifo')
    #         cam.set_exposure_time(settings.EXP_TIME)
    #     except ValueError:
    #         cam = None
    # elif settings.SYSTEM == "Linux":
    #     cam = None

    # Start image analysis
    main = Main(cam, SEM)
    main.daemon = True
    main.start()

    # Define min and max values of histogram
    HMIN = settings.HIST_MIN
    HMAX = settings.HIST_MAX

    # Define image angle
    RANGLE = settings.RANGLE

    # Define cross_hairs with their color
    list_cross_hairs = settings.CROSS_HAIRS

    # Close camera when exiting
    def on_exit():
        """Close camera on exit"""
        # Close camera
        if cam is not None:
            print("Closing camera.")
            main.fetch_from_camera = False
            cam.close()

    atexit.register(on_exit)

    @app.route('/')
    def index():
        """
        Run html script.
        """

        if main.fetch_from_camera is True:
            cam_status = 'ON'
        else:
            cam_status = 'OFF'

        if main.ion_detection is True:
            ion_status = 'ON'
        else:
            ion_status = 'OFF'

        exp_time = (round(cam.get_exposure_time(), 3) if cam is not None else 0)

        return render_template('index.html', decode_responses=True, data={'status': cam_status,
                                                                        'expTime': exp_time,
                                                                        'ionStatus': ion_status,
                                                                        'angle': RANGLE})

    def gen():
        """
        Acquire image from shared memory, plot on it the positions of the detected ions and
        return it.
        """
        while True:
            # Load image and coords from shared memory
            acquire_semaphore_read(SEM)
            image = main.output['image'].copy()
            coord = main.output['coord'].copy()
            release_semaphore_read(SEM)
            image = set_min_max(image, HMIN, HMAX)
            image = plot_detection(image, coord.astype(int), list_cross_hairs)
            image = cv2.resize(image, (int(image.shape[1]/2.5), int(image.shape[0]/2.5)), interpolation=cv2.INTER_AREA)
            image = rotate(image, RANGLE)
            frame = cv2.imencode(".jpg", image)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def gen_roi():
        """
        Acquire image from shared memory, plot on it the positions of the detected ions, crop the ROI and
        return it.
        """
        while True:
            # Load ROI
            file_path = os.path.dirname(os.path.abspath(__file__)) + "\\data\\region.json"
            with open(file_path, "r") as f:
                region = np.array(json.load(f)["ROI"])
            # Load image and coordinates
            acquire_semaphore_read(SEM)
            image = main.output['image'].copy()
            coord = main.output['coord'].copy()
            release_semaphore_read(SEM)
            image = set_min_max(image, HMIN, HMAX)
            image = plot_detection(image, coord.astype(int), list_cross_hairs)
            image = image[region[0, 1]:region[1, 1], region[0, 0]:region[1, 0]]
            image = rotate(image, RANGLE)
            frame = cv2.imencode(".jpg", image)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def gen_hist():
        """
        Return the image's histogram
        """
        while True:
            # Load image and coords from shared memory
            acquire_semaphore_read(SEM)
            image = main.output['image'].copy()
            release_semaphore_read(SEM)
            image = plot_hist(image, HMIN, HMAX)
            frame = cv2.imencode(".jpg", image)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def gen_hist_img():
        """
        Return image displayed in /hist 
        """
        while True:
            # Load image and coords from shared memory
            acquire_semaphore_read(SEM)
            image = main.output['image'].copy()
            coord = main.output['coord'].copy()
            release_semaphore_read(SEM)
            image = set_min_max(image, HMIN, HMAX)
            image = plot_detection(image, coord.astype(int), list_cross_hairs)
            image = cv2.resize(image, (int(image.shape[1]/2.5), int(image.shape[0]/2.5)), interpolation=cv2.INTER_AREA)
            image = rotate(image, RANGLE)
            frame = cv2.imencode(".jpg", image)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def gen_capture():
        # Load image and coords from shared memory
        while True:
            acquire_semaphore_read(SEM)
            image = main.output['image'].copy()
            coord = main.output['coord'].copy()
            release_semaphore_read(SEM)
            image = set_min_max(image, HMIN, HMAX)
            image = plot_detection(image, coord.astype(int), list_cross_hairs)
            image = cv2.resize(image, (2000, 2000), interpolation=cv2.INTER_AREA)
            frame = cv2.imencode(".jpg", image)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    @app.route('/stream')
    def video_feed():
        """
        Stream the image returned from gen().
        """
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/capture')
    def capture():
        """
        Stream the image returned from gen_capture().
        """
        return Response(gen_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/stream_text')
    def ion_number():
        """
        Stream the values obtained from processing.
        """
        acquire_semaphore_read(SEM)
        att = main.output.copy()
        release_semaphore_read(SEM)
        try:
            del att['image']
            del att['coord']
            del att['com_coord']
        except KeyError:
            pass

        return att


    @app.route('/get_cam_toggled_status')
    def toggled_status():
        current_status = request.args.get('status')
        if current_status == 'OFF':
            if cam is not None:
                cam.record(number_of_images=4, mode='fifo')
            main.fetch_from_camera = True
        else:
            if cam is not None:
                cam.stop()
            main.fetch_from_camera = False

        return 'ON' if current_status == 'OFF' else 'OFF'


    @app.route('/get_ion_toggled_status')
    def ion_toggled_status():
        current_status = request.args.get('status')
        if current_status == 'OFF':
            main.ion_detection = True
        else:
            main.ion_detection = False

        return 'ON' if current_status == 'OFF' else 'OFF'


    @app.route('/set_exposure_time')
    def set_exposure_time():
        exp_time = request.args.get('newExp')

        if cam is not None:
            cam.set_exposure_time(float(exp_time))

            change_settings('EXP_TIME', float(exp_time))

            time.sleep(0.5)

        return exp_time


    @app.route('/ROI/stream')
    def video_feed_roi():
        """
        Stream the image returned from gen_roi().
        """
        return Response(gen_roi(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/ROI')
    def roi_index():
        return render_template('setROI.html', decode_responses=True)


    @app.route('/ROI/update')
    def set_roi():
        """
        Change region of interest.
        """
        acquire_semaphore_read(SEM)
        image = main.output['image'].copy()  # read image
        release_semaphore_read(SEM)

        x1, y1, x2, y2 = request.args.values()  # load values from UI
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        x1 *= image.shape[1]/700  # scale values
        x2 *= image.shape[1]/700
        y1 *= image.shape[0]/700
        y2 *= image.shape[0]/700

        # create region array
        region = np.rint(np.array([[x1, y1], [x2, y2]])).astype(np.int32)

        # update ROI in shared memory
        file_path = os.path.dirname(os.path.abspath(__file__)) + "\\data\\region.json"
        with open(file_path, "w") as f:
            data = {"ROI": region.tolist()}
            json.dump(data, f)

        return '[[{}, {}], [{}, {}]]'.format(x1, y1, x2, y2)


    @app.route('/cross_hair')
    def cross_hair():
        return render_template('cross_hair.html', decode_responses=True, data={'ch': list_cross_hairs['cross_hairs'],
                                                                            'color': list_cross_hairs['color']})


    @app.route('/cross_hair/add')
    def add_cross_hair():
        global list_cross_hairs
        acquire_semaphore_read(SEM)
        image = main.output['image'].copy()  # read image
        release_semaphore_read(SEM)

        x, y = request.args.values()  # load values from UI
        x, y = float(x), float(y)

        ch = np.rint(np.array([x*image.shape[1], y*image.shape[0]])/700).astype(np.int32)

        # get random color
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        color_rgb = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]

        # add cross hair and color to the global list
        list_cross_hairs['cross_hairs'].append(list(ch))
        list_cross_hairs['color'].append(color_rgb)

        # add it also to settings.py
        change_settings('CROSS_HAIRS', list_cross_hairs)

        return {'x': str(ch[0]), 'y': str(ch[1]), 'color': color_rgb, 'num': len(list_cross_hairs['cross_hairs'])}

    @app.route('/cross_hair/update')
    def update_cross_hair():
        global list_cross_hairs
        acquire_semaphore_read(SEM)
        image = main.output['image'].copy()
        release_semaphore_read(SEM)

        old_x, old_y, new_x, new_y, color = request.args.values()

        # Delete old crosshair
        idx = (np.array(list_cross_hairs['cross_hairs']) == [float(old_x), float(old_y)]).any(axis=1).nonzero()[0][0]

        del list_cross_hairs['cross_hairs'][idx]
        del list_cross_hairs['color'][idx]

        # Add new crosshair
        color = ast.literal_eval(color[3:])

        color_rgb = [int(color[0]), int(color[1]), int(color[2])]

        img_ch = np.array([float(new_x), float(new_y)])
        html_ch = np.rint(np.array([float(new_x)/image.shape[1], float(new_y)/image.shape[0]])*700).astype(np.int32)

        list_cross_hairs['cross_hairs'].append(list(img_ch))
        list_cross_hairs['color'].append(color_rgb)

        change_settings('CROSS_HAIRS', list_cross_hairs)

        return {'html_x': str(html_ch[0]), 'html_y': str(html_ch[1]), 'color': color_rgb}

    @app.route('/cross_hair/delete')
    def delete_cross_hair():
        """
        Change region of interest.
        """
        global list_cross_hairs

        x, y = request.args.values()

        idx = (np.array(list_cross_hairs['cross_hairs']) == [int(x), int(y)]).any(axis=1).nonzero()[0][0]

        del list_cross_hairs['cross_hairs'][idx]
        del list_cross_hairs['color'][idx]

        change_settings('CROSS_HAIRS', list_cross_hairs)

        return "0"


    @app.route('/hist/stream')
    def video_feed_hist():
        """
        Stream the image returned from gen().
        """
        return Response(gen_hist(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/hist/stream_img')
    def video_feed_hist_img():
        """
        Stream the image returned from gen().
        """
        return Response(gen_hist_img(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/hist')
    def hist_index():
        return render_template('hist.html', decode_responses=True, data={'min': HMIN, 'max': HMAX})


    @app.route('/tweak_hist')
    def hist_tweak():
        global HMIN, HMAX
        HMIN = float(request.args.get('min'))
        HMAX = float(request.args.get('max'))
        change_settings('HIST_MIN', HMIN)
        change_settings('HIST_MAX', HMAX)
        return '{}, {}'.format(HMIN, HMAX)

    @app.route('/rotate')
    def rotate_img():
        global RANGLE
        angle = request.args.get("angle")
        RANGLE = int(angle) + 90
        if RANGLE == 360:
            RANGLE = 0
        change_settings('RANGLE', RANGLE)
        return str(RANGLE)

    app.run(debug=False, host="0.0.0.0", port=5000)
