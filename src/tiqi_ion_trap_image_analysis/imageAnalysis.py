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

    # Define ion intensity threshold for bright/dark classification
    ION_THRESHOLD = getattr(settings, 'ION_THRESHOLD', 1000)

    # Hamamatsu C15550-20UP (ORCA-Quest) conversion factor: ADU to photoelectrons
    # This camera uses qCMOS technology with photon number resolving capability
    ADU_TO_PHOTOELECTRONS = 0.107  # electrons per count (verified from technical specs)

    # Camera specifications:
    # - Ultra-low readout noise: 0.27 electrons rms (@Ultra quiet scan)
    # - Individual pixel calibration and real-time correction
    # - World's first photon number resolving qCMOS camera

    # Background noise storage
    background_image = None
    background_captured = False

    # Time series data storage for graphs
    photoelectron_history = {'timestamps': [], 'counts': [], 'roi_data': {}}

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
        Return the photoelectron histogram (frequency vs photoelectrons)
        """
        while True:
            # Load image and coords from shared memory
            acquire_semaphore_read(SEM)
            image = main.output['image'].copy()
            release_semaphore_read(SEM)

            # Convert image to photoelectrons and create histogram
            hist_image = plot_photoelectron_hist(image)
            frame = cv2.imencode(".jpg", hist_image)[1].tobytes()
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


    # Multiple ROI management
    roi_regions = {}

    @app.route('/roi/<int:roi_id>/stream')
    def roi_stream(roi_id):
        """
        Stream the cropped image for a specific ROI region
        """
        def gen_dynamic_roi():
            while True:
                if roi_id in roi_regions:
                    region = roi_regions[roi_id]

                    # Load image and coordinates
                    acquire_semaphore_read(SEM)
                    image = main.output['image'].copy()
                    coord = main.output['coord'].copy()
                    release_semaphore_read(SEM)

                    image = set_min_max(image, HMIN, HMAX)
                    image = plot_detection(image, coord.astype(int), list_cross_hairs)

                    # Scale coordinates from display to actual image size
                    scale_x = image.shape[1] / region['display_width']
                    scale_y = image.shape[0] / region['display_height']

                    x1 = int(region['x'] * scale_x)
                    y1 = int(region['y'] * scale_y)
                    x2 = int((region['x'] + region['width']) * scale_x)
                    y2 = int((region['y'] + region['height']) * scale_y)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, image.shape[1]))
                    y1 = max(0, min(y1, image.shape[0]))
                    x2 = max(x1, min(x2, image.shape[1]))
                    y2 = max(y1, min(y2, image.shape[0]))

                    if x2 > x1 and y2 > y1:
                        cropped_image = image[y1:y2, x1:x2]
                        cropped_image = rotate(cropped_image, RANGLE)
                        frame = cv2.imencode(".jpg", cropped_image)[1].tobytes()
                    else:
                        # Create a small placeholder image if ROI is invalid
                        placeholder = np.zeros((50, 50, 3), dtype=np.uint8)
                        frame = cv2.imencode(".jpg", placeholder)[1].tobytes()
                else:
                    # Create a small placeholder image if ROI doesn't exist
                    placeholder = np.zeros((50, 50, 3), dtype=np.uint8)
                    frame = cv2.imencode(".jpg", placeholder)[1].tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)  # Small delay to prevent overwhelming

        return Response(gen_dynamic_roi(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/roi/<int:roi_id>/stats')
    def roi_stats(roi_id):
        """
        Get statistics for a specific ROI region
        """
        if roi_id not in roi_regions:
            return {'error': 'ROI not found'}

        region = roi_regions[roi_id]

        try:
            # Load image and coordinates
            acquire_semaphore_read(SEM)
            if 'image' not in main.output or 'coord' not in main.output:
                release_semaphore_read(SEM)
                return {
                    'bright_ions': 0,
                    'dim_ions': 0,
                    'dark_ions': 0,
                    'max_intensity': 0,
                    'photoelectron_count': 0,
                    'total_ions': 0,
                    'coordinates': [],
                    'error': 'Camera data not ready yet'
                }

            image = main.output['image'].copy()
            coord = main.output['coord'].copy()
            release_semaphore_read(SEM)
        except Exception as e:
            try:
                release_semaphore_read(SEM)
            except:
                pass
            return {
                'bright_ions': 0,
                'dim_ions': 0,
                'dark_ions': 0,
                'max_intensity': 0,
                'photoelectron_count': 0,
                'total_ions': 0,
                'coordinates': [],
                'error': f'Error accessing camera data: {str(e)}'
            }

        # Scale coordinates from display to actual image size
        scale_x = image.shape[1] / region['display_width']
        scale_y = image.shape[0] / region['display_height']

        x1 = int(region['x'] * scale_x)
        y1 = int(region['y'] * scale_y)
        x2 = int((region['x'] + region['width']) * scale_x)
        y2 = int((region['y'] + region['height']) * scale_y)

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, image.shape[1]))
        y1 = max(0, min(y1, image.shape[0]))
        x2 = max(x1, min(x2, image.shape[1]))
        y2 = max(y1, min(y2, image.shape[0]))

        # Filter coordinates within this ROI
        roi_coords = []
        for c in coord:
            if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
                roi_coords.append(c)

        # Count ions by intensity using the threshold
        bright_ions = 0
        dim_ions = 0
        dark_ions = 0

        for c in roi_coords:
            if len(c) >= 3:  # Assuming coordinate format includes intensity
                intensity = c[2] if len(c) > 2 else 0
                if intensity > ION_THRESHOLD * 1.5:  # Bright threshold
                    bright_ions += 1
                elif intensity > ION_THRESHOLD * 0.5:  # Dim threshold
                    dim_ions += 1
                else:  # Dark threshold
                    dark_ions += 1
            else:
                # If no intensity info, classify based on pixel value at coordinate
                if 0 <= c[0] < image.shape[1] and 0 <= c[1] < image.shape[0]:
                    pixel_intensity = image[int(c[1]), int(c[0])]
                    if pixel_intensity > ION_THRESHOLD * 1.5:
                        bright_ions += 1
                    elif pixel_intensity > ION_THRESHOLD * 0.5:
                        dim_ions += 1
                    else:
                        dark_ions += 1

        # Get max intensity and photoelectron count in ROI
        if x2 > x1 and y2 > y1:
            roi_image = image[y1:y2, x1:x2]
            max_intensity = int(np.max(roi_image)) if roi_image.size > 0 else 0
            photoelectron_count = calculate_photoelectrons(roi_image) if roi_image.size > 0 else 0
        else:
            max_intensity = 0
            photoelectron_count = 0

        roi_coords = np.array(roi_coords) if roi_coords else np.array([])

        return {
            'bright_ions': bright_ions,
            'dim_ions': dim_ions,
            'dark_ions': dark_ions,
            'max_intensity': max_intensity,
            'photoelectron_count': photoelectron_count,
            'total_ions': len(roi_coords),
            'coordinates': roi_coords.tolist() if len(roi_coords) > 0 else []
        }

    @app.route('/roi/register', methods=['POST'])
    def register_roi():
        """
        Register a new ROI region
        """
        data = request.json
        roi_id = data['id']
        roi_regions[roi_id] = {
            'x': data['x'],
            'y': data['y'],
            'width': data['width'],
            'height': data['height'],
            'display_width': data['display_width'],
            'display_height': data['display_height']
        }
        return {'success': True}

    @app.route('/roi/<int:roi_id>/delete', methods=['POST'])
    def delete_roi(roi_id):
        """
        Delete an ROI region
        """
        if roi_id in roi_regions:
            del roi_regions[roi_id]
        return {'success': True}

    @app.route('/set_ion_threshold')
    def set_ion_threshold():
        """
        Set the ion intensity threshold for bright/dark classification
        """
        global ION_THRESHOLD
        threshold = request.args.get('threshold')
        if threshold:
            ION_THRESHOLD = float(threshold)
            change_settings('ION_THRESHOLD', ION_THRESHOLD)
        return str(ION_THRESHOLD)

    @app.route('/full_camera_stats')
    def full_camera_stats():
        """
        Get statistics for the full camera view including photoelectron counts
        """
        try:
            # Load image
            acquire_semaphore_read(SEM)
            if 'image' not in main.output or 'coord' not in main.output:
                release_semaphore_read(SEM)
                return {
                    'photoelectron_count': 0,
                    'bright_ions': 0,
                    'dim_ions': 0,
                    'dark_ions': 0,
                    'total_ions': 0,
                    'max_intensity': 0,
                    'error': 'Camera data not ready yet'
                }

            image = main.output['image'].copy()
            coord = main.output['coord'].copy()
            release_semaphore_read(SEM)
        except Exception as e:
            try:
                release_semaphore_read(SEM)
            except:
                pass
            return {
                'photoelectron_count': 0,
                'bright_ions': 0,
                'dim_ions': 0,
                'dark_ions': 0,
                'total_ions': 0,
                'max_intensity': 0,
                'error': f'Error accessing camera data: {str(e)}'
            }

        # Calculate photoelectron count using proper conversion and background subtraction
        photoelectron_count = calculate_photoelectrons(image)

        # Count bright, dim, and dark ions based on intensity threshold
        bright_ions = 0
        dim_ions = 0
        dark_ions = 0

        for c in coord:
            if len(c) >= 3:  # Assuming coordinate format includes intensity
                intensity = c[2] if len(c) > 2 else 0
                if intensity > ION_THRESHOLD * 1.5:  # Bright threshold
                    bright_ions += 1
                elif intensity > ION_THRESHOLD * 0.5:  # Dim threshold
                    dim_ions += 1
                else:  # Dark threshold
                    dark_ions += 1
            else:
                # If no intensity info, classify based on pixel value at coordinate
                if 0 <= c[0] < image.shape[1] and 0 <= c[1] < image.shape[0]:
                    pixel_intensity = image[int(c[1]), int(c[0])]
                    if pixel_intensity > ION_THRESHOLD * 1.5:
                        bright_ions += 1
                    elif pixel_intensity > ION_THRESHOLD * 0.5:
                        dim_ions += 1
                    else:
                        dark_ions += 1

        return {
            'photoelectron_count': photoelectron_count,
            'bright_ions': bright_ions,
            'dim_ions': dim_ions,
            'dark_ions': dark_ions,
            'total_ions': len(coord),
            'max_intensity': int(np.max(image)) if image.size > 0 else 0
        }

    @app.route('/capture_background', methods=['POST'])
    def capture_background():
        """
        Capture the current image as background for noise subtraction
        """
        global background_image, background_captured
        try:
            # Load current image
            acquire_semaphore_read(SEM)
            current_image = main.output['image'].copy()
            release_semaphore_read(SEM)

            background_image = current_image.copy()
            background_captured = True

            return {'success': True, 'message': 'Background captured successfully'}
        except Exception as e:
            return {'success': False, 'message': f'Error capturing background: {str(e)}'}

    def calculate_photoelectrons(image):
        """
        Calculate photoelectron count using Hamamatsu C15550-20UP photon number resolving
        This camera has individual pixel calibration and real-time correction built-in
        """
        if background_captured and background_image is not None:
            # Subtract background noise (important for accurate photon counting)
            corrected_image = image.astype(np.float64) - background_image.astype(np.float64)
            # Ensure no negative values (physical constraint)
            corrected_image = np.maximum(corrected_image, 0)
        else:
            corrected_image = image.astype(np.float64)

        # The C15550-20UP has ultra-low readout noise (0.27 electrons rms)
        # which enables photon number resolving at the individual pixel level

        # Convert ADU to photoelectrons using verified Hamamatsu conversion factor
        # Each ADU count represents 0.107 photoelectrons
        photoelectron_count = np.sum(corrected_image) * ADU_TO_PHOTOELECTRONS

        return int(photoelectron_count)

    def calculate_photon_counts_per_pixel(image):
        """
        Calculate photon counts per pixel for histogram analysis
        Takes advantage of the C15550-20UP's photon number resolving capability
        """
        if background_captured and background_image is not None:
            corrected_image = image.astype(np.float64) - background_image.astype(np.float64)
            corrected_image = np.maximum(corrected_image, 0)
        else:
            corrected_image = image.astype(np.float64)

        # Convert each pixel value to photoelectrons
        # The camera's individual pixel calibration ensures accuracy
        photoelectron_image = corrected_image * ADU_TO_PHOTOELECTRONS

        return photoelectron_image

    def plot_photoelectron_hist(image):
        """
        Create a histogram showing frequency vs photoelectrons per pixel
        Uses C15550-20UP photon number resolving capability
        """
        try:
            # Use the dedicated photon counting function
            photoelectron_image = calculate_photon_counts_per_pixel(image)

            # Create histogram of photoelectron values per pixel
            # Range adjusted for typical photon counting applications
            max_photons = max(5, np.percentile(photoelectron_image.flatten(), 99))
            hist, bin_edges = np.histogram(photoelectron_image.flatten(), bins=50, range=(0, max_photons))
        except Exception as e:
            print(f"Error in photon counting: {e}")
            # Create fallback histogram with simple ADU values
            hist, bin_edges = np.histogram(image.flatten(), bins=50)

        # Create plot
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            fig = Figure(figsize=(8, 6))
            canvas = FigureCanvasAgg(fig)
        except ImportError:
            # Fallback: create a simple text-based histogram placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'Matplotlib not available', (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, 'Histogram cannot be displayed', (50, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return placeholder
        ax = fig.add_subplot(111)

        # Plot histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], alpha=0.7, color='blue')

        ax.set_xlabel('Photoelectrons per Pixel')
        ax.set_ylabel('Frequency')
        ax.set_title('Photoelectron Histogram')
        ax.grid(True, alpha=0.3)

        # Convert plot to image
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()

        # Use the newer matplotlib API
        try:
            # Try the newer method first
            buf = canvas.buffer_rgba()
            image_array = np.asarray(buf).reshape(int(height), int(width), 4)
            # Convert RGBA to RGB
            image_array = image_array[:, :, :3]
        except AttributeError:
            # Fallback to older method if available
            try:
                image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            except AttributeError:
                # Use alternative method
                buf = canvas.print_to_buffer()
                image_array = np.frombuffer(buf[0], dtype='uint8').reshape(int(height), int(width), 4)
                image_array = image_array[:, :, :3]

            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            return image_bgr

        except Exception as e:
            print(f"Error creating histogram plot: {e}")
            # Return a simple error message image
            error_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_image, 'Histogram Error', (200, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_image, str(e)[:50], (50, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return error_image

    @app.route('/photoelectron_graph')
    def photoelectron_graph():
        """
        Display photoelectrons vs time graphs for full camera and ROIs
        """
        return render_template('photoelectron_graph.html', decode_responses=True)

    @app.route('/photoelectron_data')
    def photoelectron_data():
        """
        Return time series data for photoelectron graphs
        """
        return photoelectron_history

    def update_photoelectron_history():
        """
        Update time series data with current photoelectron counts
        """
        global photoelectron_history, roi_regions
        current_time = time.time()

        try:
            # Get full camera photoelectron count
            acquire_semaphore_read(SEM)
            if 'image' not in main.output:
                release_semaphore_read(SEM)
                return  # Skip update if image not ready yet

            image = main.output['image'].copy()
            release_semaphore_read(SEM)
        except Exception as e:
            try:
                release_semaphore_read(SEM)
            except:
                pass
            print(f"Error updating photoelectron history: {e}")
            return

        full_camera_count = calculate_photoelectrons(image)

        # Add to history (keep last 100 points)
        photoelectron_history['timestamps'].append(current_time)
        photoelectron_history['counts'].append(full_camera_count)

        if len(photoelectron_history['timestamps']) > 100:
            photoelectron_history['timestamps'].pop(0)
            photoelectron_history['counts'].pop(0)

        # Update ROI data
        for roi_id in roi_regions:
            region = roi_regions[roi_id]

            # Scale coordinates from display to actual image size
            scale_x = image.shape[1] / region['display_width']
            scale_y = image.shape[0] / region['display_height']

            x1 = int(region['x'] * scale_x)
            y1 = int(region['y'] * scale_y)
            x2 = int((region['x'] + region['width']) * scale_x)
            y2 = int((region['y'] + region['height']) * scale_y)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0]))
            x2 = max(x1, min(x2, image.shape[1]))
            y2 = max(y1, min(y2, image.shape[0]))

            if x2 > x1 and y2 > y1:
                roi_image = image[y1:y2, x1:x2]
                roi_count = calculate_photoelectrons(roi_image)

                if roi_id not in photoelectron_history['roi_data']:
                    photoelectron_history['roi_data'][roi_id] = {'timestamps': [], 'counts': []}

                photoelectron_history['roi_data'][roi_id]['timestamps'].append(current_time)
                photoelectron_history['roi_data'][roi_id]['counts'].append(roi_count)

                # Keep last 100 points for each ROI
                if len(photoelectron_history['roi_data'][roi_id]['timestamps']) > 100:
                    photoelectron_history['roi_data'][roi_id]['timestamps'].pop(0)
                    photoelectron_history['roi_data'][roi_id]['counts'].pop(0)

    # Start background thread to update photoelectron history
    import threading
    def history_updater():
        while True:
            time.sleep(1)  # Update every second
            try:
                if 'photoelectron_history' in globals() and 'roi_regions' in globals():
                    update_photoelectron_history()
            except Exception as e:
                print(f"Error updating photoelectron history: {e}")

    history_thread = threading.Thread(target=history_updater, daemon=True)
    history_thread.start()

    app.run(debug=False, host="0.0.0.0", port=5000)
