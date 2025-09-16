"""Settings file"""
import numpy as np
import platform

# Port used for shared memory
ZMQ_PORT = 5555

# Operating system
SYSTEM = platform.system()

# Shared memory names
SHM_NAME = 'zwoioncameraone'

# Semaphore name (ion images)
SEM_NAME = 'pcocamdata'

# Influxdb info
DB_USER = 'qchub-admin'#'tiqi_rw'
DB_PWD = 'Be+=313nm'#'y7PHgZxGELw2v1FOn40AudLf9QyRhnkv'
DB_HOST = 'avalon.qchub.ch'#'influxdb.phys.ethz.ch'
DB_PORT = '8086'#443
DB_DATASET = 'qchub'#'tiqi_fiber'

# Minimum pixel distance between ions
ION_DIST = 5

# Relative intensity threshold to differentiate good/bad isotopes (w.r.t. to max intensity)
INT_THS = 0.1

# Number of iterations before restarting the position's memory
NUM_ITERATIONS = 2

# Shape of image
# SHAPE = (1000, 1000, 1)

# Type of image
TYPE = np.uint16

# Blob detection size
BLOB_SIZE = 4

# Threshold for peak detection
PEAK_THRESHOLD = 10

# Threshold for binarization before center of mass calculation for the ion position
# between 0 and 1, the smaller the more pixel will be used to calculate the center of mass
# only used for a single ion
COM_THRESHOLD= 1

# Region of interest of the image
# REGION = [[0, 1000], [0, 1000]]

# Radius of circle that captures the intensity of each ion
RADIUS = 7

# Radius of circle plotted on the image
RADIUS_PLT = 7

# Area threshold of biggest contour in image (for ion cloud detection)
AREA_THS = 7

# Transparency of the circle plotted in the html
ALPHA = 0.5

# Relation factor between pixels and distances
DIST_FACTOR = 1

# Histogram min and max values
HIST_MIN = 190.0
HIST_MAX = 500.0

# Rotation angle of image
RANGLE = 90

# Camera exposure time
EXP_TIME = 0.001

# Cross hairs
# CROSS_HAIRS = {'cross_hairs': [[576, 496]], 'color': [[209, 132, 13]]}
CROSS_HAIRS = {'cross_hairs': [], 'color': []}

