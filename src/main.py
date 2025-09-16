import settings

import tiqi_ion_trap_image_analysis as imageAnalysis
import HamamatsuCamera

print('connect camera')
cam = HamamatsuCamera.HamamatsuCameraPlugin(exposureTime=0.2)
imageAnalysis.imageAnalysis(cam, settings)