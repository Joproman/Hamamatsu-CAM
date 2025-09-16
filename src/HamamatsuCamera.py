# Class to control an Hamamatsu camera
# the function have been adjusted to be compatible with the PCO camera interface
# https://pypi.org/project/pco/#image
# such that it is compatible with this repo
# https://gitlab.phys.ethz.ch/tiqi-projects/gkp/devices/ion-trap-image-analysis/-/tree/molecule
# therefore some parameters don't really make sense for the Hamamatsu camera

from Hamamatsu import camera
import numpy as np 

class HamamatsuCameraPlugin():
    def __init__(self, exposureTime= 1.) -> None:
        self.cam = camera(0)
        self.cam.initializeCamera()
        self.cam.setExposure(exposureTime*1000)
        self.cam.setAcquisitionMode(self.cam.MODE_CONTINUOUS)
        self.cam.triggerCamera()
        self.image_obj= None

    def record(self, number_of_images= 1, mode= None) -> None:
        # self.cam.triggerCamera()
        image = self.cam.readCamera()
        # self.cam.stopAcq()
        if len(image) > 0: 
            self.image_obj= np.array(image[0])

    def image(self, image_number= 0xFFFFFFFF):
        assert image_number== 0xFFFFFFFF
        return self.image_obj, None
    
    def set_exposure_time(self, exposureTime) -> None:
        self.cam.setExposure(exposureTime*1000)

    def get_exposure_time(self) -> float:
        return self.cam.getExposure()

    def wait_for_first_image(self) -> None:
        self.record()

    def close(self) -> None:
        self.cam.stopCamera()

    def stop(self):
        self.cam.stopCamera()
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cam = HamamatsuCameraPlugin(exposureTime=0.2)
    print(cam)
    for i in range(1, 2):
        cam.record()
        img = cam.image()
        plt.imshow(img[0])
        plt.colorbar()
        plt.show()
    cam.close()
    
