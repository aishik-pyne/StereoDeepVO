import os, numpy as np, cv2

class Sequence:
    def __init__(self, sequence=0,
                        disparity_setting={ 'window_size' : 5,
                                            'min_disp' : 32,
                                            'num_disp' : 112-32}):

        self.PATH = '/home/guardian/Datasets/KITTI/dataset/sequences/{}/'.format(str(sequence).zfill(2))
        self.disparity_setting = disparity_setting
        self.stereo = cv2.StereoBM_create(16,self.disparity_setting['window_size'])

    def get_image(self, filenumber, camera=0):
        """
        camera: left-0
                right-1
        """
        filename = self.PATH + 'image_{}/'.format(str(camera)) + '{}.png'.format(str(filenumber).zfill(6))
        assert (os.path.isfile(filename)), "File Not Found"
        return cv2.imread(filename)

    def get_disparity(self, filenumber):
        img_left = self.get_image(filenumber, camera=0)
        img_right = self.get_image(filenumber, camera=1)

        disparity = self.stereo.compute(img_left, img_right).astype(np.float32) / 16.0
        disparity = (disparity+self.disparity_setting['min_disp'])/self.disparity_setting['num_disp']
        return disparity
