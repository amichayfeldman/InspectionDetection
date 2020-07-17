import numpy as np
import scipy.signal
from skimage.measure import label, regionprops, compare_ssim
import configparser
import cv2


def kl(p, q):
    """
    Computation of Kullback-Leibler distance.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def correlation_two_img(im1, im2):
    """
    Use FFT to compute correlation.
    """
    im1 = im1.astype(np.float64) - np.mean(im1)
    im2 = im2.astype(np.float64) - np.mean(im2)
    return scipy.signal.fftconvolve(im1, im2[::-1, ::-1], mode='same')


class FindInspection:
    def __init__(self):
        self.ref_path = None
        self.inspected_path = None
        self.ref_img = None
        self.inspected_img = None
        self.diff_map = None
        self.shift = np.array([0, 0])
        self.shifted_inspected_img = None
        self.new_ref_img = None

    def process(self, reference_img_path, suspected_img_path):
        self.ref_path = reference_img_path
        self.inspected_path = suspected_img_path
        self.ref_img = cv2.imread(reference_img_path, 0)
        self.inspected_img = cv2.imread(suspected_img_path, 0)

        self.__align_images()
        self.__compute_diff_mask()
        self.__compute_similarity()
        self.__reverse_shift_map()
        self.__save_diff_map()

        self.diff_map = None
        self.shift = np.array([0, 0])
        self.shifted_inspected_img = None
        self.new_ref_img = None

    def __compute_diff_mask(self):
        """
        Compute difference mask between two aligned input images.
        """
        morphology_iteration = config.getint('Params', 'morphology_iterations')
        binarization_thr = config.getint('Params', 'binarization_thr')

        ref_bilateral = cv2.bilateralFilter(src=self.new_ref_img, d=7, sigmaColor=40, sigmaSpace=40)
        inspected_bilateral = cv2.bilateralFilter(src=self.shifted_inspected_img, d=7, sigmaColor=40, sigmaSpace=40)
        # get differences map by XOR images:
        diff_map = cv2.bitwise_xor(src1=ref_bilateral, src2=inspected_bilateral)
        blured_diff_map = cv2.medianBlur(src=diff_map, ksize=5)  # help in filter weak diff like noise
        filtered_diff_map = (blured_diff_map > binarization_thr)
        eroded = cv2.erode(src=filtered_diff_map.astype('uint8'), kernel=(3, 3), iterations=morphology_iteration)
        median_on_eroded = cv2.medianBlur(src=eroded, ksize=7)
        self.diff_map = cv2.dilate(src=median_on_eroded, kernel=(3, 3), iterations=morphology_iteration)

    def __align_images(self):
        """
        Align inspected image to fit reference image. Computing is done by adjust computation of convolution (by FFT) to
        execute correlation computation.
        """

        ref_bilateral = cv2.bilateralFilter(src=self.ref_img, d=11, sigmaColor=80, sigmaSpace=80)
        inspected_bilateral = cv2.bilateralFilter(src=self.inspected_img, d=11, sigmaColor=80, sigmaSpace=80)
        correlation = correlation_two_img(im1=ref_bilateral, im2=inspected_bilateral)
        best_correlation = np.unravel_index(np.argmax(correlation), correlation.shape)
        self.shift = best_correlation - (np.array(self.ref_img.shape) / 2).astype(np.int)
        affine_mat = np.array([[1, 0, self.shift[1]], [0, 1, self.shift[0]]])
        self.shifted_inspected_img = cv2.warpAffine(src=self.inspected_img, M=affine_mat.astype(np.float64),
                                                    dsize=(self.ref_img.shape[1], self.ref_img.shape[0]))
        # mask irrelevant area in ref img:
        self.new_ref_img = self.ref_img.copy()
        w, h = self.shift[1], self.shift[0]
        if w >= 0:
            self.new_ref_img[:, :w] = 0
        else:
            self.new_ref_img[:, w:] = 0
        if h >= 0:
            self.new_ref_img[:h, :] = 0
        else:
            self.new_ref_img[h:, :] = 0

        cv2.imwrite('shifted_inspection3.png', self.shifted_inspected_img)
        cv2.imwrite('new_ref3.png', self.new_ref_img)

    def __compute_similarity(self):
        """
        For input map, get blobs locations, extract adjusted original images data and compare these
        histograms similarity.
        """
        similarity_thr = config.getfloat('Params', 'similarity_thr')

        labeled_map = label(input=self.diff_map, neighbors=8)
        for region in regionprops(labeled_map):
            ty, tx, by, bx = region.bbox
            if region.area < config.getint('Params', 'min_area'):  # min size of single blob
                self.diff_map[ty:by, tx:bx] = 0
                continue
            # Extract relevant rect from origin images:
            ref_patch = self.new_ref_img[ty:by, tx:bx]
            inspected_patch = self.shifted_inspected_img[ty:by, tx:bx]
            # comparison:
            ref_hist = cv2.calcHist(images=[ref_patch], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            inspected_hist = cv2.calcHist(images=[inspected_patch], channels=[0], mask=None, histSize=[256],
                                          ranges=[0, 256])
            norm_ref_hist = cv2.normalize(ref_hist, ref_hist).flatten()
            norm_inspected_hist = cv2.normalize(inspected_hist, inspected_hist).flatten()

            similarity_score = cv2.compareHist(norm_inspected_hist[10:], norm_ref_hist[10:],
                                               cv2.HISTCMP_CORREL)  # filter black BG pixels
            if similarity_score > similarity_thr:
                self.diff_map[ty:by, tx:bx] = 0

    def __save_diff_map(self):
        name = self.inspected_path[:self.inspected_path.find('_inspected')]
        new_name = name + '_diff_map.tif'
        if self.diff_map.max() <= 1:
            self.diff_map *= 255
        cv2.imwrite(filename=new_name, img=self.diff_map)

    def __reverse_shift_map(self):
        reverse_affine_mat = np.array([[1, 0, -self.shift[1]], [0, 1, -self.shift[0]]])
        self.diff_map = cv2.warpAffine(src=self.diff_map, M=reverse_affine_mat.astype(np.float64),
                                       dsize=(self.diff_map.shape[1], self.diff_map.shape[0]))


def main():
    inspection_detector = FindInspection()
    inspection_detector.process(reference_img_path=config.get('Params', 'ref_img_path'),
                                suspected_img_path=config.get('Params', 'inspected_img_path'))
    
    
if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    main()

