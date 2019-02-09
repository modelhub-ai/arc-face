from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
import mxnet as mx
import sklearn
from sklearn.preprocessing import normalize
from mtcnn_detector import MtcnnDetector
from skimage import transform as trans
from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json

class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            # NOTE: for images will multiple faces, this model will only return
            # one vector.
            # switches PIL to cv2
            npArr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            det_threshold = [0.6,0.7,0.8]
            mtcnn_path = os.path.join(os.path.dirname('__file__'), 'model/mtcnn-model')
            ctx = mx.cpu()
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1,
            accurate_landmark = True, threshold=det_threshold)
            # Pass input images through face detector
            ret = detector.detect_face(npArr, det_type = 0)
            if ret is None:
                raise Exception("No face detected in input image.")
            bbox, points = ret
            if bbox.shape[0]==0:
                raise Exception("No face detected in input image.")
            bbox = bbox[0,0:4]
            points = points[0,:].reshape((2,5)).T
            # Call preprocess() to generate aligned images
            nimg = self._preprocess(npArr, bbox, points, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2,0,1))
            return aligned
        else:
            raise IOError("Image Type not supported for preprocessing.")

    def _preprocessAfterConversionToNumpy(self, npArr):
        # pass
        return npArr

    def computeOutput(self, inferenceResults):
        result = normalize(inferenceResults).flatten()
        return result

    def _preprocess(self, img, bbox=None, landmark=None, **kwargs):
        M = None
        image_size = []
        str_image_size = kwargs.get('image_size', '')
        # Assert input shape
        if len(str_image_size)>0:
            image_size = [int(x) for x in str_image_size.split(',')]
            if len(image_size)==1:
                image_size = [image_size[0], image_size[0]]
            assert len(image_size)==2
            assert image_size[0]==112
            assert image_size[0]==112 or image_size[1]==96

        # Do alignment using landmark points
        if landmark is not None:
            assert len(image_size)==2
            src = np.array([
              [30.2946, 51.6963],
              [65.5318, 51.5014],
              [48.0252, 71.7366],
              [33.5493, 92.3655],
              [62.7299, 92.2041] ], dtype=np.float32 )
            if image_size[1]==112:
                src[:,0] += 8.0
            dst = landmark.astype(np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2,:]
            assert len(image_size)==2
            warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]),
            borderValue = 0.0)
            return warped

        # If no landmark points available, do alignment using bounding box.
        # If no bounding box available use center crop
        if M is None:
            if bbox is None:
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1]*0.0625)
                det[1] = int(img.shape[0]*0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            margin = kwargs.get('margin', 44)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
            bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
            ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
            if len(image_size)>0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret
