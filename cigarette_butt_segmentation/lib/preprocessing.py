import json
import os
import numpy as np
from keras.utils import Sequence
from PIL import Image
from .utils import get_mask
import cv2
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

MEAN = 0.51
STD = 0.27


class ImageGenerator(Sequence):
    """Generate and transform images for butt segmentation task.

    __getitem__ returns len(tuple)==2, that contains batches of images and masks.

    __len__  returns available number of butches for this folder.

    Parameters
    ----------
       directory: str
            Path to folder 'images' and COCO annotation file
       batch: int
            Number of images in batch
       pipeline: list, optional
            This pipeline transforms images and masks.
            List should contain objects of augmentation classes(with postfix -Aug).
            Recommended order in pipeline:
                [ShadowsAug(), <-- comes first
                 DrawStripsAug(),
                 DrawThingsAug(),
                 BrightnessAndContrastAug(),
                 BlurAug(),
                 ZoomAug(),
                 RotationAug()]
            Default: None
        normalize: bool, optional
            If it's True, then output image has type float with range (0.0, 1.0)
            Default: True
        yield_image: bool, optional
            If it's True, then output image has type uint8
            Default: False
        probability: list, optional
            Every list item should represent probability, that corresponding
            transformation in pipeline will be applied to an image.
            If one_trans_per_image=True, then items represent relative weights of transformations.
            If it's None, then will be formed list with ones.
            Default: None
        one_trans_per_image: bool, optional
            If is's True, then only one randomly
            (according to weights in 'probability' list) chosen transformation will be applied to an image.
            Default: True

    """

    def __init__(self, directory, batch, pipeline=None,
                 normalize=False, yield_image=False, probability = None,
                 one_trans_per_image=True):
        self.dir = directory
        self.batch = batch
        self.images = os.listdir(f"{directory}/images")
        self.annotations = json.load(open(f"{directory}/coco_annotations.json", "r"))
        self.pipeline = pipeline
        self.normalize = normalize
        self.yield_image = yield_image
        self.probability = probability
        self.one_trans_per_image = one_trans_per_image

        if (self.probability is None) and (self.pipeline is not None):
            self.probability = np.zeros(len(self.pipeline)) + 1

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch)))

    def __getitem__(self, item):
        batch_x = list()
        batch_y = list()
        # define batch index borders
        start_bindex = item * self.batch
        end_bindex = (item + 1) * self.batch
        if end_bindex > len(self.images):
            end_bindex = len(self.images)

        for i in range(start_bindex, end_bindex):
            img_id, ext = self.images[i].split(".")
            # load image
            with Image.open(f"{self.dir}/images" + os.sep + self.images[i]) as img:
                if ext == 'png':
                    img = img.convert('RGB')
                img = np.asarray(img, dtype=np.uint8)
            # get corresponding mask
            mask = get_mask(int(img_id), self.annotations)
            # get bbox and polygon, that represent mask contour
            gt = self.annotations['annotations'][int(img_id)]
            mask_polygon = np.array(gt["segmentation"][0], dtype=np.int32).reshape((-1, 2))
            bbox = [int(x) for x in gt['bbox']]

            # augmentation
            params = {'mask_polygon': mask_polygon,
                      'bbox': bbox}
            if self.pipeline is not None:
                if self.one_trans_per_image:
                    dist = np.array(self.probability)
                    dist /= dist.sum()
                    item = np.random.choice(self.pipeline, p=dist)
                    img, mask = item(img, mask, **params)
                    assert img.shape == (512, 512, 3)
                    assert mask.shape == (512, 512)
                else:
                    for item, p in zip(self.pipeline, self.probability):
                        if np.random.uniform(0,1) < p:
                            img, mask = item(img, mask, **params)
                            assert img.shape == (512, 512, 3)
                            assert mask.shape == (512, 512)

            batch_x.append(img)
            batch_y.append(mask)
        # normalization
        if not self.yield_image:
            batch_x = np.array(batch_x, dtype=np.float32)
            batch_y = np.array(batch_y, dtype=np.float32)
            batch_y = np.expand_dims(batch_y, axis=-1)
            batch_x = batch_x / 255.0
            batch_y = batch_y / 255.0
            if self.normalize:
                batch_x = (batch_x - MEAN)/STD


        return batch_x, batch_y


class RotationAug:
    """Rotate image around the image center.

    Parameters
    ----------
        angle: tuple, int, optional
            Tuple represents minimum and maximum angles.
            If int, then random angle will be chosen from range (-int, int)
            If it's None, then angle=0.
            Default: None
        flip_x, flip_y: bool, optional
            Either image will be flipped or not.
            Actually, it will be flipped only with 50% probability.
            Default: True

    """

    def __init__(self, angle=None, flip_x=True, flip_y=True):
        self.angle = angle
        self.flip_x = flip_x
        self.flip_y = flip_y

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"
        elif self.angle is not None:
            self.angle = (-self.angle, self.angle)

    def __call__(self, image, mask, angle=None, **kwargs):
        image = np.copy(image)
        mask = np.copy(mask)
        if angle is None:
            if self.angle is not None:
                angle = np.random.uniform(*self.angle)
            else:
                angle = 0

        if self.flip_x:
            if np.random.uniform(0, 1) > 0.5:
                image = image[:,::-1, :]
                mask = mask[:,::-1]
        if self.flip_y:
            if np.random.uniform(0, 1) > 0.5:
                image = image[::-1,:, :]
                mask = mask[::-1,:]
        if angle != 0:
            rot_image = self.rotate_im(image, angle)
            image = cv2.resize(rot_image, image.shape[:2])

            mask = self.rotate_im(mask, angle)
            mask = cv2.resize(mask, image.shape[:2])

        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        return image, mask

    @staticmethod
    def rotate_im(image, angle):
        """Rotate the image.

        Rotate the image such that the rotated image is enclosed inside the tightest
        rectangle. The area not occupied by the pixels of the original image is colored
        black.

        https://github.com/Paperspace/DataAugmentationForObjectDetection

        Parameters
        ----------

        image : numpy.ndarray
            numpy image

        angle : float
            angle by which the image is to be rotated

        Returns
        -------

        numpy.ndarray
            Rotated Image

        """
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = cv2.warpAffine(image, M, (nW, nH))

        return image


class ShadowsAug:
    """Draw shadow with random direction and length for butt on the picture.

    Notes
    -----
        sun_height: tuple, scalar, optional
            Parameter randomly varies in specified range.
            This parameters visually correspond to the Sun height above horizon.
            Default: (0.1, 2)
        sigma: tuple, scalar, optional
            Range for gaussian filter's sigma parameter.
            Gaussian filtration is used for shadow blurring.
        transparency: scalar, optional
            Transparency of shadows.
            1.0 - transparent
            0.0 - opaque
            Default: 0.5
        shadow_on_butt: bool, optional
            Try to imitate shadows of a cylindrical form.

                Sun
                |||
            _________________
            |________________|
            |\\\\\\\\\\\\\\\\|

            There is half of the butt in the shadow.

    """

    def __init__(self, sun_height=(0.1, 2), sigma=(2, 5), transparency=0.5,
                 shadow_on_butt=True):
        self.sun_height = sun_height
        self.sigma = sigma
        self.transparency = transparency
        self.shadow_on_butt = shadow_on_butt

        if type(self.sun_height) == tuple:
            assert len(self.sun_height) == 2, "Invalid range"
        else:
            self.sun_height = (0, self.sun_height)

        if type(self.sigma) == tuple:
            assert len(self.sigma) == 2, "Invalid range"
        else:
            self.sigma = (0, self.sigma)

    def __call__(self, image, mask, mask_polygon, direction=None, sun_height=None, butt_height=None,
                 sigma=None, **kwargs):
        image = np.copy(image)
        mask = np.copy(mask)
        # direction
        if direction is None:
            direction = np.random.uniform(-1, 1, size=2)
            direction /= np.linalg.norm(direction)
        if sun_height is None:
            sun_height = np.random.uniform(*self.sun_height)
        if sigma is None:
            sigma = np.random.uniform(*self.sigma)

        # get the min area rect
        rect = cv2.minAreaRect(mask_polygon)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)

        dm = distance_matrix(box, box)
        # butt height is proportional to butt width
        butt_height = np.min(dm[dm > 0]) * np.random.uniform(0.2, 1)
        # get projections on the surface
        shadow_poly = np.int0(box + direction * butt_height / sun_height)
        # include points, that already on the surface
        shadow_poly = np.vstack((shadow_poly, box))
        # we should create convex hull
        shadow_poly = cv2.convexHull(shadow_poly)
        shadow_poly = shadow_poly.reshape(-1, 2)

        # draw shadows on background
        # draw shadows on a blank image
        w, h = image.shape[:2]
        shadows = np.zeros((w, h), dtype=np.uint8) + 255
        # make shadows semi-transparent
        shadow_color = int(255 * self.transparency)

        shadows = cv2.fillPoly(shadows, [shadow_poly], color=shadow_color)
        # blur shadows
        shadows = cv2.GaussianBlur(shadows, ksize=(-1, -1), sigmaX=sigma)
        shadows_mask = 255 - shadows
        # shadows shouldn't cast on butts
        shadows_mask = (1 - mask // 255) * shadows_mask
        # blend image with shadows
        image = np.asarray(image * (1 - np.expand_dims(shadows_mask, axis=-1) / 255),
                           dtype=np.uint8)
        # shadow on butt
        rshadow_width = 0.5 * (1 - sun_height / self.sun_height[1])
        # find butt point in shadow
        if self.shadow_on_butt:
            for i in range(box.shape[0]):
                temp = shadow_poly[np.all(shadow_poly == box[i], axis=1)]
                if temp.size == 0:
                    break
            else:
                self.shadow_on_butt = False
                # print("Warning: Can't cast shadow on butt. There are no support points in shadow.")

        if self.shadow_on_butt:
            pt1 = box[i]
            # find second point
            # nearest to the furthest to pt1 point.
            idx_furthest = np.argmax(dm[i])
            idx_nearest = np.argmin(dm[idx_furthest][dm[idx_furthest]>0])
            if idx_nearest >= idx_furthest:
                idx_nearest += 1
            pt2 = box[idx_nearest]
            # nearest to the pt1
            idx_nearest_pt1 = np.argmin(dm[i][dm[i]>0])
            if idx_nearest_pt1 >= i:
                idx_nearest_pt1 += 1
            pt3 = box[idx_nearest_pt1]
            # nearest to the pt2
            idx_nearest_pt2 = np.argmin(dm[idx_nearest][dm[idx_nearest] > 0])
            if idx_nearest_pt2 >= i:
                idx_nearest_pt2 += 1
            pt4 = box[idx_nearest_pt2]
            # find points, that correspond to our shadow on butt
            # pt1 and pt2 were founded
            pt5 = pt1 + (pt3 - pt1) * rshadow_width
            pt6 = pt2 + (pt4 - pt2) * rshadow_width
            # pt1, pt2, pt5, pt6 are points of shadow on butt
            butt_shadow_poly = np.concatenate([pt1, pt2, pt6, pt5]).astype(np.int32).reshape(-1, 2)
            # draw shadows on butt
            # draw shadows on a blank image
            w, h = image.shape[:2]
            shadows = np.zeros((w, h), dtype=np.uint8) + 255
            # make shadows semi-transparent
            shadow_color = int(255 * self.transparency)

            shadows = cv2.fillPoly(shadows, [butt_shadow_poly], color=shadow_color)
            # blur shadows
            shadows = cv2.GaussianBlur(shadows, ksize=(-1, -1), sigmaX=sigma)
            # blend image with shadows
            casted_image = image.astype(np.float64) * (np.expand_dims(shadows, axis=-1) / 255)
            image = casted_image.astype(np.uint8)

        return image, mask


class DrawStripsAug:
    """Draw crossing butt stripes with mean background color.

        Parameters
        ----------
            strips: int
                Number of strips will be drawn.
            rwidth, rheight: tuple, scalar
                Sizes of a stripe is calculated relative to sizes if a butt.
            draw_on_mask: bool
                If it's True, then stripes will also be subtracted from a mask of the butt.
    """
    def __init__(self, strips=1, rwidth=(0, 0.1), rheight=(-0.9, 1), draw_on_mask=False):
        self.strips = strips
        self.rwidth = rwidth
        self.rheight = rheight
        self.draw_on_mask = draw_on_mask

        if type(self.rwidth) == tuple:
            assert len(self.rwidth) == 2, "Invalid range"
        else:
            self.rwidth = (0, self.rwidth)

        if type(self.rheight) == tuple:
            assert len(self.rheight) == 2, "Invalid range"
        else:
            self.rheight = (-self.rheight, self.rheight)

    def __call__(self, image, mask, bbox,
                 rwidth=None, rheight=None, **kwargs):
        image = np.copy(image)
        mask = np.copy(mask)

        # center coordinates of bbox
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        # draw strips
        for i in range(self.strips):
            if rwidth is None:
                rwidth = np.random.uniform(*self.rwidth)

            if rheight is None:
                rheight = np.random.uniform(*self.rheight)
            # get random shift and rotation
            rot_angle = np.random.randint(0, 180)
            shift_x = np.random.randint(-bbox[2] // 2, bbox[2] // 2)
            shift_y = np.random.randint(-bbox[3] // 2, bbox[3] // 2)
            # get strip parameters
            strip_width = max(bbox[2], bbox[3]) * rwidth
            strip_height = max(bbox[2], bbox[3]) * (1 + rheight)
            strip = np.array([strip_width, strip_height])
            # get shifted strip's center and corners
            shifted = np.array([cx, cy]) + np.array([shift_x, shift_y])
            pt1 = shifted - strip / 2
            pt2 = shifted + strip / 2
            pt3 = pt1 + np.array([strip[0], 0])
            pt4 = pt1 + np.array([0, strip[1]])
            points = np.concatenate((pt1, pt3, pt2, pt4), axis=0).reshape((-1, 2))
            # rotate
            rot_matrix = cv2.getRotationMatrix2D(tuple(shifted), rot_angle, 1)
            points = rot_matrix.dot(np.concatenate((points,
                                                    np.zeros(shape=(points.shape[0], 1)) + 1),
                                                   axis=-1).T)
            points = np.asarray(points.T, dtype=np.int32)
            # set color as background tone
            r = int((image[:, :, 0] * (1 - mask / 255)).mean())
            g = int((image[:, :, 1] * (1 - mask / 255)).mean())
            b = int((image[:, :, 2] * (1 - mask / 255)).mean())
            # draw strip on the image
            image = cv2.fillPoly(image,
                                 [points],
                                 color=(r, g, b))
            if self.draw_on_mask:
                mask = cv2.fillPoly(mask,
                                    [points],
                                    color=0)
            image = image.astype(np.uint8)
            mask = mask.astype(np.uint8)
        return image, mask


class BlurAug:
    """Blur image with gaussian filter.

        Parameters
        ----------
            ksize: iterable, int
                Define set of kernel sizes for gaussian filter computation.
                Items should be odd.
            only_butt: bool
                If true, than butt will be blurred only
            only_background: bool
                Background will be blurred only
                If both False, then whole image will be blurred.
                If both True, then butt will be blurred.

    """
    def __init__(self, ksize=(1, 3), only_butt=False, only_background=False):
        self.ksize = ksize
        self.only_butt = only_butt
        self.only_background = only_background

    def __call__(self, image, mask, ksize=5, **kwargs):
        image = np.copy(image)
        mask = np.copy(mask)
        if ksize is None:
            if type(self.ksize) == tuple:
                ksize = np.random.choice(self.ksize)
            else:
                ksize = self.ksize

        # blur channels
        blur_image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)
        # blur mask
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=0)

        norm_mask = np.expand_dims(mask, -1) / 255
        # if only butt need to blur, then subtract background
        if self.only_butt:
            image = np.int0(image * (1 - norm_mask) + blur_image * norm_mask)
        elif self.only_background:
            image = np.int0(blur_image * (1 - norm_mask) + image * norm_mask)
        else:
            image = blur_image

        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        return image, mask


class ZoomAug:
    """Zoom images trying to hold butt in frame.

        Parameters:
        -----------
            scale: tuple, int
                Scale factor.
                Tuple represents minimum and maximum values.
                If int, then random value will be chosen from range (-int, int).
            fill_mode: str
                Fill mode for numpy.pad function.
            fill_value: int
                Value that fills image's gaps when scale < 1 and fill_mode='constant'.

    """
    def __init__(self, scale=(0.8, 1.5), fill_mode='reflect', fill_value=0):
        self.scale = scale
        self.fill_mode = fill_mode
        self.fill_value = fill_value

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
        else:
            self.scale = (0, self.scale)

    def __call__(self, image, mask, bbox, scale=None, **kwargs):
        image = np.copy(image)
        mask = np.copy(mask)
        if scale is None:
            scale = np.random.uniform(*self.scale)
        # zoom channels
        zoomed_image = cv2.resize(image, None, fx=scale, fy=scale)
        # zoom mask
        zoomed_mask = cv2.resize(mask, None, fx=scale, fy=scale)
        # cropping and padding
        w, h = image.shape[:2]
        zw, zh = zoomed_image.shape[:2]
        if scale > 1:
            center = zw // 2, zh // 2
            borders_x = [center[0] - w // 2, center[0] + w // 2]
            borders_y = [center[1] - h // 2, center[1] + h // 2]
            if w % 2 != 0:
                borders_x[1] += 1
            if h % 2 != 0:
                borders_y[1] += 1
            # we want butt to be in frame
            pt1 = bbox[:2]
            pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            # zoom bbox
            pt1 = [int(x * scale) for x in pt1]
            pt2 = [int(x * scale) for x in pt2]
            # check if bbox is out of frame and shift
            if pt1[0] < borders_x[0]:
                shift = borders_x[0] - pt1[0]
                borders_x[0] -= shift
                borders_x[1] -= shift
            if pt1[1] < borders_y[0]:
                shift = borders_y[0] - pt1[1]
                borders_y[0] -= shift
                borders_y[1] -= shift
            if pt2[0] > borders_x[1]:
                shift = borders_x[1] - pt2[0]
                borders_x[1] -= shift
                borders_x[0] -= shift
            if pt2[1] > borders_y[1]:
                shift = borders_y[1] - pt2[1]
                borders_y[1] -= shift
                borders_y[0] -= shift
            # crop
            image = zoomed_image[int(borders_y[0]): int(borders_y[1]),
                    int(borders_x[0]): int(borders_x[1]), :]

            mask = zoomed_mask[int(borders_y[0]): int(borders_y[1]),
                   int(borders_x[0]): int(borders_x[1])]
        else:
            pad_width = w - zw
            if pad_width % 2 == 0:
                pad_left = pad_right = pad_width // 2
            else:
                pad_left = pad_width // 2
                pad_right = pad_left + 1

            if self.fill_mode == 'constant':
                pars = {'mode': self.fill_mode,
                        'constant_values': self.fill_value}
            else:
                pars = {'mode': self.fill_mode}

            image = np.pad(zoomed_image, ((pad_left, pad_right),
                                          (pad_left, pad_right),
                                          (0, 0)),
                           **pars)
            mask = np.pad(zoomed_mask, ((pad_left, pad_right),
                                        (pad_left, pad_right)),
                          **pars)

        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        return image, mask


class ContrastAndBrightnessAug:
    """Changes contrast and brightness of target image.
        
       Parameters
       ----------
            contrast: tuple, int
                Tuple represents minimum and maximum angles.
                If int, then random angle will be chosen from range (-int, int)
            brightness: tuple, int
                Tuple represents minimum and maximum angles.
                If int, then random angle will be chosen from range (-int, int)
            only_butt: bool
                If true, than butt will be changed only
            only_background: bool
                Background will be changed only
                If both False, then whole image will be changed.
                If both True, then butt will be changed.
    """
    def __init__(self, contrast=(-0.1, 0.1), brightness=(-0.1, 0.1), only_butt=False, only_background=False):
        self.only_butt = only_butt
        self.contrast = contrast
        self.brightness = brightness
        self.only_background = only_background

        if type(self.contrast) == tuple:
            assert len(self.contrast) == 2, "Invalid range"
        else:
            self.contrast = (-self.contrast, self.contrast)

        if type(self.brightness) == tuple:
            assert len(self.brightness) == 2, "Invalid range"
        else:
            self.brightness = (-self.brightness, self.brightness)

    def __call__(self, image, mask, contrast=None, brightness=None, **kwargs):
        image = np.copy(image)
        mask = np.copy(mask)
        if contrast is None:
            contrast = np.random.uniform(*self.contrast)

        if brightness is None:
            brightness = np.random.uniform(*self.brightness)
        # changing contrast and brightness here
        g_image = np.clip(np.int0(image * (1 + contrast) + brightness * np.mean(image)), 0, 255)
        # crop background if need to change butt only
        norm_mask = np.expand_dims(mask, -1) / 255
        if self.only_butt:
            image = np.int0(image * (1 - norm_mask) + g_image * norm_mask)
        elif self.only_background:
            image = np.int0(g_image * (1 - norm_mask) + image * norm_mask)
        else:
            image = g_image

        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        return image, mask


class DrawThingsAug:
    """
        Either draw images from a chosen dataset or draw monochromatic rectangles and place it
        onto original image.

        Parameters
        ----------
            path: str
                Path to the folder with images to be drawn
                If draw_simple is True, then 'path' can be None.
            quantity: int
                Number of things to draw.
            rwidth, rheight: tuple, scalar
                Ranges represent variation of images sizes relative to
                sizes of butt on original image.
                Sizes of a stripe is calculated relative to sizes if a butt.
            overlap: bool
                If it's True, then things can overlap butt.
                If it's False, then will try to find free area on the image and
                place thing on it if possible. If it's not possible, then image of that thing will be skipped.
            draw_simple: bool
                If it's True, then monochromatic rectangles with random color will be drawn
                instead of images from 'path'.

    """
    def __init__(self, path=None, quantity=1, rwidth=(0, 0.5), rheight=(0, 0.5), overlap=False,
                 draw_simple=True):
        self.quantity = quantity
        self.overlap = overlap
        self.rwidth = rwidth
        self.rheight = rheight
        self.path = path
        self.draw_simple = draw_simple

        if (not self.draw_simple) and (self.path is None):
            raise Exception("Need to specify path, if you want to draw sprites.")

        if type(self.rwidth) == tuple:
            assert len(self.rwidth) == 2, "Invalid range"
        else:
            self.rwidth = (0, self.rwidth)

        if type(self.rheight) == tuple:
            assert len(self.rheight) == 2, "Invalid range"
        else:
            self.rheight = (0, self.rheight)

    def __call__(self, image, mask, bbox,
                 rwidth=None, rheight=None, **kwargs):
        image = np.copy(image)
        mask = np.copy(mask)

        # center coordinates of bbox
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2

        images = os.listdir(self.path)

        # draw things
        for i in range(self.quantity):
            if rwidth is None:
                rwidth = np.random.uniform(*self.rwidth)

            if rheight is None:
                rheight = np.random.uniform(*self.rheight)
            # thing sizes
            width = int(max(bbox[2], bbox[3]) * (1 + rwidth))
            height = int(max(bbox[2], bbox[3]) * (1 + rheight))
            rot_angle = np.random.randint(0, 360)
            # choose image or color
            if self.draw_simple:
                thing = np.zeros(shape=(width, height, 3))
                color = np.random.randint(0, 255, 3)
                for channel in range(thing.shape[-1]):
                    thing[:, :, channel] = color[channel]
            else:
                thing = np.random.choice(images)
                with Image.open(self.path + os.sep + thing) as im:
                    thing = np.array(im)

                # simple by white color mask
                mask_thing = thing.mean(axis=-1, keepdims=True)[:, :, 0]
                thing = thing[np.any(mask_thing < 255, axis=1), :, :]
                thing = thing[:, np.any(mask_thing < 255, axis=0), :]
                # resize
                thing = cv2.resize(thing, (width, height))
            # rotate
            thing = RotationAug.rotate_im(thing, rot_angle)
            # shifting relative to bbox center
            if self.overlap:
                shift_x = np.random.randint(-bbox[2] // 2, bbox[2] // 2)
                shift_y = np.random.randint(-bbox[3] // 2, bbox[3] // 2)
            else:
                range_x1 = (bbox[2] // 2 + thing.shape[0] // 2, image.shape[0] - (cx + thing.shape[0] // 2))
                range_x2 = (-(cx - thing.shape[0] // 2), -(bbox[2] // 2 + thing.shape[0] // 2))
                range_y1 = (bbox[3] // 2 + thing.shape[1] // 2, image.shape[1] - (cy + thing.shape[1] // 2))
                range_y2 = (-(cy - thing.shape[1] // 2), -(bbox[3] // 2 + thing.shape[1] // 2))
                # if we can't place sprite on the image, then skip
                range_x = list()
                range_y = list()
                if range_x1[0] < range_x1[1]:
                    range_x.append(range_x1)
                if range_x2[0] < range_x2[1]:
                    range_x.append(range_x2)
                if len(range_x) == 0:
                    continue

                if range_y1[0] < range_y1[1]:
                    range_y.append(range_y1)
                if range_y2[0] < range_y2[1]:
                    range_y.append(range_y2)
                if len(range_y) == 0:
                    continue
                shift_x = np.random.randint(*range_x[np.random.randint(0, len(range_x))])
                shift_y = np.random.randint(*range_y[np.random.randint(0, len(range_y))])
            # place thing on canvas
            shift_x = cx + shift_x
            shift_y = cy + shift_y
            thing_canvas = np.zeros(shape=image.shape)
            start_x = int(np.clip(shift_x - thing.shape[1] / 2, 0, image.shape[1]))
            start_y = int(np.clip(shift_y - thing.shape[0] / 2, 0, image.shape[0]))
            end_x = int(np.clip(shift_x + thing.shape[1] / 2, 0, image.shape[1]))
            end_y = int(np.clip(shift_y + thing.shape[0] / 2, 0, image.shape[0]))

            thing_canvas[start_y:end_y, start_x: end_x, :] = thing[:end_y - start_y, :end_x - start_x]

            mask_thing = thing_canvas.mean(axis=-1, keepdims=True)
            if not self.draw_simple:
                mask_thing[mask_thing > 254] = 0
            mask_thing[mask_thing > 0] = 1
            # and then on the image
            image = image * (1 - mask_thing) + mask_thing * thing_canvas
            mask = (1 - mask_thing[:, :, 0]) * mask

            image = image.astype(np.uint8)
            mask = mask.astype(np.uint8)
        return image, mask


class DumbAug:
    def __call__(self, image, mask, **kwargs):
        return image, mask






if __name__ == '__main__':
    from show import show_img_with_mask

    # pipe = [ShadowsAug(), DrawStripsAug(), DrawThingsAug("../data/things", draw_simple=False),
    #         ContrastAndBrightnessAug(), BlurAug(), ZoomAug(), RotationAug()]
    # #pipe = [ShadowsAug()]
    #
    # data = ImageGenerator("../data/train", 1, pipeline=pipe, yield_image=True,
    #                       one_trans_per_image=False)
    # for i, m in data:
    #     #print(i[0], m[0])
    #     show_img_with_mask(i[0], m[0])
    #     input()
    mean = list()
    std = list()
    for batch in ImageGenerator('../data/train', 100):
        mean.append(batch[0].mean())
        std.append(batch[0].std())
    for batch in ImageGenerator('../data/val', 100):
        mean.append(batch[0].mean())
        std.append(batch[0].std())

    mean = np.array(mean).mean()
    std = np.array(std).mean()
    print(mean, std)
