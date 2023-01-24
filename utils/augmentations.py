# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

from cProfile import label
from curses import endwin
import logging
import math
import random

import cv2
import numpy as np

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xyxy2xywh as xyxy2cxcywh, clip_coords
from utils.metrics import bbox_ioa, box_iou
import torch


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.3),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)],
                #bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
                )

            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            logging.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            #new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            new = self.transform(image=im)  # transformed
            #im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
            im = new['image']
        return im, labels

class AlbumentationsTemporal:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, num_frames):
        self.transform = None
        self.num_frames = num_frames
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement
            additional_targets = {f'image{i}':'image' for i in range(1, num_frames)}
                # A.Blur(p=0.01),
                # A.MedianBlur(p=0.3),
                # A.ToGray(p=0.01),
                # A.CLAHE(p=0.3),
                # A.RandomBrightnessContrast(p=0.3),
                # A.RandomGamma(p=0.0),
                # A.ImageCompression(quality_lower=75, p=0.0)
            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.3),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)],
                #bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
                #additional_targets=additional_targets
                )

            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            logging.info(colorstr('albumentations: ') + f'{e}')
        
        self.transformation_expression = "self.transform(image=ims[0], "
        for ti in range(1, self.num_frames):
            self.transformation_expression += f"image{ti}=ims[{ti}], "
        self.transformation_expression += ")"
        #self.transformation_expression += "bboxes=labels[:, 1:], class_labels=labels[:, 0])"

    def __call__(self, ims, labels, p=1.0):
        if self.transform and random.random() < p:
            # n_i, t, enddim = labels.shape
            # labels = labels.reshape(n_i*t, enddim).astype(np.float32)
            #LOGGER.info(f"img shape before albumentations adjustment {ims.shape}")
            try:
                new = eval(self.transformation_expression) #transformed
            except Exception as e:
                LOGGER.critical(f"Error occured {self.transformation_expression}, {labels[:, 1:]}, {str(e)}")
                exit()
            #new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            ims = [new['image']] + [new[f'image{ti}'] for ti in range(1, self.num_frames)]
            #labels = [np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])]
            
            ims = np.stack(ims, 0) # T X H X W X C
            #labels = np.concatenate(labels, 0) #n_i*t X 5
            #labels = np.reshape(n_i, t, enddim)
            #LOGGER.info(f"img shape after albumentations adjustment {ims.shape}")
            
        return ims, labels

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def augment_hsv_temporal(im, hgain=0.5, sgain=0.5, vgain=0.5, frame_wise_aug=False):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        dtype = im.dtype  # uint8
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        for ti in range(len(im)):
            if frame_wise_aug:
                r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
                dtype = im.dtype  # uint8
                x = np.arange(0, 256, dtype=r.dtype)
                lut_hue = ((x * r[0]) % 180).astype(dtype)
                lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
                lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            hue, sat, val = cv2.split(cv2.cvtColor(im[ti], cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im[ti])

def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def letterbox_temporal(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im[0].shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        for ti in range(len(im)):
            im[ti] = cv2.resize(im[ti], new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    for ti in range(len(im)):
        im[ti] = cv2.copyMakeBorder(im[ti], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def random_perspective_temporal(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0), frame_wise_aug=False):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    if frame_wise_aug:
        max_n = -1
        _, t, enddim = targets.shape
        new_images, new_labels = [], []
        for ii in range(t):
            label_ = targets[:, ii, :]
            image, label_ = random_perspective(im[ii], label_, segments=segments, degrees=degrees, translate=translate, scale=scale, shear=shear, perspective=perspective, border=border)
            new_images.append(image)
            new_labels.append(label_) # n x 5
            max_n = len(label_) if len(label_) > max_n else max_n
        
        new_labels_ = np.zeros((max_n, t, enddim), dtype=np.float32)
        for ti, label_ in enumerate(new_labels):
            n, enddim = label_.shape
            new_labels_[:n, ti, :] = label_
        new_images = np.stack(new_images, 0)
        #print(new_images.shape)
        return new_images, new_labels_
            

    t, h, w, c = im.shape
    height = h + border[0] * 2  # shape(h,w,c)
    width = w + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -w / 2  # x translation (pixels)
    C[1, 2] = -h / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        new_images = []
        if perspective:
            for ii in range(len(im)):
                new_images.append(cv2.warpPerspective(im[ii], M, dsize=(width, height), borderValue=(114, 114, 114)))
        else:  # affine
            for ii in range(len(im)):
                new_images.append(cv2.warpAffine(im[ii], M[:2], dsize=(width, height), borderValue=(114, 114, 114)))
        new_images = np.stack(new_images, 0)
        assert len(new_images.shape) == 4
        im = new_images
    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    
    n_instance, t, enddim = targets.shape
    #LOGGER.info(f"before warping , {targets.shape}")
    targets = targets.reshape(n_instance*t, enddim)
    #LOGGER.info(f"before warping after reshaping , {targets.shape}")
    n = len(targets)
    if n:
        #segments not recoded
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine
                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        #LOGGER.info(f"warping instances {i}")
        i = i.reshape(n_instance, t)
        i_instance = np.prod(i, axis=-1).astype(bool)
        new = new.reshape(n_instance, t, -1)
        targets = targets.reshape(n_instance, t, enddim)
        new_targets = []
        for ni, ii in enumerate(i_instance):
            if ii:
                for ti in range(t):
                    tt = [targets[ni, ti, 0]] + new[ni, ti, :].tolist() if i[ni, ti] else [0.]*4
                    new_targets.append(tt)
        targets = np.array(new_targets).reshape(-1, t, enddim).astype(np.float32)
    #LOGGER.info(f"after warping , {targets.shape}")
    #targets = targets.astype(np.float32).reshape(n_instance, t, enddim)
    return im, targets


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments

def make_cuboid_from_temporal_annotation(labels):
    #Labels of form n X T X 4 with x1,y1,x2,y2 format convert to n X 4
    n, t = labels.shape[:2]
    labels = labels.reshape(n*t, 4)
    labels_with_wh = xyxy2cxcywh(labels)
    labels_with_wh = labels_with_wh.reshape(n, t, 4)[..., 2:]
    labels = labels.reshape(n, t, 4)
    new_labels = []
    for ni in range(n):
        temporal_candidates = labels_with_wh[ni].all(axis=-1)
        labels_at_n = labels[ni, temporal_candidates].reshape(-1, 4)
        x1, y1, x2, y2 = labels_at_n[:, 0].min(), labels_at_n[:, 1].min(), labels_at_n[:, 2].max(), labels_at_n[:, 3].max()
        new_labels.append([x1, y1, x2, y2])
    new_labels = np.array(new_labels).reshape(-1, 4)
    assert new_labels.shape[0] == n, "in cuboid formation number of instances not matching"
    return new_labels

def mixup_drones(im, labels1, im2, labels2):
    #Labels of form n X T X 5 with x1,y1,x2,y2 format
    h,w,c = im[-1].shape
    cuboid_labels1, cuboid_labels2 = make_cuboid_from_temporal_annotation(labels1[:, :, 1:]), make_cuboid_from_temporal_annotation(labels2[:, :, 1:])
    cuboid_labels1, cuboid_labels2 = torch.tensor(cuboid_labels1), torch.tensor(cuboid_labels2)
    ious = box_iou(cuboid_labels2, cuboid_labels1).numpy()
    mergable_candidates = ~ious.any(axis=-1)
    labels2 = labels2[mergable_candidates]
    n2, t, enddim = labels2.shape
    labels2[..., [1, 3]] = labels2[..., [1, 3]].clip(0, w)  # x1, x2
    labels2[..., [2, 4]] = labels2[..., [2, 4]].clip(0, h)  # y1, y2
    r = np.random.beta(32.0, 32.0)
    if n2:
        for ti in range(t):
            for ni in range(n2):
                x1, y1, x2, y2 = labels2[ni, ti, 1:]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                im[ti][y1:y2, x1:x2, :] = (r*im[ti][y1:y2, x1:x2, :] + (1-r)*im2[ti][y1:y2, x1:x2, :]).astype(np.uint8)
        labels = np.concatenate((labels1, labels2), 0).reshape(-1, t, enddim)
    else:
        labels = labels1

    return im, labels



def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels

def mixup_temporal(im, labels, im2, labels2, frame_wise_aug=False):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    t = len(im)
    for ti in range(t):
        if frame_wise_aug: r = np.random.beta(32.0, 32.0)
        im[ti] = (im[ti] * r + im2[ti] * (1 - r)).astype(np.uint8)
    enddim = labels.shape[-1]
    labels = np.concatenate((labels, labels2), 0).reshape(-1, t, enddim)
    
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
