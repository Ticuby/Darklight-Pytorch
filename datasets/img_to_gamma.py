#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 1 13:39:00 2021
This repository is authored by Jiajun Chen
"""
import cv2
import numpy as np

def gamma_intensity_correction(img, gamma):
    """
    :param img: the img of input
    :param gamma:
    :return: a new img
    """
    invGamma = 1.0 / gamma
    LU_table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_img = cv2.LUT(img, LU_table)
    return gamma_img

