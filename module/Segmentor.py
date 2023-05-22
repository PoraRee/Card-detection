import cv2
import math
import numpy as np
from numpy import random
import random as rng
from matplotlib import pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets
import urllib.request
import time

class Segmentor:
    def __init__(
        self,
        min_p=4,
        max_p=400,
        min_a=1000,
        max_a=20000,
        suit_x_ratio=2 / 7,
        suit_y_ratio=1 / 5,
    ):
        # For get_card_contour
        self.min_p = min_p
        self.max_p = max_p
        self.min_a = min_a
        self.max_a = max_a
        self.suit_x_ratio = suit_x_ratio
        self.suit_y_ratio = suit_y_ratio

    def euc(self, pt_1, pt_2):
        return np.sqrt((pt_1[0] - pt_2[0]) ** 2 + (pt_1[1] - pt_2[1]) ** 2)

    def get_card_contour(self, image):
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholdImage = cv2.threshold(
            imageGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        morphImage = cv2.morphologyEx(
            thresholdImage, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        cardContours, _ = cv2.findContours(
            morphImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cardContours = [
            contour
            for contour in cardContours
            if len(contour) >= self.min_p
            and len(contour) <= self.max_p
            and cv2.contourArea(contour) > self.min_a
            and cv2.contourArea(contour) < self.max_a
        ]
        return cardContours
    
    def mark_card_corner(self, cardContours):
        result = []
        for contour in cardContours:
            convex = cv2.convexHull(contour) # should be card contour
            eps = 0.1*cv2.arcLength(convex, True)
            approximate_hull = cv2.approxPolyDP(convex, eps, True)
            if approximate_hull.shape[0] != 4:
                continue
            approximate_hull = np.reshape(approximate_hull, (4,2))
            result.append(approximate_hull)
        return result

    def crop_cards(self, cardCorners, inputImage):
        result = []
        for corner in cardCorners:
            point_A = corner[0]
            point_B = corner[3]
            point_C = corner[2]
            point_D = corner[1]

            AD = self.euc(point_A, point_D)
            BC = self.euc(point_B, point_C)
            AB = self.euc(point_A, point_B)
            CD = self.euc(point_C, point_D)

            maxWidth = max(int(AD), int(BC))
            maxHeight = max(int(AB), int(CD))

            input_points = np.float32([point_A,point_B,point_C,point_D])
            output_points = np.float32([(0,0), (0, maxHeight-1), (maxWidth-1, maxHeight-1), (maxWidth-1, 0)])

            # transform card to axis-aligned rectangle
            transform = cv2.getPerspectiveTransform(input_points, output_points)
            output = cv2.warpPerspective(inputImage.copy(), transform, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

            # rotate card
            if output.shape[0] < output.shape[1]:
                output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
                
            # resize card
            output = cv2.resize(output, (80, 100), interpolation=cv2.INTER_AREA)
            result.append(output)

        return result
        
    
    def get_card_suits(self, cardResult):
        return [
            card[
                : int(card.shape[0] * self.suit_x_ratio),
                : int(card.shape[1] * self.suit_y_ratio),
            ]
            for card in cardResult
        ]

    def seg(self, image):
        cardContours = self.get_card_contour(image)
        cardCorners = self.mark_card_corner(cardContours)
        cardResult = self.crop_cards(cardCorners, image)
        cardSuits = self.get_card_suits(cardResult)
        
        return [{
            'bbox': corner,
            'card': result,
            'suit': suit
            } for corner, result, suit in zip(cardCorners, cardResult, cardSuits)] 


if __name__ == "__main__":
    inputImage = cv2.imread("bicyclecard/images/IMG_20230520_185914631.jpg")

    segmentor = Segmentor(
        min_p=4,
        max_p=400,
        min_a=1000,
        max_a=20000,
        suit_x_ratio=2 / 7,
        suit_y_ratio=1 / 5,
    )
    
    st = time.time()
    results = segmentor.seg(inputImage)
    en = time.time()
    print("usage time:", en - st)
    
    print(results[0]['bbox'])
    
    inputImageCorners = inputImage.copy()
    for result in results:
        for corner in result['bbox']:
            cv2.circle(inputImageCorners, corner, 3, (0, 0, 255), -1)
    cv2.imshow("Corners", inputImageCorners)

    cv2.waitKey(0)
