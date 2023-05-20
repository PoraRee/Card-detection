import os
import cv2 as cv
from tqdm.auto import tqdm
import numpy as np
import json

IMG_PATH = "images/"
ANNOT_PATH = "annotated/"

images = []
annotations = []

def process(filename):
    print("processing image", filename)
    img_path = os.path.join(IMG_PATH, filename)
    img = cv.imread(img_path)
    
    if img.shape != (576, 1024, 3):
        print("resizing", img.shape, "to 1024x576")
        img = cv.resize(img, (1024, 576))
        cv.imwrite(img_path, img)
    
        
    imgEdge = np.sum(np.abs(np.gradient(cv.cvtColor(img, cv.COLOR_RGB2GRAY))), axis=0)/255
    levelSet = np.zeros(img.shape[:-1]) + 0.00001
    for i in range(0, levelSet.shape[0], 100):
        for j in range(0, levelSet.shape[1], 100):
            levelSet[i:i+50,j:j+50] = -0.00001

    kimg = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, :2]

    print("segmenting")
    for i in tqdm(range(50)):
        bgMean = kimg[levelSet>0].mean(axis=0)
        fgMean = kimg[levelSet<0].mean(axis=0)

        """
        if fgMean[1] > bgMean[1]:
            bgMean, fgMean = fgMean, bgMean
        """

        levelGradY, levelGradX = np.gradient(levelSet)
        gradNorm = np.sqrt(np.square(levelGradX) + np.square(levelGradY))
        levelGradY /= (gradNorm + 0.00001)
        levelGradX /= (gradNorm + 0.00001)
        colorGrad = -np.sqrt(np.sum(np.power(np.abs(kimg - bgMean), 2), axis=2))/255 + np.sqrt(np.sum(np.power(np.abs(kimg - fgMean), 2), axis=2))/255

        levelSet += 1.0 * (1-imgEdge) * np.abs(levelGradX) * colorGrad
        levelSet += 1.0 * (1-imgEdge) *np.abs(levelGradY) * colorGrad
        levelDiv = (levelSet[:-2, 1:-1] + levelSet[2:, 1:-1] + levelSet[1:-1, :-2] + levelSet[1:-1, 2:] - 4*levelSet[1:-1, 1:-1])/4
        levelSet += 1.0*np.pad(levelDiv, 1)

    mask = levelSet < 0
    contours, _ = cv.findContours(mask.astype('uint8'), cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    valid_contours = [cv.approxPolyDP(c, 5, True) for c in contours if cv.contourArea(c) > 0.005 * img.shape[0] * img.shape[1]]

    if (len(valid_contours) < 5):
        # Flipped dist
        mask = levelSet > 0
        contours, _ = cv.findContours(mask.astype('uint8'), cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
        valid_contours = [cv.approxPolyDP(c, 5, True) for c in contours if cv.contourArea(c) > 0.005 * img.shape[0] * img.shape[1]]
    cimg = img.copy()
    cv.drawContours(cimg, valid_contours, -1, (0,255,0), 3)
    cv.imwrite(os.path.join(ANNOT_PATH, filename), cimg)
    
    img_id = abs(hash(filename))% 1000000
    images.append(
        {
                "file_name": filename,
                "height": img.shape[0],
                "width": img.shape[1],
                "id": img_id
        }
    )
    for (ct, c) in [(x, x[:, 0, :]) for x in valid_contours]:
         annotations.append(           
            {
                    "segmentation": [c.reshape(-1).tolist()],
                    "area": cv.contourArea(ct),
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": [int(x) for x in [np.min(c[:, 0]), np.min(c[:, 1]), np.max(c[:, 0]) - np.min(c[:, 0]), np.max(c[:, 1]) - np.min(c[:, 1])]],
                    "category_id": 1,
                    "id": len(annotations),
            }
         )


for fn in tqdm(os.listdir(IMG_PATH)):
    process(fn)
    
coco = {
    "info": {
        "year": 2023,
        "version": "0.1",
        "description": "playing card (Bicycle Rider Black) on uniform background with mask",
        "contributor": "Pipat Saengow",
        "url": "",
        "date_created": "2023/05/20",
    },
    "licenses": [],
    "images": images,
    "categories": [
        {"supercategory": "cards","id": 1,"name": "card"},
    ],
    "annotations": annotations
}


with open("coco.json", "w") as f:
    json.dump(coco, f)
