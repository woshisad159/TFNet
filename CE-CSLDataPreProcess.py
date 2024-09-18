import random
import os
import numpy as np
import shutil
from tqdm import tqdm
import imageio
import cv2
import csv
import DataProcessMoudle

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def main(dataPath, saveDataPath):
    fileTypes = sorted(os.listdir(dataPath))

    framesList = []
    fpsList = []
    videoTimeList = []
    resolutionList = []
    for fileType in fileTypes:
        filePath = os.path.join(dataPath, fileType)
        saveFilePath = os.path.join(saveDataPath, fileType)
        translators = sorted(os.listdir(filePath))

        for translator in translators:
            translatorPath = os.path.join(filePath, translator)
            saveTranslatorPath = os.path.join(saveFilePath, translator)
            videos = sorted(os.listdir(translatorPath))

            for video in tqdm(videos):
                videoPath = os.path.join(translatorPath, video)
                nameString = video.split(".")
                saveImagePath = os.path.join(saveTranslatorPath, nameString[0])

                if not os.path.exists(saveImagePath):
                    os.makedirs(saveImagePath)

                vid = imageio.get_reader(videoPath)  # 读取视频
                # nframes = vid.get_meta_data()['nframes']
                nframes = vid.count_frames()
                fps = vid.get_meta_data()['fps']
                videoTime = vid.get_meta_data()['duration']
                resolution = vid.get_meta_data()['size']

                framesList.append(nframes)
                fpsList.append(fps)
                videoTimeList.append(videoTime)
                resolutionList.append(resolution)

                for i in range(nframes):
                    try:
                        image = vid.get_data(i)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (256, 256))

                        nameString = str(i)
                        for i in range(5 - len(nameString)):
                            nameString = "0" + nameString

                        imagePath = os.path.join(saveImagePath, nameString + ".jpg")
                        cv2.imencode('.jpg', image)[1].tofile(imagePath)
                    except:
                        print(nframes)
                        print(videoPath)

                vid.close()

    maxframeNum = max(framesList)
    minframeNum = min(framesList)
    maxVideoTime = max(videoTimeList)
    minVideoTime = min(videoTimeList)
    fpsSet = set(fpsList)
    resolutionSet = set(resolutionList)

    print(f"Max Frames Number:{maxframeNum}\n"
          f"Min Frames Number:{minframeNum}\n"
          f"Max Video Time:{maxVideoTime}\n"
          f"Min Video Time:{minVideoTime}\n"
          f"Fps Set:{fpsSet}\n"
          f"Resolution Set:{resolutionSet}\n")


if __name__ == '__main__':
    dataPath = "/home/lj/lj/program/python/DataSets/CE-CSL/video"
    saveDataPath = "/home/lj/lj/program/python/DataSets/CE-CSL/video2"

    seed_torch()
    main(dataPath, saveDataPath)