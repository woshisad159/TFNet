import csv
import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pickle
import cv2

PAD = ' '

def PreWords(words):
    for i in range(len(words)):
        word = words[i]

        n = 0
        subFlag = False
        wordList = list(word)
        for j in range(len(word)):
            if word[j] == "(" or word[j] == "{" or word[j] == "[" or word[j] == "（":
                subFlag = True

            if subFlag:
                wordList.pop(j - n)
                n = n + 1

            if word[j] == ")" or word[j] == "}" or word[j] == "]" or word[j] == "）":
                subFlag = False

        word = "".join(wordList)

        if word[-1].isdigit():
            if not word[0].isdigit():
                wordList = list(word)
                wordList.pop(len(word) - 1)
                word = "".join(wordList)

        if word[0] == "," or word[0] == "，":
            wordList = list(word)
            wordList[0] = '，'
            word = ''.join(wordList)

        if word[0] == "?" or word[0] == "？":
            wordList = list(word)
            wordList[0] = '？'
            word = ''.join(wordList)

        if word.isdigit():
            word = str(int(word))

        words[i] = word

    return words

def Word2Id(trainLabelPath, validLabelPath, testLabelPath, dataSetName):
    if dataSetName == "RWTH":
        wordList = []
        with open(trainLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[3].split()
                    wordList += words

        with open(validLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[3].split()
                    wordList += words

        with open(testLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[3].split()
                    wordList += words
    elif dataSetName == "RWTH-T":
        wordList = []
        with open(trainLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[5].split()
                    wordList += words

        with open(validLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[5].split()
                    wordList += words

        with open(testLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[5].split()
                    wordList += words
    elif dataSetName == "CSL":
        with open(trainLabelPath, "r", encoding="utf-8") as f:
            sourceStr = f.read()

        txtStr = sourceStr.split()[1::2]

        for i, s in enumerate(txtStr):
            txtStr[i] = s.strip("\ufeff")

        wordList = ''.join((x for x in txtStr))
    elif dataSetName == "CSL-Daily":
        with open("/home/lj/lj/program/python/SLR20240523/data/CSL-Daily/csl2020ct_v2.pkl", 'rb') as f:
            data = pickle.load(f)
        wordList = data["gloss_map"]
    elif dataSetName == "CE-CSL":
        wordList = []
        with open(trainLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    words = row[3].split("/")
                    words = PreWords(words)
                    wordList += words

        with open(validLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    words = row[3].split("/")
                    words = PreWords(words)
                    wordList += words

        with open(testLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    words = row[3].split("/")
                    words = PreWords(words)
                    wordList += words

    idx2word = [PAD]
    set2list = sorted(list(set(wordList)))
    idx2word.extend(set2list)

    word2idx = {w: i for i, w in enumerate(idx2word)}

    return word2idx, len(idx2word) - 1, idx2word


class MyDataset(Dataset):
    def __init__(self, ImagePath, LabelPath, word2idx, dataSetName, isTrain=False, transform=None):
        """
        path : 数据路径，包含了图像的路径
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        self.ImagePath = ImagePath
        self.transform = transform
        self.dataSetName = dataSetName
        self.p_drop = 0.5
        self.random_drop = True
        self.isTrain = isTrain

        if dataSetName == "RWTH":
            lableDict = {}
            with open(LabelPath, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    if n != 0:
                        rowStrList = row[0].split("|")
                        lableDict[rowStrList[0]] = rowStrList[-1]

            lable = {}
            for line in lableDict:
                sentences = lableDict[line].split()

                txtInt = []
                for i in sentences:
                    txtInt.append(word2idx[i])

                lable[line] = txtInt

            fileName = sorted(os.listdir(ImagePath))

            imgs = []
            for name in fileName:
                try:
                    imageSeqPath = os.path.join(ImagePath, name)
                    imgs.append((imageSeqPath, lable[name]))  # 路径和标签添加到列表中
                except:
                    print(name)
        elif dataSetName == "RWTH-T":
            lableDict = {}
            with open(LabelPath, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    if n != 0:
                        rowStrList = row[0].split("|")
                        lableDict[rowStrList[0]] = rowStrList[-2]

            lable = {}
            for line in lableDict:
                sentences = lableDict[line].split()

                txtInt = []
                for i in sentences:
                    txtInt.append(word2idx[i])

                lable[line] = txtInt

            fileName = sorted(os.listdir(ImagePath))

            imgs = []
            for name in fileName:
                try:
                    imageSeqPath = os.path.join(ImagePath, name)
                    imgs.append((imageSeqPath, lable[name]))  # 路径和标签添加到列表中
                except:
                    print(name)
        elif dataSetName == "CSL":
            lableDict = {}
            with open(LabelPath, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    rowStrList = row[0].split()
                    lableDict[rowStrList[0]] = rowStrList[-1]

            lable = {}
            for line in lableDict:
                sentences = lableDict[line]
                sentences = sentences.strip("\ufeff")

                txtInt = []
                for i in sentences:
                    try:
                        txtInt.append(word2idx[i])
                    except:
                        print(sentences)

                lable[line] = txtInt

            fileName = sorted(os.listdir(ImagePath))

            imgs = []
            for name in fileName:
                imageSeqPath = os.path.join(ImagePath, name)

                ImageSeq = sorted(os.listdir(imageSeqPath))

                for i in ImageSeq:
                    frames = os.path.join(imageSeqPath, i)
                    imgs.append((frames, lable[name]))  # 路径和标签添加到列表中
        elif dataSetName == "CSL-Daily":
            with open("/home/lj/lj/program/python/SLR20240523/data/CSL-Daily/csl2020ct_v2.pkl", 'rb') as f:
                data = pickle.load(f)

            info = data["info"]

            lable = {}
            for n, value in enumerate(info):
                txtInt = []
                for i, gloss in enumerate(value["label_gloss"]):
                    txtInt.append(word2idx[gloss])

                lable[value["name"]] = txtInt

            imgs = []
            with open(LabelPath, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    rowStrList = row[0].split("|")
                    frames = os.path.join(ImagePath, rowStrList[0])
                    try:
                        imgs.append((frames, lable[rowStrList[0]]))
                    except:
                        print(rowStrList[0])
        elif dataSetName == "CE-CSL":
            lableDict = {}
            with open(LabelPath, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    if n != 0:
                        lableDict[row[0]] = row[3]

            lable = {}
            for line in lableDict:
                sentences = lableDict[line].split("/")
                sentences = PreWords(sentences)

                txtInt = []
                for i in sentences:
                    txtInt.append(word2idx[i])

                lable[line] = txtInt

            fileNames = sorted(os.listdir(ImagePath))

            imgs = []
            for name in fileNames:
                fileName = os.path.join(ImagePath, name)
                imageNames = sorted(os.listdir(fileName))
                for imageName in imageNames:
                    try:
                        imageSeqPath = os.path.join(ImagePath, name, imageName)
                        imgs.append((imageSeqPath, lable[imageName]))  # 路径和标签添加到列表中
                    except:
                        print(imageName)

        self.imgs = imgs

    def sample_indices(self, n):
        indices = np.linspace(0, n - 1, num=int(n // 1), dtype=int)
        return indices

    def __getitem__(self, index):
        fn, label = self.imgs[index]# 通过index索引返回一个图像路径fn 与 标签label
        if self.dataSetName == "RWTH":
            info = fn.split("/")[-1]

            imageSeqPath = os.path.join(fn, "1")

            ImageSeq = sorted(os.listdir(imageSeqPath))

            indices = self.sample_indices(len(ImageSeq))

            frames = [os.path.join(imageSeqPath, i) for i in ImageSeq]
            frames = [frames[i] for i in indices]

            imgSeq = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (256, 256)) for img_path in frames]

            imgSeq = self.transform(imgSeq)
            imgSeq = imgSeq.float() / 127.5 - 1
        elif self.dataSetName == "RWTH-T":
            info = fn.split("/")[-1]

            imageSeqPath = fn

            ImageSeq = sorted(os.listdir(imageSeqPath))

            indices = self.sample_indices(len(ImageSeq))

            frames = [os.path.join(imageSeqPath, i) for i in ImageSeq]
            frames = [frames[i] for i in indices]

            imgSeq = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (256, 256)) for img_path in frames]

            imgSeq = self.transform(imgSeq)
            imgSeq = imgSeq.float() / 127.5 - 1
        elif self.dataSetName == "CSL":
            # 图片序列
            info = fn.split("/")[-1]

            ImagePath = fn

            ImageSeq = sorted(os.listdir(fn))
            # 读取多序列图片
            for i, image in enumerate(ImageSeq):
                imageSeqPath = os.path.join(ImagePath, image)
                ImgSeq = cv2.cvtColor(cv2.imread(imageSeqPath), cv2.COLOR_BGR2RGB)
                seqShape = ImgSeq.shape
                frames1 = ImgSeq.reshape((seqShape[0] // seqShape[1], seqShape[1], seqShape[1], seqShape[2]))
                if i != 0:
                    frames = np.concatenate((frames, frames1), axis=0)
                else:
                    frames = frames1

            imageSize = 256

            imgSeq = [cv2.resize(img, (imageSize, imageSize)) for img in frames]
            indices = self.sample_indices(len(imgSeq) // 2)
            imgSeq = [imgSeq[i*2] for i in indices]

            imgSeq = self.transform(imgSeq)
            imgSeq = imgSeq.float() / 127.5 - 1
        elif self.dataSetName == "CSL-Daily":
            info = fn
            imageSeqPath = fn

            ImageSeq = sorted(os.listdir(imageSeqPath))

            indices = self.sample_indices(len(ImageSeq))

            frames = [os.path.join(imageSeqPath, i) for i in ImageSeq]
            frames = [frames[i] for i in indices]

            imgSeq = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (256, 256)) for img_path in
                      frames]

            imgSeq = self.transform(imgSeq)
            imgSeq = imgSeq.float() / 127.5 - 1
        elif self.dataSetName == "CE-CSL":
            info = fn.split("/")[-1]

            imageSeqPath = fn

            ImageSeq = sorted(os.listdir(imageSeqPath))

            indices = self.sample_indices(len(ImageSeq))

            frames = [os.path.join(imageSeqPath, i) for i in ImageSeq]
            frames = [frames[i] for i in indices]

            imgSeq = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (256, 256)) for img_path in frames]

            imgSeq = self.transform(imgSeq)
            imgSeq = imgSeq.float() / 127.5 - 1

        sample = {"video": imgSeq, "label": label, "info": info}

        return sample  # 这就返回一个样本

    def __len__(self):
        return len(self.imgs)  # 返回长度，index就会自动的指导读取多少

class defaultdict_with_warning(defaultdict):
    warned = set()
    warning_enabled = False

    def __getitem__(self, key):
        if key == "text" and key not in self.warned and self.warning_enabled:
            print(
                'Warning: using batch["text"] to obtain label is deprecated, '
                'please use batch["label"] instead.'
            )
            self.warned.add(key)

        return super().__getitem__(key)

def collate_fn(batch):
    collated = defaultdict_with_warning(list)

    batch = [item for item in sorted(batch, key=lambda x: len(x["video"]), reverse=True)]
    max_len = len(batch[0]["video"])

    # MAM-FSD、CorrNet、VAC、TFNet
    left_pad = 6# 6
    total_stride = 4# 4
    right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    max_len = max_len + left_pad + right_pad

    # MSTNet
    # left_pad = 0  # 6
    # total_stride = 4  # 4
    # right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    # max_len = max_len + left_pad + right_pad

    padded_video = []
    for sample in batch:
        vid = sample["video"]
        collated["videoLength"].append(torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad]))
        padded_video.append(torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
            )
            , dim=0))

        collated["label"].append(torch.tensor(sample["label"]).long())
        collated["info"].append(sample["info"])
        collated["expand"].append([left_pad, max_len - len(vid) - left_pad])

    padded_video = torch.stack(padded_video)
    collated["video"] = padded_video
    collated.warning_enabled = True

    return dict(collated)

def DataReshape(seqData, device):
    xl = list(map(len, seqData))
    batchSize = len(xl)
    seqData = torch.cat(seqData, dim=0).to(device)

    return seqData, batchSize, xl

def RemoveBlank(labels, maxSentenceLen, blank=0):
    new_labels = []
    # 合并相同的标签
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]

    if len(new_labels) < maxSentenceLen:
        for _ in range(maxSentenceLen - len(new_labels)):
            new_labels.append(0)
        new_labelsTmp = new_labels
    else:
        new_labelsTmp = new_labels[:maxSentenceLen]

    outPut = torch.Tensor(new_labelsTmp)

    return outPut

def CTCGreedyDecode(y, maxSentenceLen, blank=0):
    # 按列取最大值，即每个时刻t上最大值对应的下标
    raw_rs = y.argmax(dim=-1)
    # 移除blank,值为0的位置表示这个位置是blank
    rs = RemoveBlank(raw_rs, maxSentenceLen, blank)
    return rs

def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))

class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        return loss