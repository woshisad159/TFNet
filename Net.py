import torch.nn as nn
import torch
import Transformer
import torchvision.models as models
import numpy as np
import Module
from BiLSTM import BiLSTMLayer
import SEN

class moduleNet(nn.Module):
    def __init__(self, hiddenSize, wordSetNum, moduleChoice="Seq2Seq", device=torch.device("cuda:0"), dataSetName='RWTH', isFlag=False):
        super().__init__()
        self.device = device
        self.moduleChoice = moduleChoice
        self.outDim = wordSetNum
        self.dataSetName = dataSetName
        self.logSoftMax = nn.LogSoftmax(dim=-1)
        self.softMax = nn.Softmax(dim=-1)
        self.isFlag = isFlag
        self.probs_log = []

        if "MSTNet" == moduleChoice:
            self.conv2d = getattr(models, "resnet34")(pretrained=True)
            self.conv2d.fc = Module.Identity()

            hidden_size = hiddenSize
            inputSize = hiddenSize

            self.conv1D1_1 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=3, stride=1,
                                       padding=1)
            self.conv1D1_2 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=5, stride=1,
                                       padding=2)
            self.conv1D1_3 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=7, stride=1,
                                       padding=3)
            self.conv1D1_4 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=9, stride=1,
                                       padding=4)

            self.conv2D1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 2), stride=2,
                                     padding=0)

            self.conv1D2_1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1,
                                       padding=1)
            self.conv1D2_2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, stride=1,
                                       padding=2)
            self.conv1D2_3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=7, stride=1,
                                       padding=3)
            self.conv1D2_4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9, stride=1,
                                       padding=4)

            self.conv2D2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 2), stride=2,
                                     padding=0)

            self.batchNorm1d1_1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_2 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_3 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_4 = nn.BatchNorm1d(hidden_size)

            self.batchNorm1d2_1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_2 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_3 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_4 = nn.BatchNorm1d(hidden_size)

            self.batchNorm2d1 = nn.BatchNorm2d(hidden_size)
            self.batchNorm2d2 = nn.BatchNorm2d(hidden_size)

            self.relu = nn.ReLU(inplace=True)

            heads = 8
            semantic_layers = 2
            dropout = 0
            rpe_k = 8
            self.temporal_model = Transformer.TransformerEncoder(hidden_size, heads, semantic_layers, dropout, rpe_k)

            self.linear1 = nn.Linear(512, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)

            self.batchNorm1d1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2 = nn.BatchNorm1d(hidden_size)

            self.classifier1 = Module.NormLinear(hidden_size, self.outDim)
            self.classifier2 = Module.NormLinear(hidden_size, self.outDim)

            if self.dataSetName == 'RWTH' or self.dataSetName == 'CE-CSL':
                self.classifier3 = Module.NormLinear(hidden_size, self.outDim)
                self.classifier4 = Module.NormLinear(inputSize, self.outDim)
        elif "VAC" == self.moduleChoice:
            hidden_size = hiddenSize
            self.conv2d = getattr(models, "resnet18")(pretrained=True)
            self.conv2d.fc = Module.Identity()

            self.conv1d = Module.TemporalConv(input_size=512,
                                           hidden_size=hidden_size,
                                           conv_type=2)

            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier = Module.NormLinear(hidden_size, self.outDim)
            self.classifier1 = self.classifier
        elif "CorrNet" == self.moduleChoice:
            hidden_size = hiddenSize
            self.conv2d = Module.resnet18Corr()
            self.conv2d.fc = Module.Identity()

            self.conv1d = Module.TemporalConv(input_size=512,
                                           hidden_size=hidden_size,
                                           conv_type=2)

            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier = Module.NormLinear(hidden_size, self.outDim)
            self.classifier1 = self.classifier
        elif "MAM-FSD" == self.moduleChoice:
            hidden_size = hiddenSize
            self.conv2d = Module.resnet34MAM()
            self.conv2d.fc = Module.Identity()

            self.conv1d = Module.TemporalConv(input_size=512,
                                           hidden_size=hidden_size,
                                           conv_type=2)

            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier = Module.NormLinear(hidden_size, self.outDim)
            self.classifier1 = self.classifier

            self.conv1 = nn.Conv3d(64, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0),
                                   bias=False)
            self.conv2 = nn.Conv3d(128, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0),
                                   bias=False)
            self.conv3 = nn.Conv3d(256, 512, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0),
                                   bias=False)

            self.batchNorm3d1 = nn.BatchNorm3d(128)
            self.batchNorm3d2 = nn.BatchNorm3d(256)
            self.batchNorm3d3 = nn.BatchNorm3d(512)

            self.reLU = nn.ReLU(inplace=True)
        elif "SEN" == self.moduleChoice:
            hidden_size = hiddenSize
            self.conv2d = SEN.resnet18()
            self.conv2d.fc = Module.Identity()

            self.conv1d = Module.TemporalConv(input_size=512,
                                           hidden_size=hidden_size,
                                           conv_type=2)

            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier = nn.Linear(hidden_size, self.outDim)
            self.classifier1 = nn.Linear(hidden_size, self.outDim)
        elif "TFNet" == self.moduleChoice:
            hidden_size = hiddenSize
            self.conv2d = Module.resnet34MAM()
            self.conv2d.fc = Module.Identity()

            self.conv1d = Module.TemporalConv(input_size=512,
                                           hidden_size=hidden_size,
                                           conv_type=2)

            self.conv1d1 = Module.TemporalConv(input_size=512,
                                                 hidden_size=hidden_size,
                                                 conv_type=2)

            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.temporal_model1 = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier1 = Module.NormLinear(hidden_size, self.outDim)
            self.classifier2 = self.classifier1

            self.classifier3 = Module.NormLinear(hidden_size, self.outDim)
            self.classifier4 = self.classifier3

            self.classifier5 = Module.NormLinear(hidden_size, self.outDim)

            self.reLU = nn.ReLU(inplace=True)

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    def forward(self, seqData, dataLen=None, isTrain=True):
        outData1 = None
        outData2 = None
        outData3 = None
        logProbs1 = None
        logProbs2 = None
        logProbs3 = None
        logProbs4 = None
        logProbs5 = None

        if "MSTNet" == self.moduleChoice:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            inputs = seqData.reshape(batch * temp, channel, height, width)

            x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])

            n = len(x)
            indices = np.arange(n)
            np.random.shuffle(indices)
            trainIndex = indices[: int(n * 0.5)]
            trainIndex = sorted(trainIndex)
            testIndex = indices[int(n * 0.5):]
            testIndex = sorted(testIndex)

            trainData = x[trainIndex, :, :, :]
            testData = x[testIndex, :, :, :]

            trainData = self.conv2d(trainData)

            with torch.no_grad():
                testData = self.conv2d(testData)

            shape = trainData.shape
            x1 = torch.zeros(((shape[0] // 1) * 2, shape[1])).cuda()

            for i in range(len(trainIndex)):
                x1[trainIndex[i], :] = trainData[i, :]

            for i in range(len(testIndex)):
                x1[testIndex[i], :] = testData[i, :]

            framewise = torch.cat([self.pad(x1[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                                   for idx, lgt in enumerate(len_x)])

            framewise = framewise.reshape(batch, temp, -1)

            framewise = self.linear1(framewise).transpose(1, 2)
            framewise = self.batchNorm1d1(framewise)
            framewise = self.relu(framewise).transpose(1, 2)
            #
            framewise = self.linear2(framewise).transpose(1, 2)
            framewise = self.batchNorm1d2(framewise)
            framewise = self.relu(framewise)

            inputData = self.conv1D1_1(framewise)
            inputData = self.batchNorm1d1_1(inputData)
            inputData = self.relu(inputData)

            glossCandidate = inputData.unsqueeze(2)

            inputData = self.conv1D1_2(framewise)
            inputData = self.batchNorm1d1_2(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv1D1_3(framewise)
            inputData = self.batchNorm1d1_3(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv1D1_4(framewise)
            inputData = self.batchNorm1d1_4(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv2D1(glossCandidate)
            inputData = self.batchNorm2d1(inputData)
            inputData1 = self.relu(inputData).squeeze(2)

            # 2
            inputData = self.conv1D2_1(inputData1)
            inputData = self.batchNorm1d2_1(inputData)
            inputData = self.relu(inputData)

            glossCandidate = inputData.unsqueeze(2)

            inputData = self.conv1D2_2(inputData1)
            inputData = self.batchNorm1d2_2(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv1D2_3(inputData1)
            inputData = self.batchNorm1d2_3(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv1D2_4(inputData1)
            inputData = self.batchNorm1d2_4(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv2D2(glossCandidate)
            inputData = self.batchNorm2d2(inputData)
            inputData = self.relu(inputData).squeeze(2)

            if not self.dataSetName == 'CSL':
                lgt = torch.cat(len_x, dim=0) // 4
                x = inputData.permute(0, 2, 1)
            else:
                lgt = (torch.cat(len_x, dim=0) // 4) - 6
                x = inputData.permute(0, 2, 1)
                x = x[:, 3:-3, :]

            outputs = self.temporal_model(x)

            outputs = outputs.permute(1, 0, 2)
            logProbs1 = self.classifier1(outputs)

            outputs = x.permute(1, 0, 2)
            logProbs2 = self.classifier2(outputs)

            if not self.dataSetName == 'CSL':
                outputs = inputData1.permute(2, 0, 1)
                logProbs3 = self.classifier3(outputs)

                outputs = framewise.permute(2, 0, 1)
                logProbs4 = self.classifier4(outputs)

            logProbs5 = logProbs1
        elif "VAC" == self.moduleChoice:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            inputs = seqData.reshape(batch * temp, channel, height, width)

            x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])

            x = self.conv2d(x)

            framewise = torch.cat([self.pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                                   for idx, lgt in enumerate(len_x)])

            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

            conv1d_outputs = self.conv1d(framewise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            lgt = torch.cat(lgt, dim=0)

            outputs = self.temporal_model(x, lgt)

            encoderPrediction = self.classifier(outputs['predictions'])
            logProbs1 = encoderPrediction

            encoderPrediction = self.classifier1(x)
            logProbs2 = encoderPrediction
        elif "CorrNet" == self.moduleChoice:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            x = seqData.transpose(1, 2)

            framewise = self.conv2d(x)

            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

            conv1d_outputs = self.conv1d(framewise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            lgt = torch.cat(lgt, dim=0)

            outputs = self.temporal_model(x, lgt)

            encoderPrediction = self.classifier(outputs['predictions'])
            logProbs1 = encoderPrediction

            encoderPrediction = self.classifier1(x)
            logProbs2 = encoderPrediction
        elif "MAM-FSD" == self.moduleChoice:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape

            x = seqData.transpose(1, 2)

            framewise, outData1, outData2, outData3 = self.conv2d(x)

            tmpOut = self.conv1(outData1[0])
            tmpOut = self.batchNorm3d1(tmpOut)
            outData1[0] = self.reLU(tmpOut)

            tmpOut = self.conv2(outData2[0])
            tmpOut = self.batchNorm3d2(tmpOut)
            outData2[0] = self.reLU(tmpOut)

            tmpOut = self.conv3(outData3[0])
            tmpOut = self.batchNorm3d3(tmpOut)
            outData3[0] = self.reLU(tmpOut)

            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

            conv1d_outputs = self.conv1d(framewise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            lgt = torch.cat(lgt, dim=0)

            outputs = self.temporal_model(x, lgt)

            encoderPrediction = self.classifier(outputs['predictions'])
            logProbs1 = encoderPrediction

            encoderPrediction = self.classifier1(x)
            logProbs2 = encoderPrediction

            logProbs3 = None
            logProbs4 = None
        elif "SEN" == self.moduleChoice:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            x = seqData.transpose(1, 2)

            framewise = self.conv2d(x)

            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

            conv1d_outputs = self.conv1d(framewise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            lgt = torch.cat(lgt, dim=0)

            outputs = self.temporal_model(x, lgt)

            encoderPrediction = self.classifier(outputs['predictions'])
            logProbs1 = encoderPrediction

            encoderPrediction = self.classifier1(x)
            logProbs2 = encoderPrediction
        elif "TFNet" == self.moduleChoice:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            x = seqData.transpose(1, 2)

            framewise, outData1, outData2, outData3 = self.conv2d(x)

            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

            # 傅里叶变换
            framewise1 = framewise.transpose(1, 2).float()
            X = torch.fft.fft(framewise1, dim=-1, norm="forward")
            X = torch.abs(X)
            framewise1 = X.transpose(1, 2)

            conv1d_outputs = self.conv1d(framewise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            lgt = torch.cat(lgt, dim=0)

            conv1d_outputs1 = self.conv1d1(framewise1, len_x)
            # x: T, B, C
            x1 = conv1d_outputs1['visual_feat']
            x1 = x1.permute(2, 0, 1)

            outputs = self.temporal_model(x, lgt)
            outputs1 = self.temporal_model1(x1, lgt)

            encoderPrediction = self.classifier1(outputs['predictions'])
            logProbs1 = encoderPrediction

            encoderPrediction = self.classifier2(x)
            logProbs2 = encoderPrediction

            encoderPrediction = self.classifier3(outputs1['predictions'])
            logProbs3 = encoderPrediction

            encoderPrediction = self.classifier4(x1)
            logProbs4 = encoderPrediction

            x2 = outputs['predictions'] + outputs1['predictions']
            logProbs5 = self.classifier5(x2)

            if not isTrain:
                logProbs1 = logProbs5

        return logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, outData1, outData2, outData3