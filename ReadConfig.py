import configparser
import os
import torch

def readConfig():
    # 默认配置参数
    configParams = {
        "trainDataPath":"data\RWTH\\train",
        "validDataPath": "data\RWTH\\valid",
        "testDataPath": "data\RWTH\\test",
        "trainLabelPath": "data\RWTH\\train.corpus.csv",
        "validLabelPath": "data\RWTH\\dev.corpus.csv",
        "testLabelPath": "data\RWTH\\test.corpus.csv",
        "bestModuleSavePath": "module/bestMoudleNet.pth",
        "currentModuleSavePath": "module/currentMoudleNet.pth",
        "device": 1, # 0:CPU  1:GPU
        "hiddenSize":512,
        "lr": 0.1,
        "batchSize": 1,
        "numWorkers": 2,
        "pinmMemory": 1,
        "dataSetName": "RWTH",
    }

    configPath = "params/config.ini"
    if os.path.exists(configPath):
        print("开始读取配置参数")
        cf = configparser.ConfigParser()
        cf.read(configPath)  # 读取配置文件，如果写文件的绝对路径，就可以不用os模块

        # 读取路径参数
        configParams["trainDataPath"] = cf.get("Path", "trainDataPath")
        configParams["validDataPath"] = cf.get("Path", "validDataPath")
        configParams["testDataPath"] = cf.get("Path", "testDataPath")
        configParams["trainLabelPath"] = cf.get("Path", "trainLabelPath")
        configParams["validLabelPath"] = cf.get("Path", "validLabelPath")
        configParams["testLabelPath"] = cf.get("Path", "testLabelPath")
        configParams["bestModuleSavePath"] = cf.get("Path", "bestModuleSavePath")
        configParams["currentModuleSavePath"] = cf.get("Path", "currentModuleSavePath")
        # 读取数值参数
        configParams["device"] = cf.get("Params", "device")
        configParams["hiddenSize"] = cf.get("Params", "hiddenSize")
        configParams["lr"] = cf.get("Params", "lr")
        configParams["batchSize"] = cf.get("Params", "batchSize")
        configParams["numWorkers"] = cf.get("Params", "numWorkers")
        configParams["pinmMemory"] = cf.get("Params", "pinmMemory")
        configParams["moduleChoice"] = cf.get("Params", "moduleChoice")
        configParams["dataSetName"] = cf.get("Params", "dataSetName")

        print("GPU is %s" % torch.cuda.is_available())
        if 1 == int(configParams["device"]):
            configParams["device"] = torch.device("cuda:0")
        else:
            configParams["device"] = torch.device("cpu")
    else:
        print("配置文件不存在 %s" % (configPath))
        print("使用默认参数")

    for key in configParams:
        print("%s: %s" %(key, configParams[key]))

    return configParams
