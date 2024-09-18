from Train import train, seed_torch
from ReadConfig import readConfig

def main():
    # 读取配置文件
    configParams = readConfig()
    # isTrain为True是训练模式，isTrain为False是验证模式
    train(configParams, isTrain=True, isCalc=False)


if __name__ == '__main__':
    seed_torch(10)
    main()

