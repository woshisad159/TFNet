import os
from evaluationT.python_wer_evaluationT import wer_calculation
from evaluationT.python_wer_evaluationT import wer_calculation1
# from python_wer_evaluationT import wer_calculation
# from python_wer_evaluationT import wer_calculation1
import sys

def evaluate3(mode="dev", evaluate_prefix=None,
             output_file=None, isPrint=True):
    '''
    TODO  change file save path
    '''
    os.system(f"bash evaluationT/preprocess.sh evaluationT/{output_file} evaluationT/tmp.ctm evaluationT/tmp2.ctm")
    os.system(f"bash evaluationT/preprocess1.sh evaluationT/{evaluate_prefix}-{mode}.stm evaluationT/tmp.stm")
    # os.system(f"cat evaluationT/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > evaluationT/tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python evaluationT/mergectmstm1.py evaluationT/tmp2.ctm evaluationT/tmp.stm")
    os.system(f"cp evaluationT/tmp2.ctm evaluationT/out.{output_file}")

    ret = wer_calculation1(f"evaluationT/{evaluate_prefix}-{mode}.stm", f"evaluationT/out.{output_file}", isPrint)

    os.system(f"rm -rf evaluationT/out.{output_file}")
    return ret

def evaluate2(mode="dev", evaluate_prefix=None,
             output_file=None, isPrint=True):
    '''
    TODO  change file save path
    '''
    os.system(f"bash evaluationT/preprocess.sh evaluationT/{output_file} evaluationT/tmp.ctm evaluationT/tmp2.ctm")
    os.system(f"bash evaluationT/preprocess1.sh evaluationT/{evaluate_prefix}-{mode}.stm evaluationT/tmp.stm")
    # os.system(f"cat evaluationT/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > evaluationT/tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python evaluationT/mergectmstm.py evaluationT/tmp2.ctm evaluationT/tmp.stm")
    os.system(f"cp evaluationT/tmp2.ctm evaluationT/out.{output_file}")

    ret = wer_calculation(f"evaluationT/{evaluate_prefix}-{mode}.stm", f"evaluationT/out.{output_file}", isPrint)

    os.system(f"rm -rf evaluationT/out.{output_file}")
    return ret

def evaluate1(mode="dev", evaluate_prefix=None,
             output_file=None):
    '''
    TODO  change file save path
    '''
    os.system(f"bash preprocess.sh {output_file} tmp.ctm tmp2.ctm")
    os.system(f"bash preprocess1.sh {evaluate_prefix}-{mode}.stm tmp.stm")
    # os.system(f"cat {evaluate_prefix}-{mode}.stm | sort  -k1,1 > tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python mergectmstm.py tmp2.ctm tmp.stm")
    os.system(f"cp tmp2.ctm out.{output_file}")

    ret = wer_calculation(f"{evaluate_prefix}-{mode}.stm", f"out.{output_file}", mode)

    os.system(f"rm -rf out.{output_file}")
    return ret

def evaluteModeT(mode="dev", isPrint=True):
    if mode == 'dev':
        filePath = "./wer/dev/"
        path = 'wer.txt'

        fileReader = open(path, "w")

        fileList = os.listdir(filePath)
        fileList.sort(key=lambda x: int(x[21:25]))

        werResultList = []
        fileNameList = []
        for i, fileName in enumerate(fileList):
            path = os.path.join(filePath, fileName)
            os.system(f"cp {path} {fileName}")

            ret = evaluate1(
                        mode=mode, output_file=fileName,
                        evaluate_prefix='PHOENIX-2014-T-groundtruth',
                )

            werResultList.append(ret)
            fileNameList.append(fileName)

            os.system(f"rm -rf {fileName}")

            fileReader.writelines(
                "{} {} {:.2f}\n".format(i,fileName,ret))

        indexValue = werResultList.index(min(werResultList))

        fileReader.writelines(
            "Min WER:index:{}, fileName:{}, WER:{:.2f}\n".format(indexValue, fileNameList[indexValue], werResultList[indexValue]))

        print("Min WER:index:{}, fileName:{}, WER:{:.2f}".format(indexValue, fileNameList[indexValue], werResultList[indexValue]))
    elif mode == 'test':
        filePath = "./wer/test/"
        path = 'wer.txt'

        fileReader = open(path, "w")

        fileList = os.listdir(filePath)
        fileList.sort(key=lambda x: int(x[22:26]))

        werResultList = []
        fileNameList = []
        for i, fileName in enumerate(fileList):
            path = os.path.join(filePath, fileName)
            os.system(f"cp {path} {fileName}")

            ret = evaluate1(
                mode=mode, output_file=fileName,
                evaluate_prefix='PHOENIX-2014-T-groundtruth',
            )

            werResultList.append(ret)
            fileNameList.append(fileName)

            os.system(f"rm -rf {fileName}")

            fileReader.writelines(
                "{} {} {:.2f}\n".format(i, fileName, ret))

        indexValue = werResultList.index(min(werResultList))

        fileReader.writelines(
            "Min WER:index:{}, fileName:{}, WER:{:.2f}\n".format(indexValue, fileNameList[indexValue],
                                                                 werResultList[indexValue]))

        print("Min WER:index:{}, fileName:{}, WER:{:.2f}".format(indexValue, fileNameList[indexValue],
                                                                 werResultList[indexValue]))
    elif mode == 'evalute_dev':
        path = "evaluationT/wer/evalute/output-hypothesis-dev.ctm"
        fileName = "output-hypothesis-dev.ctm"
        os.system(f"cp {path} evaluationT/{fileName}")

        mode = 'dev'
        ret = evaluate3(
            mode=mode, output_file=fileName,
            evaluate_prefix='PHOENIX-2014-T-groundtruth',
            isPrint=isPrint,
        )

        os.system(f"rm -rf evaluationT/{fileName}")
    elif mode == 'evalute_dev1':
        path = "evaluationT/wer/evalute/output-hypothesis-dev.ctm"
        fileName = "output-hypothesis-dev.ctm"
        os.system(f"cp {path} evaluationT/{fileName}")

        mode = 'dev'
        ret = evaluate2(
            mode=mode, output_file=fileName,
            evaluate_prefix='PHOENIX-2014-T-groundtruth',
            isPrint=isPrint,
        )

        os.system(f"rm -rf evaluationT/{fileName}")
    elif mode == 'evalute_test':
        path = "evaluationT/wer/evalute/output-hypothesis-test.ctm"
        fileName = "output-hypothesis-test.ctm"
        os.system(f"cp {path} evaluationT/{fileName}")

        mode = 'test'
        ret = evaluate2(
            mode=mode, output_file=fileName,
            evaluate_prefix='PHOENIX-2014-T-groundtruth',
            isPrint=isPrint,
        )

        os.system(f"rm -rf evaluationT/{fileName}")

    return ret

if __name__ == '__main__':
    try:
        inputArgv = sys.argv[1]

        mode = inputArgv
    except:
        mode = 'test'

    evaluteModeT(mode)
