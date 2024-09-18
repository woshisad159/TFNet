import numpy as np

WER_COST_DEL = 1
WER_COST_INS = 1
WER_COST_SUB = 1

def WerList(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    for r, h in zip(references, hypotheses):
        res = wer_single(r=r, h=h)
        total_error += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref_len += res["num_ref"]

    wer = (total_error / total_ref_len) * 100
    del_rate = (total_del / total_ref_len) * 100
    ins_rate = (total_ins / total_ref_len) * 100
    sub_rate = (total_sub / total_ref_len) * 100

    return {
        "wer": wer,
        "del_rate": del_rate,
        "ins_rate": ins_rate,
        "sub_rate": sub_rate,
    }


def wer_single(r, h):
    r = r.strip().split()
    h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)

    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }


def edit_distance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j * WER_COST_DEL
            elif j == 0:
                d[i][0] = i * WER_COST_INS
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_DEL
                delete = d[i - 1][j] + WER_COST_INS
                d[i][j] = min(substitute, insert, delete)
    return d

def get_alignment(r, h, d):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 1 * (x + y)

    alignlist = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_DEL:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            alignment = " " + "D" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("D")
            x = max(x, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and d[x][y] == d[x - 1][y] + WER_COST_INS:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " + "I" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("I")
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        alignlist[::-1],
        {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
    )

def WerScore(predictionResult, targetOutData, idx2word, batchSize):
    hypotheses = [" "]
    references = [" "]
    werScoreSum = 0
    for i in range(batchSize):
        predictionResultStr = [idx2word[j] for j in predictionResult[i]]
        targetOutDataStr = [idx2word[j] for j in targetOutData[i]]

        hypotheses[0] = ''.join((x+" " for x in predictionResultStr))
        references[0] = ''.join((x+" " for x in targetOutDataStr))
        assert len(predictionResultStr) == len(predictionResultStr)
        werScore = WerList(hypotheses=hypotheses, references=references)
        werScoreSum = werScoreSum + werScore["wer"]

    # SaveSLRInfo(hypotheses, references, werScoreSum)

    return werScoreSum/batchSize

def WerScore1(predictionResult, targetOutData, idx2word, batchSize):
    hypotheses = [" "]
    references = [" "]
    werScoreSum = 0
    for i in range(batchSize):
        predictionResultStr = [idx2word[j] for j in predictionResult[i]]
        targetOutDataStr = [idx2word[j] for j in targetOutData[i]]

        for j, word in enumerate(predictionResultStr):
            if word[-1].isdigit():
                if not word[0].isdigit():
                    wordList = list(word)
                    wordList.pop(len(word) - 1)
                    word = "".join(wordList)

                    predictionResultStr[j] = word

        hypotheses[0] = ''.join((x+" " for x in predictionResultStr))
        references[0] = ''.join((x+" " for x in targetOutDataStr))
        assert len(predictionResultStr) == len(predictionResultStr)
        werScore = WerList(hypotheses=hypotheses, references=references)
        werScoreSum = werScoreSum + werScore["wer"]

    return werScoreSum/batchSize

def SaveSLRInfo(groud, pred, wer):
    with open('/home/lj/lj/program/python/SLR20240523/pred.txt', 'a') as file:
        file.write(groud[0])
        file.write(pred[0])
        file.write(str(wer))
        file.write("\n")
    return

if __name__ == '__main__':
    predictionResult = [3,4,5,6,6,5]
    decoderInputData = [3,4,5,2,0,0]
    idx2word=[" ", "<bos>", "<eos>", "我", "是", "谁", "她", "不", "人"]

    # GLS Metrics
    gls_wer_score = WerScore(predictionResult, decoderInputData, idx2word)
    print(gls_wer_score)
