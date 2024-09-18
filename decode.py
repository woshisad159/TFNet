import torch
import ctcdecode
from itertools import groupby
import torch.nn.functional as F
from six.moves import xrange


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        # self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        # self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.g2i_dict = {}
        for k, v in gloss_dict.items():
            if v == 0:
                continue
            self.g2i_dict[k] = v
        self.i2g_dict = {v: k for k, v in self.g2i_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
                                                    num_processes=10)
        # self.ctc_decoder = None

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            tmp = [(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)]
            if len(tmp) > 0:
                ret_list.append(tmp)
            else:
                try:
                    ret_list.append(ret_list[-1])
                except:
                    ret_list.append([('EMPTY', 0)])

        return ret_list, first_result

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        # result_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
            # result_list.append(max_result)
        # return ret_list, result_list
        return ret_list

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    params: [vocab_size, T], logits.softmax(-1). T 是输入序列的长度，vocab_size是词表大小。
    seq: [seq_len] 输出序列的长度。

    CTC loss function.
    params - n x m matrix of n-D probability distributions over m frames.
    seq - sequence of phone id's for given example.
    is_prob - whether params have already passed through a softmax
    Returns objective and gradient.
    """
    batchSize = len(target_lengths)
    numphones = log_probs.shape[-1]  # Number of labels
    n = 0

    for i in range(batchSize):
        seqLen = target_lengths[i]  # Length of label sequence (# phones)
        L = 2 * seqLen + 1  # Length of label sequence with blanks, 拓展后的 l'.
        T = input_lengths[i]  # Length of utterance (time)

        # 建立表格 l' x T.
        alphas = torch.zeros((L, T))  # 前向概率
        betas = torch.zeros((L, T))  # 后向概率

        # 这里dp的map：
        # 横轴为 2*seq_len+1, 也就是 ground truth label中每个token前后插入 blank
        # 纵轴是 T frames

        log_probs = F.softmax(log_probs, dim=-1)

        # 初始条件：T=0时，只能为 blank 或 seq[0]
        alphas[0, 0] = log_probs[0, i,  blank]
        alphas[1, 0] = log_probs[0, i,  targets[n]]
        # T=0， alpha[:, 0] 其他的全部为 0

        c = torch.sum(alphas[:, 0])
        alphas[:, 0] = alphas[:, 0] / c  # 这里 T=0 时刻所有可能节点的概率要归一化

        llForward = torch.log(c)  # 转换为log域

        for t in xrange(1, T):
            # 第一个循环： 计算每个时刻所有可能节点的概率和
            start = max(0, L - 2 * (T - t))  # 对于时刻 t, 其可能的节点.与公式2一致。
            end = min(2 * t + 2, L)  # 对于时刻 t，最大节点范围不可能超过 2t+2
            for s in xrange(start, L):
                l = int((s - 1) / 2)
                # blank，节点s在偶数位置，意味着s为blank
                if s % 2 == 0:
                    if s == 0: # 初始位置，单独讨论
                        alphas[s, t] = alphas[s, t - 1] * log_probs[t, i, blank]
                    else:
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * log_probs[t, i, blank]
                # s为奇数，非空
                # l = (s-1/2) 就是 s 所对应的 lable 中的字符。
                # ((s-2)-1)/2 = (s-1)/2-1 = l-1 就是 s-2 对应的lable中的字符
                elif s == 1 or targets[l] == targets[l - 1]:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * log_probs[t, i, targets[l]]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                                   * log_probs[t, i, targets[l]]

            # normalize at current time (prevent underflow)
            c = torch.sum(alphas[start:end, t])
            alphas[start:end, t] = alphas[start:end, t] / c
            llForward = llForward + torch.log(c)

        n = n + target_lengths[i]
        sumN = torch.sum(input_lengths)

    return llForward / sumN