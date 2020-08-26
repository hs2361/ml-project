import os, json
from copy import deepcopy
from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Seq2SQL_v1(nn.Module):
    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, old=False):
        super(Seq2SQL_v1, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr

        self.max_wn = 4
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops

        self.scp = SCP(iS, hS, lS, dr)
        self.sap = SAP(iS, hS, lS, dr, n_agg_ops, old=old)
        self.wnp = WNP(iS, hS, lS, dr)
        self.wcp = WCP(iS, hS, lS, dr)
        self.wop = WOP(iS, hS, lS, dr, n_cond_ops)
        self.wvp = WVP_se(iS, hS, lS, dr, n_cond_ops, old=old) # start-end-search-discriminative model


    def forward(self, wemb_n, l_n, wemb_h, l_hpu, l_hs,
                g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None,
                show_p_sc=False, show_p_sa=False,
                show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False,
                knowledge = None,
                knowledge_header = None):

        # sc
        s_sc = self.scp(wemb_n, l_n, wemb_h, l_hpu, l_hs, show_p_sc=show_p_sc,
                        knowledge=knowledge, knowledge_header=knowledge_header)

        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(s_sc)

        # sa
        s_sa = self.sap(wemb_n, l_n, wemb_h, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa,
                        knowledge=knowledge, knowledge_header=knowledge_header)
        if g_sa:
            # it's not necessary though.
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(s_sa)


        # wn
        s_wn = self.wnp(wemb_n, l_n, wemb_h, l_hpu, l_hs, show_p_wn=show_p_wn,
                        knowledge=knowledge, knowledge_header=knowledge_header)

        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)

        # wc
        s_wc = self.wcp(wemb_n, l_n, wemb_h, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True, predict_select_column = pr_sc,
                        knowledge=knowledge, knowledge_header=knowledge_header)

        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)

        # for b, columns in enumerate(pr_wc):
        #     for c in columns:
        #         s_sc[b, c] = -1e+10

        # wo
        s_wo = self.wop(wemb_n, l_n, wemb_h, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_wo=show_p_wo,
                        knowledge=knowledge, knowledge_header=knowledge_header)

        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)

        # wv
        s_wv = self.wvp(wemb_n, l_n, wemb_h, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv,
                        knowledge=knowledge, knowledge_header=knowledge_header)

        return s_sc, s_sa, s_wn, s_wc, s_wo, s_wv

    def beam_forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, engine, tb,
                     nlu_t, nlu_wp_t, wp_to_wh_index, nlu,
                     beam_size=4,
                     show_p_sc=False, show_p_sa=False,
                     show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False,
                     knowledge = None,
                     knowledge_header = None):
        """
        Execution-guided beam decoding.
        """
        # s_sc = [batch_size, header_len]
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc,
                        knowledge=knowledge, knowledge_header=knowledge_header)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape

        # minimum_hs_length = min(l_hs)
        # beam_size = minimum_hs_length if beam_size > minimum_hs_length else beam_size

        # sa
        # Construct all possible sc_sa_score
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops]).to(device)
        prob_sca = torch.zeros_like(prob_sc_sa).to(device)

        # get the top-k indices.  pr_sc_beam = [B, beam_size]
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)

        # calculate and predict s_sa.
        for i_beam in range(beam_size):
            pr_sc = list( array(pr_sc_beam)[:,i_beam] ) # pr_sc = [batch_size]
            s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa,
                        knowledge=knowledge, knowledge_header=knowledge_header)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa

            prob_sc_selected = prob_sc[range(bS), pr_sc] # [B]
            prob_sca[:,i_beam,:] =  (prob_sa.t() * prob_sc_selected).t()
            # [mcL, B] * [B] -> [mcL, B] (element-wise multiplication)
            # [mcL, B] -> [B, mcL]

        # Calculate the dimension of tensor
        # tot_dim = len(prob_sca.shape)

        # First flatten to 1-d
        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        # Now as sc_idx is already sorted, re-map them properly.

        idxs = remap_sc_idx(idxs, pr_sc_beam) # [sc_beam_idx, sa_idx] -> [sc_idx, sa_idx]
        idxs_arr = array(idxs)
        # [B, beam_size, remainig dim]
        # idxs[b][0] gives first probable [sc_idx, sa_idx] pairs.
        # idxs[b][1] gives of second.

        # Calculate prob_sca, a joint probability
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS),beam_idx_sca,0]
            pr_sa = idxs_arr[range(bS),beam_idx_sca,1]

            # map index properly

            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)

            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1: # wrong pair
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True

            if sum(beam_meet_the_final) == bS:
                break


        # Now pr_sc, pr_sa are properly predicted.
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)

        # Now, Where-clause beam search.
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn,
                        knowledge=knowledge, knowledge_header=knowledge_header)
        prob_wn = F.softmax(s_wn, dim=-1).detach().to('cpu').numpy()

        # Found "executable" most likely 4(=max_num_of_conditions) where-clauses.
        # wc
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True,
                        knowledge=knowledge, knowledge_header=knowledge_header)
        prob_wc = F.sigmoid(s_wc).detach().to('cpu').numpy()
        # pr_wc_sorted_by_prob = pred_wc_sorted_by_prob(s_wc)

        # get max_wn # of most probable columns & their prob.
        pr_wn_max = [self.max_wn]*bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc) # if some column do not have executable where-claouse, omit that column
        prob_wc_max = zeros([bS, self.max_wn])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b,:] = prob_wc[b,pr_wc_max1]

        # get most probable max_wn where-clouses
        # wo
        s_wo_max = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, show_p_wo=show_p_wo,
                        knowledge=knowledge, knowledge_header=knowledge_header)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().to('cpu').numpy()
        # [B, max_wn, n_cond_op]

        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        for i_op  in range(self.n_cond_ops-1):
            pr_wo_temp = [ [i_op]*self.max_wn ]*bS
            # wv
            s_wv = self.wvp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, wo=pr_wo_temp, show_p_wv=show_p_wv,
                        knowledge=knowledge, knowledge_header=knowledge_header)
            prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()

            # prob_wv
            pr_wvi_beam, prob_wvi_beam = pred_wvi_se_beam(self.max_wn, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)
            prob_wvi_beam_op_list.append(prob_wvi_beam)
            # pr_wvi_beam = [B, max_wn, k_logit**2 [st, ed] paris]

            # pred_wv_beam

        # Calculate joint probability of where-clause
        # prob_w = [batch, wc, wo, wv] = [B, max_wn, n_cond_op, n_pairs]
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.max_wn, self.n_cond_ops-1, n_wv_beam_pairs])
        for b in range(bS):
            for i_wn in range(self.max_wn):
                for i_op in range(self.n_cond_ops-1): # do not use final one
                    for i_wv_beam in range(n_wv_beam_pairs):
                        # i_wc = pr_wc_max[b][i_wn] # already done
                        p_wc = prob_wc_max[b, i_wn]
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv

        # Perform execution guided decoding
        conds_max = []
        prob_conds_max = []
        # while len(conds_max) < self.max_wn:
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        # idxs = [B, i_wc_beam, i_op, i_wv_pairs]

        # Construct conds1
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]

                # get wv_str
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [wp_to_wh_index[b]], [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]

                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2] ]

                # test execution
                # print(nlu[b])
                # print(tb[b]['id'], tb[b]['types'], pr_sc[b], pr_sa[b], [conds11])
                pr_ans = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], [conds11])
                if bool(pr_ans):
                    # pr_ans is not empty!
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)

            # May need to do more exhuastive search?
            # i.e. up to.. getting all executable cases.

        # Calculate total probability to decide the number of where-clauses
        pr_sql_i = []
        prob_wn_w = []
        pr_wn_based_on_prob = []

        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len( conds_max[b] )
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])  # wn=0 case.
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn+1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)

            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_sql_i.append(pr_sql_i1)
        # s_wv = [B, max_wn, max_nlu_tokens, 2]
        return prob_sca, prob_w, prob_wn_w, pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_sql_i

class SCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(SCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.question_knowledge_dim = 5
        self.header_knowledge_dim = 3
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS + self.question_knowledge_dim, hS + self.header_knowledge_dim)
        self.W_c = nn.Linear(hS + self.question_knowledge_dim, hS)
        self.W_hs = nn.Linear(hS+self.header_knowledge_dim, hS)
        self.sc_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=False,
                knowledge=None,
                knowledge_header=None):
        # Encode
        mL_n = max(l_n)
        bS = len(l_hs)
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]
        knowledge = [k + (mL_n - len(k)) * [0] for k in knowledge]
        knowledge = torch.tensor(knowledge).unsqueeze(-1)

        feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1,
                                                                              index=knowledge,
                                                                              value=1).to(device)
        wenc_n = torch.cat([wenc_n, feature], -1)
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]
        knowledge_header = [k + (max(l_hs) - len(k)) * [0] for k in knowledge_header]
        knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1,
                                                                                        index=knowledge_header,
                                                                                        value=1).to(device)
        wenc_hs = torch.cat([wenc_hs, feature2], -1)
        bS = len(l_hs)
        mL_n = max(l_n)

        #   [bS, mL_hs, 100] * [bS, 100, mL_n] -> [bS, mL_hs, mL_n]
        att_h = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))

        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_h[b, :, l_n1:] = -10000000000

        p_n = self.softmax_dim2(att_h)
        if show_p_sc:
            # p = [b, hs, n]
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001, figsize=(12,3.5))
            # subplot(6,2,7)
            subplot2grid((7,2), (3, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p_n[0][i_h][:].data.numpy() - i_h, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('sc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()



        #   p_n [ bS, mL_hs, mL_n]  -> [ bS, mL_hs, mL_n, 1]
        #   wenc_n [ bS, mL_n, 100] -> [ bS, 1, mL_n, 100]
        #   -> [bS, mL_hs, mL_n, 100] -> [bS, mL_hs, 100]
        c_n = torch.mul(p_n.unsqueeze(3), wenc_n.unsqueeze(1)).sum(dim=2)

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)
        s_sc = self.sc_out(vec).squeeze(2) # [bS, mL_hs, 1] -> [bS, mL_hs]


        # Penalty
        mL_hs = max(l_hs)
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                s_sc[b, l_hs1:] = -10000000000

        return s_sc

class SAP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_agg_ops=-1, old=False):
        super(SAP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.question_knowledge_dim = 5
        self.header_knowledge_dim = 3
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS + self.question_knowledge_dim, hS + self.header_knowledge_dim)
        self.sa_out = nn.Sequential(nn.Linear(hS + self.question_knowledge_dim, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, n_agg_ops))  # Fixed number of aggregation operator.

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

        if old:
            # for backwoard compatibility
            self.W_c = nn.Linear(hS, hS)
            self.W_hs = nn.Linear(hS, hS)

    # wemb_hpu [batch_size*header_num, max_header_len, hidden_dim]
    # l_hpu [batch_size*header_num]
    # l_hs [batch_size]
    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=False,
                knowledge=None,
                knowledge_header=None):
        # Encode
        mL_n = max(l_n)
        bS = len(l_hs)
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]
        knowledge = [k + (mL_n - len(k)) * [0] for k in knowledge]
        knowledge = torch.tensor(knowledge).unsqueeze(-1)

        feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1,
                                                                              index=knowledge,
                                                                              value=1).to(device)
        wenc_n = torch.cat([wenc_n, feature], -1)

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]
        knowledge_header = [k + (max(l_hs) - len(k)) * [0] for k in knowledge_header]
        knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1,
                                                                                        index=knowledge_header,
                                                                                        value=1).to(device)
        wenc_hs = torch.cat([wenc_hs, feature2], -1)
        bS = len(l_hs)
        mL_n = max(l_n)

        wenc_hs_ob = wenc_hs[list(range(bS)), pr_sc]  # list, so one sample for each batch.

        # [bS, mL_n, 100] * [bS, 100, 1] -> [bS, mL_n]
        att = torch.bmm(self.W_att(wenc_n), wenc_hs_ob.unsqueeze(2)).squeeze(2)

        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, l_n1:] = -10000000000
        # [bS, mL_n]
        p = self.softmax_dim1(att)

        if show_p_sa:
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,3)
            cla()
            plot(p[0].data.numpy(), '--rs', ms=7)
            title('sa: nlu_weight')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
            
        #    [bS, mL_n, 100] * ( [bS, mL_n, 1] -> [bS, mL_n, 100])
        #       -> [bS, mL_n, 100] -> [bS, 100]
        c_n = torch.mul(wenc_n, p.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_sa = self.sa_out(c_n)

        return s_sa

