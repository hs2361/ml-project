from dbengine_sqlnet import DBEngine

import os
import seq2sql_model_training_functions
import corenlp_local
import load_data
import bert_training
import infer_functions
import torch
from tqdm.notebook import tqdm
import seq2sql_model_testing
#import torch_xla
#import torch_xla.core.xla_model as xm


def train(seq2sql_model,bert_model,model_optimizer,roberta_optimizer,bert_tokenizer,bert_configs,path_wikisql,train_loader):

        bert_model.train()
        seq2sql_model.train()
        
        results=[]
        ave_loss = 0
        cnt = 0  # count the # of examples
        cnt_sc = 0  # count the # of correct predictions of select column
        cnt_sa = 0  # of selectd aggregation
        cnt_wn = 0  # of where number
        cnt_wc = 0  # of where column
        cnt_wo = 0  # of where operator
        cnt_wv = 0  # of where-value
        cnt_wvi = 0  # of where-value index (on question tokens)
        cnt_lx = 0  # of logical form acc
        cnt_x = 0  # of execution acc

        #Train Function Parameters
        st_pos=0
        max_seq_length = 222
        num_target_layers=2
        accumulate_gradients=1 
        check_grad=True
        opt_bert=roberta_optimizer 
        path_db=None
        opt=model_optimizer

        # Engine for SQL querying.
        engine = DBEngine(os.path.join(path_wikisql, f"train.db"))

        for iB, t in enumerate(tqdm(train_loader)):

            cnt += len(t)

            # if iB > 2:
            #     break
            # Get fields
            nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = load_data.get_fields(t)
            # nlu  : natural language utterance
            # nlu_t: tokenized nlu
            # sql_i: canonical form of SQL query
            # sql_q: full SQL query text. Not used.
            # sql_t: tokenized SQL query
            # tb   : table metadata. No row data needed
            # hs_t : tokenized headers. Not used.

            g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = bert_training.get_g(sql_i)
            # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
            g_wvi_corenlp = corenlp_local.get_g_wvi_corenlp(t)

            wemb_n, wemb_h, l_n, l_hpu, l_hs, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
                = bert_training.get_wemb_bert(bert_configs, bert_model, bert_tokenizer, nlu_t, hds, max_seq_length,
                                num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
            # wemb_n: natural language embedding
            # wemb_h: header embedding
            # l_n: token lengths of each question
            # l_hpu: header token lengths
            # l_hs: the number of columns (headers) of the tables.
            try:
                #
                g_wvi = corenlp_local.get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            except:
                # Exception happens when where-condition is not found in nlu_tt.
                # In this case, that train example is not used.
                # During test, that example considered as wrongly answered.
                # e.g. train: 32.
                continue

            knowledge = []
            for k in t:
                if "bertindex_knowledge" in k:
                    knowledge.append(k["bertindex_knowledge"])
                else:
                    knowledge.append(max(l_n)*[0])

            knowledge_header = []
            for k in t:
                if "header_knowledge" in k:
                    knowledge_header.append(k["header_knowledge"])
                else:
                    knowledge_header.append(max(l_hs) * [0])
            # score

            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = seq2sql_model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                    g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc,g_wo=g_wo, g_wvi=g_wvi,
                                                    knowledge = knowledge,
                                                    knowledge_header = knowledge_header)

            # Calculate loss & step
            loss = seq2sql_model_training_functions.Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # Calculate gradient
            if iB % accumulate_gradients == 0:  # mode
                # at start, perform zero_grad
                opt.zero_grad()
                if opt_bert:
                    opt_bert.zero_grad()
                loss.backward()
                if accumulate_gradients == 1:
                    opt.step()
                    if opt_bert:
                        opt_bert.step()
            elif iB % accumulate_gradients == (accumulate_gradients - 1):
                # at the final, take step with accumulated graident
                loss.backward()
                opt.step()
                if opt_bert:
                    opt_bert.step()
            else:
                # at intermediate stage, just accumulates the gradients
                loss.backward()

            # Prediction
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = seq2sql_model_training_functions.pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = seq2sql_model_training_functions.convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # Sort pr_wc:
            #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
            #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
            pr_wc_sorted = seq2sql_model_training_functions.sort_pr_wc(pr_wc, g_wc)
            pr_sql_i = seq2sql_model_training_functions.generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)
            g_sql_q = seq2sql_model_testing.generate_sql_q(sql_i, tb)
            pr_sql_q = seq2sql_model_testing.generate_sql_q(pr_sql_i, tb)
            # Cacluate accuracy
            cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
            cnt_wc1_list, cnt_wo1_list, \
            cnt_wvi1_list, cnt_wv1_list = seq2sql_model_training_functions.get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                            sql_i, pr_sql_i,
                                                            mode='train')

            cnt_lx1_list = seq2sql_model_training_functions.get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                            cnt_wo1_list, cnt_wv1_list)
            # lx stands for logical form accuracy
            # Execution accuracy test.
            cnt_x1_list, g_ans, pr_ans = seq2sql_model_training_functions.get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)
            # statistics
            ave_loss += loss.item()

            # count
            cnt_sc += sum(cnt_sc1_list)
            cnt_sa += sum(cnt_sa1_list)
            cnt_wn += sum(cnt_wn1_list)
            cnt_wc += sum(cnt_wc1_list)
            cnt_wo += sum(cnt_wo1_list)
            cnt_wvi += sum(cnt_wvi1_list)
            cnt_wv += sum(cnt_wv1_list)
            cnt_lx += sum(cnt_lx1_list)
            cnt_x += sum(cnt_x1_list)

        ave_loss /= cnt
        acc_sc = cnt_sc / cnt
        acc_sa = cnt_sa / cnt
        acc_wn = cnt_wn / cnt
        acc_wc = cnt_wc / cnt
        acc_wo = cnt_wo / cnt
        acc_wvi = cnt_wvi / cnt
        acc_wv = cnt_wv / cnt
        acc_lx = cnt_lx / cnt
        acc_x = cnt_x / cnt
        acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

        return acc


def test(seq2sql_model,bert_model,model_optimizer,bert_tokenizer,bert_configs,path_wikisql,test_loader):

        bert_model.eval()
        seq2sql_model.eval()

        results=[]
        cnt_list=[]
        ave_loss = 0
        loss = []
        cnt = 0  # count the # of examples
        cnt_sc = 0  # count the # of correct predictions of select column
        cnt_sa = 0  # of selectd aggregation
        cnt_wn = 0  # of where number
        cnt_wc = 0  # of where column
        cnt_wo = 0  # of where operator
        cnt_wv = 0  # of where-value
        cnt_wvi = 0  # of where-value index (on question tokens)
        cnt_lx = 0  # of logical form acc
        cnt_x = 0  # of execution acc

        #Train Function Parameters
        st_pos=0
        max_seq_length = 222
        num_target_layers=2
        accumulate_gradients=1 
        check_grad=True
        opt_bert=None 
        path_db=None
        opt=model_optimizer

        # Engine for SQL querying.
        engine = DBEngine(os.path.join(path_wikisql, f"dev.db"))

        for iB, t in enumerate(tqdm(test_loader)):
            cnt += len(t)

            # if iB > 2:
            #     break
            # Get fields
            nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = load_data.get_fields(t)
            # nlu  : natural language utterance
            # nlu_t: tokenized nlu
            # sql_i: canonical form of SQL query
            # sql_q: full SQL query text. Not used.
            # sql_t: tokenized SQL query
            # tb   : table metadata. No row data needed
            # hs_t : tokenized headers. Not used.

            g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = bert_training.get_g(sql_i)
            # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
            g_wvi_corenlp = corenlp_local.get_g_wvi_corenlp(t)

            wemb_n, wemb_h, l_n, l_hpu, l_hs, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
                = bert_training.get_wemb_bert(bert_configs, bert_model, bert_tokenizer, nlu_t, hds, max_seq_length,
                                num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

            # wemb_n: natural language embedding
            # wemb_h: header embedding
            # l_n: token lengths of each question
            # l_hpu: header token lengths
            # l_hs: the number of columns (headers) of the tables.
            try:
                #
                g_wvi = corenlp_local.get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            except:
                # Exception happens when where-condition is not found in nlu_tt.
                # In this case, that train example is not used.
                # During test, that example considered as wrongly answered.
                # e.g. train: 32.
                for b in range(len(nlu)):
                    results1 = {}
                    results1["error"] = "Skip happened"
                    results1["nlu"] = nlu[b]
                    results1["table_id"] = tb[b]["id"]
                    results.append(results1)
                continue
               

            knowledge = []
            for k in t:
                if "bertindex_knowledge" in k:
                    knowledge.append(k["bertindex_knowledge"])
                else:
                    knowledge.append(max(l_n)*[0])

            knowledge_header = []
            for k in t:
                if "header_knowledge" in k:
                    knowledge_header.append(k["header_knowledge"])
                else:
                    knowledge_header.append(max(l_hs) * [0])

            # s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = seq2sql_model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
            #                                         g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc,g_wo=g_wo, g_wvi=g_wvi,
            #                                         knowledge = knowledge,
            #                                         knowledge_header = knowledge_header)

            # # Calculate loss & step
            # loss = seq2sql_model_training_functions.Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # score
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = seq2sql_model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=4,
                                                       knowledge=knowledge,
                                                       knowledge_header=knowledge_header)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = infer_functions.sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None  # not used
            pr_wv_str = None
            pr_wv_str_wp = None
            loss = torch.tensor([0])
            # pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = seq2sql_model_training_functions.pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            # pr_wv_str, pr_wv_str_wp = seq2sql_model_training_functions.convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # Sort pr_wc:
            #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
            #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
            # pr_wc_sorted = seq2sql_model_training_functions.sort_pr_wc(pr_wc, g_wc)
            # pr_sql_i = seq2sql_model_training_functions.generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)
            g_sql_q = seq2sql_model_testing.generate_sql_q(sql_i, tb)
            pr_sql_q = seq2sql_model_testing.generate_sql_q(pr_sql_i, tb)

            for b, pr_sql_i1 in enumerate(pr_sql_i):
                results1 = {}
                results1["query"] = pr_sql_i1
                results1["table_id"] = tb[b]["id"]
                results1["nlu"] = nlu[b]
                results.append(results1)

            # Cacluate accuracy
            cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
            cnt_wc1_list, cnt_wo1_list, \
            cnt_wvi1_list, cnt_wv1_list = seq2sql_model_training_functions.get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                            sql_i, pr_sql_i,
                                                            mode='test')

            cnt_lx1_list = seq2sql_model_training_functions.get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                            cnt_wo1_list, cnt_wv1_list)
            # lx stands for logical form accuracy

            # Execution accuracy test.
            cnt_x1_list, g_ans, pr_ans = seq2sql_model_training_functions.get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

            # statistics
            # ave_loss += loss.item()

            # count
            cnt_sc += sum(cnt_sc1_list)
            cnt_sa += sum(cnt_sa1_list)
            cnt_wn += sum(cnt_wn1_list)
            cnt_wc += sum(cnt_wc1_list)
            cnt_wo += sum(cnt_wo1_list)
            cnt_wvi += sum(cnt_wvi1_list)
            cnt_wv += sum(cnt_wv1_list)
            cnt_lx += sum(cnt_lx1_list)
            cnt_x += sum(cnt_x1_list)

            cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,cnt_x1_list]
            cnt_list.append(cnt_list1)

        # ave_loss /= cnt
        acc_sc = cnt_sc / cnt
        acc_sa = cnt_sa / cnt
        acc_wn = cnt_wn / cnt
        acc_wc = cnt_wc / cnt
        acc_wo = cnt_wo / cnt
        acc_wvi = cnt_wvi / cnt
        acc_wv = cnt_wv / cnt
        acc_lx = cnt_lx / cnt
        acc_x = cnt_x / cnt

        acc = [None, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
        return acc, results, cnt_list