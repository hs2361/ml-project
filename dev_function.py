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


# def train(seq2sql_model,roberta_model,model_optimizer,roberta_optimizer,roberta_tokenizer,roberta_config,path_wikisql,train_loader):

#     roberta_model.train()
#     seq2sql_model.train()
    
#     results=[]
#     average_loss = 0
    
#     count_select_column = 0  # count the # of correct predictions of select column
#     count_select_agg = 0  # of selectd aggregation
#     count_where_number = 0  # of where number
#     count_where_column = 0  # of where column
#     count_where_operator = 0  # of where operator
#     count_where_value = 0  # of where-value
#     count_where_value_index = 0  # of where-value index (on question tokens)
#     count_logical_form_acc = 0  # of logical form accuracy
#     count_execution_acc = 0  # of execution accuracy


#     # Engine for SQL querying.
#     engine = DBEngine(os.path.join(path_wikisql, f"train.db"))
#     count = 0  # count the # of examples
#     for batch_index, batch in enumerate(tqdm(train_loader)):
#         count += len(batch)

#         # if batch_index > 2:
#         #     break
#         # Get fields

#         # nlu  : natural language utterance
#         # nlu_t: tokenized nlu
#         # sql_i: canonical form of SQL query
#         # sql_q: full SQL query text. Not used.
#         # sql_t: tokenized SQL query
#         # tb   : table metadata. No row data needed
#         # hs_t : tokenized headers. Not used.
#         natural_lang_utterance, natural_lang_utterance_tokenized, sql_canonical, \
#             _, _, table_metadata, _, headers = load_data.get_fields(batch)


#         select_column_ground, select_agg_ground, where_number_ground, \
#             where_column_ground, where_operator_ground, _ = bert_training.get_ground_truth_values(sql_canonical)
#         # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        

#         natural_lang_embeddings, header_embeddings, question_token_length, header_token_length, header_count, \
#         natural_lang_double_tokenized, punkt_to_roberta_token_indices, roberta_to_punkt_token_indices \
#             = bert_training.get_wemb_roberta(roberta_config, roberta_model, roberta_tokenizer, 
#                                         natural_lang_utterance_tokenized, headers,max_seq_length= 222,
#                                         num_out_layers_n=2, num_out_layers_h=2)
#         # natural_lang_embeddings: natural language embedding
#         # header_embeddings: header embedding
#         # question_token_length: token lengths of each question
#         # header_token_length: header token lengths
#         # header_count: the number of columns (headers) of the tables.

#         where_value_index_ground_corenlp = corenlp_local.get_g_wvi_corenlp(batch)
#         try:
#             #
#             where_value_index_ground = corenlp_local.get_g_wvi_bert_from_g_wvi_corenlp(punkt_to_roberta_token_indices, where_value_index_ground_corenlp)
#         except:
#             # Exception happens when where-condition is not found in natural_lang_double_tokenized.
#             # In this case, that train example is not used.
#             # During test, that example considered as wrongly answered.
#             # e.g. train: 32.
#             continue

#         knowledge = []
#         for k in batch:
#             if "bertindex_knowledge" in k:
#                 knowledge.append(k["bertindex_knowledge"])
#             else:
#                 knowledge.append(max(question_token_length)*[0])

#         knowledge_header = []
#         for k in batch:
#             if "header_knowledge" in k:
#                 knowledge_header.append(k["header_knowledge"])
#             else:
#                 knowledge_header.append(max(header_count) * [0])

#         # score

#         select_column_score, select_agg_score, where_number_score, where_column_score,\
#             where_operator_score, where_value_score = seq2sql_model(natural_lang_embeddings, question_token_length, header_embeddings, 
#                                                 header_token_length, header_count,
#                                                 g_sc=select_column_ground, g_sa=select_agg_ground,
#                                                 g_wn=where_number_ground, g_wc=where_column_ground,
#                                                 g_wo=where_operator_ground, g_wvi=where_value_index_ground,
#                                                 knowledge = knowledge,
#                                                 knowledge_header = knowledge_header)

#         # Calculate loss & step
#         loss = seq2sql_model_training_functions.Loss_sw_se(select_column_score, select_agg_score, where_number_score, 
#                                                 where_column_score, where_operator_score, where_value_score, 
#                                                 select_column_ground, select_agg_ground, 
#                                                 where_number_ground, where_column_ground,
#                                                 where_operator_ground, where_value_index_ground)

        
#         model_optimizer.zero_grad()
#         if roberta_optimizer:
#             roberta_optimizer.zero_grad()
#         loss.backward()
#         model_optimizer.step()
#         if roberta_optimizer:
#             roberta_optimizer.step()
        

#         # Prediction
#         select_column_predict, select_agg_predict, where_number_predict, \
#             where_column_predict, where_operator_predict, where_val_index_predict = seq2sql_model_training_functions.pred_sw_se(
#                                                                                     select_column_score, select_agg_score, where_number_score, 
#                                                                                     where_column_score, where_operator_score, where_value_score)
#         where_value_string_predict, _ = seq2sql_model_training_functions.convert_pr_wvi_to_string(
#                                                                         where_val_index_predict, 
#                                                                         natural_lang_utterance_tokenized, natural_lang_double_tokenized, 
#                                                                         roberta_to_punkt_token_indices, natural_lang_utterance)
        
        
#         # Sort where_column_predict:
#         #   Sort where_column_predict when training the model as where_operator_predict and where_val_index_predict are predicted using ground-truth where-column (g_wc)
#         #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
#         where_column_predict_sorted = seq2sql_model_training_functions.sort_pr_wc(where_column_predict, where_column_ground)
        
#         sql_canonical_predict = seq2sql_model_training_functions.generate_sql_i(
#                                                                         select_column_predict, select_agg_predict, where_number_predict,
#                                                                         where_column_predict_sorted, where_operator_predict, 
#                                                                         where_value_string_predict, natural_lang_utterance)

#         # Cacluate accuracy
#         select_col_batchlist, select_agg_batchlist, where_number_batchlist, \
#             where_column_batchlist, where_operator_batchlist, where_value_index_batchlist, \
#             where_value_batchlist = seq2sql_model_training_functions.get_cnt_sw_list(
#                                                         select_column_ground, select_agg_ground, 
#                                                         where_number_ground, where_column_ground,
#                                                         where_operator_ground, where_value_index_ground,
#                                                         select_column_predict, select_agg_predict, where_number_predict, 
#                                                         where_column_predict, where_operator_predict, where_val_index_predict,
#                                                         sql_canonical, sql_canonical_predict,
#                                                         mode='train')

#         logical_form_acc_batchlist = seq2sql_model_training_functions.get_cnt_lx_list(
#                                                         select_col_batchlist, select_agg_batchlist, where_number_batchlist, 
#                                                         where_column_batchlist,where_operator_batchlist, where_value_batchlist)
#         # lx stands for logical form accuracy
#         # Execution accuracy test.
#         execution_acc_batchlist, _, _ = seq2sql_model_training_functions.get_cnt_x_list(
#                                                         engine, table_metadata, select_column_ground, select_agg_ground, 
#                                                         sql_canonical, select_column_predict, select_agg_predict, sql_canonical_predict)
#         # statistics
#         average_loss += loss.item()

#         # count
#         count_select_column += sum(select_col_batchlist)
#         count_select_agg += sum(select_agg_batchlist)
#         count_where_number += sum(where_number_batchlist)
#         count_where_column += sum(where_column_batchlist)
#         count_where_operator += sum(where_operator_batchlist)
#         count_where_value_index += sum(where_value_index_batchlist)
#         count_where_value += sum(where_value_batchlist)
#         count_logical_form_acc += sum(logical_form_acc_batchlist)
#         count_execution_acc += sum(execution_acc_batchlist)

#     average_loss /= count
#     select_column_acc = count_select_column / count
#     select_agg_acc = count_select_agg / count
#     where_number_acc = count_where_number / count
#     where_column_acc = count_where_column / count
#     where_operator_acc = count_where_operator / count
#     where_value_index_acc = count_where_value_index / count
#     where_value_acc = count_where_value / count
#     logical_form_acc = count_logical_form_acc / count
#     execution_acc = count_execution_acc / count
#     accuracy = [average_loss, select_column_acc, select_agg_acc, where_number_acc, where_column_acc,
#                 where_operator_acc, where_value_index_acc, where_value_acc, logical_form_acc, execution_acc]

#     return accuracy

def train(seq2sql_model,roberta_model,model_optimizer,roberta_optimizer,bert_tokenizer,bert_configs,path_wikisql,train_loader):

        roberta_model.train()
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
                = bert_training.get_wemb_bert(bert_configs, roberta_model, bert_tokenizer, nlu_t, hds, max_seq_length,
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


def test(seq2sql_model,roberta_model,model_optimizer,roberta_tokenizer,roberta_config,path_wikisql,test_loader,mode="dev"):
    
    roberta_model.eval()
    seq2sql_model.eval()

    count_batchlist=[]    
    results=[]


    count_select_column = 0  # count the # of correct predictions of select column
    count_select_agg = 0  # of selectd aggregation
    count_where_number = 0  # of where number
    count_where_column = 0  # of where column
    count_where_operator = 0  # of where operator
    count_where_value = 0  # of where-value
    count_where_value_index = 0  # of where-value index (on question tokens)
    count_logical_form_acc = 0  # of logical form accuracy
    count_execution_acc = 0  # of execution accurac


    # Engine for SQL querying.
    engine = DBEngine(os.path.join(path_wikisql, mode+".db"))

    count = 0
    for batch_index, batch in enumerate(tqdm(test_loader)):
        count += len(batch)

        # if batch_index > 2:
        #     break
        # Get fields
        natural_lang_utterance, natural_lang_utterance_tokenized, sql_canonical, \
            _, _, table_metadata, _, headers = load_data.get_fields(batch)


        select_column_ground, select_agg_ground, where_number_ground, \
            where_column_ground, where_operator_ground, _ = bert_training.get_ground_truth_values(sql_canonical)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        

        natural_lang_embeddings, header_embeddings, question_token_length, header_token_length, header_count, \
        natural_lang_double_tokenized, punkt_to_roberta_token_indices, roberta_to_punkt_token_indices \
            = bert_training.get_wemb_roberta(roberta_config, roberta_model, roberta_tokenizer, 
                                        natural_lang_utterance_tokenized, headers,max_seq_length= 222,
                                        num_out_layers_n=2, num_out_layers_h=2)
        # natural_lang_embeddings: natural language embedding
        # header_embeddings: header embedding
        # question_token_length: token lengths of each question
        # header_token_length: header token lengths
        # header_count: the number of columns (headers) of the tables.

        where_value_index_ground_corenlp = corenlp_local.get_g_wvi_corenlp(batch)
        try:
            #
            where_value_index_ground = corenlp_local.get_g_wvi_bert_from_g_wvi_corenlp(punkt_to_roberta_token_indices, where_value_index_ground_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            for b in range(len(natural_lang_utterance)):
                curr_results = {}
                curr_results["error"] = "Skip happened"
                curr_results["nlu"] = natural_lang_utterance[b]
                curr_results["table_id"] = table_metadata[b]["id"]
                results.append(curr_results)
            continue
            

        knowledge = []
        for k in batch:
            if "bertindex_knowledge" in k:
                knowledge.append(k["bertindex_knowledge"])
            else:
                knowledge.append(max(question_token_length)*[0])

        knowledge_header = []
        for k in batch:
            if "header_knowledge" in k:
                knowledge_header.append(k["header_knowledge"])
            else:
                knowledge_header.append(max(header_count) * [0])



        # score
        _, _, _, select_column_predict, select_agg_predict, where_number_predict, sql_predict = seq2sql_model.beam_forward(
                                                            natural_lang_embeddings, question_token_length, header_embeddings,
                                                            header_token_length, header_count, table_metadata,
                                                            natural_lang_utterance_tokenized, natural_lang_double_tokenized,
                                                            roberta_to_punkt_token_indices, natural_lang_utterance,
                                                            beam_size=4, knowledge=knowledge, knowledge_header=knowledge_header)

        # sort and generate
        where_column_predict, where_operator_predict, _, sql_predict = infer_functions.sort_and_generate_pr_w(sql_predict)

        # Follosing variables are just for the consistency with no-EG case.
        where_value_index_predict = None  # not used

        for b, sql_predict_instance in enumerate(sql_predict):
            curr_results = {}
            curr_results["query"] = sql_predict_instance
            curr_results["table_id"] = table_metadata[b]["id"]
            curr_results["nlu"] = natural_lang_utterance[b]
            results.append(curr_results)

        # Cacluate accuracy
        select_column_batchlist, select_agg_batchlist, where_number_batchlist, \
        where_column_batchlist, where_operator_batchlist, \
        where_value_index_batchlist, where_value_batchlist = seq2sql_model_training_functions.get_cnt_sw_list(
                                                        select_column_ground, select_agg_ground, where_number_ground,
                                                        where_column_ground, where_operator_ground, where_value_index_ground,
                                                        select_column_predict, select_agg_predict, where_number_predict, where_column_predict, 
                                                        where_operator_predict, where_value_index_predict,
                                                        sql_canonical, sql_predict,
                                                        mode='test')

        logical_form_acc_batchlist = seq2sql_model_training_functions.get_cnt_lx_list(select_column_batchlist, select_agg_batchlist, where_number_batchlist, where_column_batchlist,
                                        where_operator_batchlist, where_value_batchlist)
        # lx stands for logical form accuracy

        # Execution accuracy test.
        execution_acc_batchlist, _, _ = seq2sql_model_training_functions.get_cnt_x_list(
                        engine, table_metadata, select_column_ground, select_agg_ground, sql_canonical, select_column_predict, select_agg_predict, sql_predict)

        # statistics
        # ave_loss += loss.item()

        # count
        count_select_column += sum(select_column_batchlist)
        count_select_agg += sum(select_agg_batchlist)
        count_where_number += sum(where_number_batchlist)
        count_where_column += sum(where_column_batchlist)
        count_where_operator += sum(where_operator_batchlist)
        count_where_value_index += sum(where_value_index_batchlist)
        count_where_value += sum(where_value_batchlist)
        count_logical_form_acc += sum(logical_form_acc_batchlist)
        count_execution_acc += sum(execution_acc_batchlist)

        count_curr_batchlist = [select_column_batchlist, select_agg_batchlist, where_number_batchlist, where_column_batchlist, where_operator_batchlist, where_value_batchlist, logical_form_acc_batchlist,execution_acc_batchlist]
        count_batchlist.append(count_curr_batchlist)

    # ave_loss /= cnt
    select_column_acc = count_select_column / count
    select_agg_acc = count_select_agg / count
    where_number_acc = count_where_number / count
    where_column_acc = count_where_column / count
    where_operator_acc = count_where_operator / count
    where_value_index_acc = count_where_value_index / count
    where_value_acc = count_where_value / count
    logical_form_acc = count_logical_form_acc / count
    execution_acc = count_execution_acc / count

    accuracy = [None, select_column_acc, select_agg_acc, where_number_acc, 
                where_column_acc, where_operator_acc, where_value_index_acc, 
                where_value_acc, logical_form_acc, execution_acc]

    return accuracy, results, count_batchlist


# def test(seq2sql_model,roberta_model,model_optimizer,bert_tokenizer,bert_configs,path_wikisql,test_loader,mode="dev"):
        
#         roberta_model.eval()
#         seq2sql_model.eval()

#         results=[]
#         cnt_list=[]
#         ave_loss = 0
#         loss = []
#         cnt = 0  # count the # of examples
#         cnt_sc = 0  # count the # of correct predictions of select column
#         cnt_sa = 0  # of selectd aggregation
#         cnt_wn = 0  # of where number
#         cnt_wc = 0  # of where column
#         cnt_wo = 0  # of where operator
#         cnt_wv = 0  # of where-value
#         cnt_wvi = 0  # of where-value index (on question tokens)
#         cnt_lx = 0  # of logical form acc
#         cnt_x = 0  # of execution acc

#         #Train Function Parameters
#         st_pos=0
#         max_seq_length = 222
#         num_target_layers=2
#         accumulate_gradients=1 
#         check_grad=True
#         opt_bert=None 
#         path_db=None
#         opt=model_optimizer

#         # Engine for SQL querying.
#         engine = DBEngine(os.path.join(path_wikisql, mode+".db"))

#         for iB, t in enumerate(tqdm(test_loader)):
#             cnt += len(t)

#             # if iB > 2:
#             #     break
#             # Get fields
#             nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = load_data.get_fields(t)
#             # nlu  : natural language utterance
#             # nlu_t: tokenized nlu
#             # sql_i: canonical form of SQL query
#             # sql_q: full SQL query text. Not used.
#             # sql_t: tokenized SQL query
#             # tb   : table metadata. No row data needed
#             # hs_t : tokenized headers. Not used.

#             g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = bert_training.get_g(sql_i)
#             # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
#             g_wvi_corenlp = corenlp_local.get_g_wvi_corenlp(t)

#             wemb_n, wemb_h, l_n, l_hpu, l_hs, \
#             nlu_tt, t_to_tt_idx, tt_to_t_idx \
#                 = bert_training.get_wemb_bert(bert_configs, roberta_model, bert_tokenizer, nlu_t, hds, max_seq_length,
#                                 num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

#             # wemb_n: natural language embedding
#             # wemb_h: header embedding
#             # l_n: token lengths of each question
#             # l_hpu: header token lengths
#             # l_hs: the number of columns (headers) of the tables.
#             try:
#                 #
#                 g_wvi = corenlp_local.get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
#             except:
#                 # Exception happens when where-condition is not found in nlu_tt.
#                 # In this case, that train example is not used.
#                 # During test, that example considered as wrongly answered.
#                 # e.g. train: 32.
#                 for b in range(len(nlu)):
#                     results1 = {}
#                     results1["error"] = "Skip happened"
#                     results1["nlu"] = nlu[b]
#                     results1["table_id"] = tb[b]["id"]
#                     results.append(results1)
#                 continue
               

#             knowledge = []
#             for k in t:
#                 if "bertindex_knowledge" in k:
#                     knowledge.append(k["bertindex_knowledge"])
#                 else:
#                     knowledge.append(max(l_n)*[0])

#             knowledge_header = []
#             for k in t:
#                 if "header_knowledge" in k:
#                     knowledge_header.append(k["header_knowledge"])
#                 else:
#                     knowledge_header.append(max(l_hs) * [0])

#             # s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = seq2sql_model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
#             #                                         g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc,g_wo=g_wo, g_wvi=g_wvi,
#             #                                         knowledge = knowledge,
#             #                                         knowledge_header = knowledge_header)

#             # # Calculate loss & step
#             # loss = seq2sql_model_training_functions.Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

#             # score
#             prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = seq2sql_model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
#                                                                                             l_hs, tb,
#                                                                                             nlu_t, nlu_tt,
#                                                                                             tt_to_t_idx, nlu,
#                                                                                             beam_size=4,
#                                                        knowledge=knowledge,
#                                                        knowledge_header=knowledge_header)
#             # sort and generate
#             pr_wc, pr_wo, pr_wv, pr_sql_i = infer_functions.sort_and_generate_pr_w(pr_sql_i)

#             # Follosing variables are just for the consistency with no-EG case.
#             pr_wvi = None  # not used
#             pr_wv_str = None
#             pr_wv_str_wp = None
#             loss = torch.tensor([0])
#             # pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = seq2sql_model_training_functions.pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
#             # pr_wv_str, pr_wv_str_wp = seq2sql_model_training_functions.convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
#             # Sort pr_wc:
#             #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
#             #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
#             # pr_wc_sorted = seq2sql_model_training_functions.sort_pr_wc(pr_wc, g_wc)
#             # pr_sql_i = seq2sql_model_training_functions.generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)
#             g_sql_q = seq2sql_model_testing.generate_sql_q(sql_i, tb)
#             pr_sql_q = seq2sql_model_testing.generate_sql_q(pr_sql_i, tb)

#             for b, pr_sql_i1 in enumerate(pr_sql_i):
#                 results1 = {}
#                 results1["query"] = pr_sql_i1
#                 results1["table_id"] = tb[b]["id"]
#                 results1["nlu"] = nlu[b]
#                 results.append(results1)

#             # Cacluate accuracy
#             cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
#             cnt_wc1_list, cnt_wo1_list, \
#             cnt_wvi1_list, cnt_wv1_list = seq2sql_model_training_functions.get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
#                                                             pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
#                                                             sql_i, pr_sql_i,
#                                                             mode='test')

#             cnt_lx1_list = seq2sql_model_training_functions.get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
#                                             cnt_wo1_list, cnt_wv1_list)
#             # lx stands for logical form accuracy

#             # Execution accuracy test.
#             cnt_x1_list, g_ans, pr_ans = seq2sql_model_training_functions.get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

#             # statistics
#             # ave_loss += loss.item()

#             # count
#             cnt_sc += sum(cnt_sc1_list)
#             cnt_sa += sum(cnt_sa1_list)
#             cnt_wn += sum(cnt_wn1_list)
#             cnt_wc += sum(cnt_wc1_list)
#             cnt_wo += sum(cnt_wo1_list)
#             cnt_wvi += sum(cnt_wvi1_list)
#             cnt_wv += sum(cnt_wv1_list)
#             cnt_lx += sum(cnt_lx1_list)
#             cnt_x += sum(cnt_x1_list)

#             cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,cnt_x1_list]
#             cnt_list.append(cnt_list1)

#         # ave_loss /= cnt
#         acc_sc = cnt_sc / cnt
#         acc_sa = cnt_sa / cnt
#         acc_wn = cnt_wn / cnt
#         acc_wc = cnt_wc / cnt
#         acc_wo = cnt_wo / cnt
#         acc_wvi = cnt_wvi / cnt
#         acc_wv = cnt_wv / cnt
#         acc_lx = cnt_lx / cnt
#         acc_x = cnt_x / cnt

#         acc = [None, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
#         return acc, results, cnt_list