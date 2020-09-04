from matplotlib.pylab import *
import re
import os
re_ = re.compile(' ')

def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1).sentence:
        for tok in sentence.token:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok

def sort_and_generate_pr_w(pr_sql_i):
    pr_wc = []
    pr_wo = []
    pr_wv = []
    for b, pr_sql_i1 in enumerate(pr_sql_i):
        conds1 = pr_sql_i1["conds"]
        pr_wc1 = []
        pr_wo1 = []
        pr_wv1 = []

        # Generate
        for i_wn, conds11 in enumerate(conds1):
            pr_wc1.append( conds11[0])
            pr_wo1.append( conds11[1])
            pr_wv1.append( conds11[2])

        # sort based on pr_wc1
        idx = argsort(pr_wc1)
        pr_wc1 = array(pr_wc1)[idx].tolist()
        pr_wo1 = array(pr_wo1)[idx].tolist()
        pr_wv1 = array(pr_wv1)[idx].tolist()

        conds1_sorted = []
        for i, idx1 in enumerate(idx):
            conds1_sorted.append( conds1[idx1] )


        pr_wc.append(pr_wc1)
        pr_wo.append(pr_wo1)
        pr_wv.append(pr_wv1)

        pr_sql_i1['conds'] = conds1_sorted

    return pr_wc, pr_wo, pr_wv, pr_sql_i

def process(data,table,model_path,tokenize,bert_model_type='uncased_L-12_H-768_A-12'):
    vocab_file = os.path.join(model_path, f'vocab_{bert_model_type}.txt')
    final_all = []
    badcase = 0
    for i, one_data in enumerate(data):
        nlu_t1 = one_data["question_tok"]

        # 1. 2nd tokenization using WordPiece
        charindex2wordindex = {}
        total = 0
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (ii, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenize(token)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(ii)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer

            token_ = re_.sub('',token)
            for iii in range(len(token_)):
                charindex2wordindex[total+iii]=ii
            total += len(token_)

        one_final = one_data
        one_table = table[one_data["table_id"]]
        final_question = [0] * len(nlu_tt1)
        one_final["bertindex_knowledge"] = final_question
        final_header = [0] * len(one_table["header"])
        one_final["header_knowledge"] = final_header
        for ii,h in enumerate(one_table["header"]):
            h = h.lower()
            hs = h.split("/")
            for h_ in hs:
                flag, start_, end_ = contains2(re_.sub('', h_), "".join(one_data["question_tok"]).lower())
                if flag == True:
                    try:
                        start = t_to_tt_idx1[charindex2wordindex[start_]]
                        end = t_to_tt_idx1[charindex2wordindex[end_]]
                        for iii in range(start,end):
                            final_question[iii] = 4
                        final_question[start] = 4
                        final_question[end] = 4
                        one_final["bertindex_knowledge"] = final_question
                    except:
                        # print("!!!!!")
                        continue

        for ii,h in enumerate(one_table["header"]):
            h = h.lower()
            hs = h.split("/")
            for h_ in hs:
                flag, start_, end_ = contains2(re_.sub('', h_), "".join(one_data["question_tok"]).lower())
                if flag == True:
                    try:
                        final_header[ii] = 1
                        break
                    except:
                        # print("!!!!")
                        continue

        for row in one_table["rows"]:
            for iiii, cell in enumerate(row):
                cell = str(cell).lower()
                flag, start_, end_ = contains2(re_.sub('', cell), "".join(one_data["question_tok"]).lower())
                if flag == True:
                    final_header[iiii] = 2

        one_final["header_knowledge"] = final_header

        for row in one_table["rows"]:
            for cell in row:
                cell = str(cell).lower()
                # cell = cell.replace('"',"")
                cell_tokens = tokenize(cell)



                if len(cell_tokens)==0:
                    continue

                flag, start_, end_ = contains2(re_.sub('', cell),  "".join(one_data["question_tok"]).lower())
                # flag, start, end = contains(cell_tokens, nlu_tt1)
                # if flag==False:
                #     flag, start, end = contains(cell_tokens, nlu_tt2)
                #     if len(nlu_tt1) != len(nlu_tt2):
                #         continue
                if flag == True:
                    try:
                        start = t_to_tt_idx1[charindex2wordindex[start_]]
                        end = t_to_tt_idx1[charindex2wordindex[end_]]
                        for ii in range(start,end):
                            final_question[ii] = 2
                        final_question[start] = 1
                        final_question[end] = 3
                        one_final["bertindex_knowledge"] = final_question
                        break
                    except:
                        # print("!!!")
                        continue
        # if i%1000==0:
        #     print(i)
        if "bertindex_knowledge" not in one_final and len(one_final["sql"]["conds"])>0:
            # print(one_data["question"])
            # print(one_table["rows"])
            one_final["bertindex_knowledge"] = [0] * len(nlu_tt1)
            badcase+=1
        final_all.append([one_data["question_tok"],one_final["bertindex_knowledge"],one_final["header_knowledge"]])
        # print(badcase)
    return final_all
  

def contains2(small_str,big_str):
    if small_str in big_str:
        start = big_str.index(small_str)
        return True,start,start+len(small_str)-1
    else:
        return False,-1,-1
