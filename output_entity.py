import sys
sys.path.append(r"C:\Users\debad\NL2SQL-RULE")

import bert.tokenization as tokenization
from sqlnet.dbengine import DBEngine
import json
import torch
import os
import torch.utils.data
from matplotlib.pylab import *
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
config = {}
config["batch_size"] = 8
config["data_path"] = "../data/"
config["num_target_layers"] = 2
config["dropout"] = 0.3
config["max_seq_length"] = 222
config["toy_model"] = False
config["toy_size"] = 12
config["accumulate_gradients"] = 2
config["EG"] = False


def get_loader_wikisql(data_train, data_dev, bS, shuffle_train=True, shuffle_dev=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_dev,
        shuffle=shuffle_dev,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader

def load_wikisql_data(path_wikisql, mode='train', toy_model=False, toy_size=10, no_hs_tok=False, aug=False):
    """ Load training sets
    """
    if aug:
        mode = f"aug.{mode}"
        print('Augmented data is loaded!')

    path_sql = os.path.join(path_wikisql, mode+'.jsonl')
    if no_hs_tok:
        path_table = os.path.join(path_wikisql, mode + '.tables.jsonl')
    else:
        path_table = os.path.join(path_wikisql, mode+'_tok.tables.jsonl')

    data = []
    table = {}
    with open(path_sql,mode="r",encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_model and idx >= toy_size:
                break

            t1 = json.loads(line.strip())
            data.append(t1)

    with open(path_table,mode="r",encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_model and idx > toy_size:
                break

            t1 = json.loads(line.strip())
            table[t1['id']] = t1

    return data, table


def load_w2i_wemb(path_wikisql, bert=False):
    """ Load pre-made subset of TAPI.
    """
    if bert:
        with open(os.path.join(path_wikisql, 'w2i_bert.json'), 'r') as f_w2i:
            w2i = json.load(f_w2i)
        wemb = load(os.path.join(path_wikisql, 'wemb_bert.npy'), )
    else:
        with open(os.path.join(path_wikisql, 'w2i.json'), 'r') as f_w2i:
            w2i = json.load(f_w2i)

        wemb = load(os.path.join(path_wikisql, 'wemb.npy'), )
    return w2i, wemb

# Load data -----------------------------------------------------------------------------------------------
def load_wikisql(path_wikisql, toy_model, toy_size, bert=False, no_w2i=False, no_hs_tok=False, aug=False):
    # Get data
    train_data, train_table = load_wikisql_data(path_wikisql, mode='train', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok, aug=aug)
    dev_data, dev_table = load_wikisql_data(path_wikisql, mode='dev', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok)


    # Get word vector
    if no_w2i:
        w2i, wemb = None, None
    else:
        w2i, wemb = load_w2i_wemb(path_wikisql, bert)


    return train_data, train_table, dev_data, dev_table, w2i, wemb

def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql,
                                                                      args["toy_model"],
                                                                      args["toy_size"],
                                                                      no_w2i=True,
                                                                      no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args["batch_size"], shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader

# def contains2(small_str,big_str):
#     if small_str in big_str:
#         start = big_str.index(small_str)
#         return True,start,start+len(small_str)-1
#     else:
#         return False,-1,-1

# def contains(small_list, big_list):
#     for i in range(len(big_list)-len(small_list)+1):
#         if big_list[i:i+len(small_list)]==small_list:
#             return True,i, i+len(small_list)
#     return False,-1,-1

# def contains(small_list,big_list):
#     result = False
#     for i,item in enumerate(big_list):
#         if item == small_list[0]:
#             result = True
#             if i+len(small_list)>len(big_list):
#                 result = False
#                 break
#             for ii in range(0,len(small_list)):
#                 if small_list[ii] != big_list[i+ii]:
#                     result=False
#                     break
#                 if ii == len(small_list)-1:
#                     return result,i,i+ii
#     return result,-1,-1

import re
re_ = re.compile(' ')
#engine_train = DBEngine("train.db")
#engine_dev = DBEngine("dev.db")
train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data("./", config)
count = 0
count_agg_0 = 0
count_agg_not_0 = 0
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def process(data,table,output_name):
  final_all = []
  badcase = 0
 
  for i, one_data in enumerate(data):

    nlu = one_data['question']
    nlu_t = tokenizer.encode(nlu.lower(), add_prefix_space=True)
    # if i<=368:
    #     continue
    # nlu_t1 = one_data["question_tok"]
    # nlu_tt2 = tokenizer.tokenize(one_data["question"])

    # 1. 2nd tokenization using WordPiece
    # charindex2wordindex = {}
    # total = 0
    # tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
    # t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
    # nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
    # for (ii, token) in enumerate(nlu_t1):
    #     t_to_tt_idx1.append(
    #         len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
    #     sub_tokens = tokenizer.tokenize(token)
    #     for sub_token in sub_tokens:
    #         tt_to_t_idx1.append(ii)
    #         nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer

    #     token_ = re_.sub('',token)
    #     for iii in range(len(token_)):
    #         charindex2wordindex[total+iii]=ii
    #     total += len(token_)

    one_final = one_data
    one_final['query'] = one_data['sql']
    one_final['roberta_enc'] = nlu_t
    one_table = table[one_data["table_id"]]
    one_final["header"] = one_table["header"]
    one_final["types"] = one_table["types"]
    final_question = [0] * len(nlu_t)
    one_final["bertindex_knowledge"] = final_question
    final_header = [0] * len(one_table["header"])
    one_final["header_knowledge"] = final_header

    header_enc = [tokenizer.encode(h.lower(), add_prefix_space=True)[1:-1] for h in one_table["header"]]

    for ii, nlu_enc in enumerate(nlu_t):
        for h_enc in header_enc:
            if nlu_enc in h_enc:
                final_question[ii] = 4

    one_final['bertindex_knowledge'] = final_question

    for ii, h_enc in enumerate(header_enc):
        temp = 0
        for enc in h_enc:
            if enc in nlu_t:
                temp += 1
        final_header[ii] = temp/len(h_enc)

    one_final['header_knowledge'] = final_header
    
    # for i,h_enc in header_enc:
    #     is_sublist,start_index,end_index = contains(h_enc,nlu_t)
    #     if is_sublist:
    #         final_question[start_index:end_index] = [4]*(end_index-start_index)
    #         final_header[i]=1
        
    # one_final['bertindex_knowledge'] = final_question
    # one_final['header_knowledge'] = final_header

    # for ii,h in enumerate(one_table["header"]):
    #     h = h.lower()
    #     hs = h.split("/")
    #     for h_ in hs:
    #         flag, start_, end_ = contains2(re_.sub('', h_), "".join(one_data["question_tok"]).lower())
    #         if flag == True:
    #             try:
    #                 start = t_to_tt_idx1[charindex2wordindex[start_]]
    #                 end = t_to_tt_idx1[charindex2wordindex[end_]]
    #                 for iii in range(start,end):
    #                     final_question[iii] = 4
    #                 final_question[start] = 4
    #                 final_question[end] = 4
    #                 one_final["bertindex_knowledge"] = final_question
    #             except:
    #                 print("!!!!!")
    #                 continue

    # for ii,h in enumerate(one_table["header"]):
    #     h = h.lower()
    #     hs = h.split("/")
    #     for h_ in hs:
    #         flag, start_, end_ = contains2(re_.sub('', h_), "".join(one_data["question_tok"]).lower())
    #         if flag == True:
    #             try:
    #                 final_header[ii] = 1
    #                 break
    #             except:
    #                 print("!!!!")
    #                 continue

    # one_final["header_knowledge"] = final_header
    

    if i%1000==0:
        print(i)
    if "bertindex_knowledge" not in one_final and len(one_final["sql"]["conds"])>0:
        print(one_data["question"])
        print(one_table["rows"])
        one_final["bertindex_knowledge"] = [0] * len(nlu_t)
        badcase+=1
    final_all.append(one_final)
  print(badcase)
  f = open(output_name,mode="w",encoding="utf-8")
  import json
  for line in final_all:
    json.dump(line, f)
    f.write('\n')
  f.close()

process(train_data,train_table,"train_knowledge.jsonl")
process(dev_data,dev_table,"dev_knowledge.jsonl")



