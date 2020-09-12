import json
import torch
import os
import json
from matplotlib.pylab import *

def get_data(file_path: str,batch_size: int):
    '''
    Gets data from the dataset and creates a data loader

    Arguments:
    file_path: The path to the directory in which the dataset is contained
    batch_size: Batch size to be used for the data loaders

    Returns:
    train_data: Training dataset (Natural Language utterances)
    train_table: Training tables (Table schema and table data) 
    dev_data: Development dataset (Natural Language utterances) 
    dev_table: Development tables (Table schema and table data) 
    train_loader: Training dataset loader
    dev_loader:  Development dataset loader
    '''
    # Loading Dev Files(Development Dataset)
    dev_data = []
    dev_table = {}

    with open(file_path + '/dev_knowledge.jsonl') as dev_data_file:
        for idx, line in enumerate(dev_data_file):
            current_line = json.loads(line.strip())
            dev_data.append(current_line)

    with open(file_path + '/dev.tables.jsonl') as dev_table_file:
        for idx, line in enumerate(dev_table_file):
            current_line = json.loads(line.strip())
            dev_table[current_line['id']] = current_line
    
    # Loading Train Files(Training Dataset)
    train_data = []
    train_table = {}

    with open(file_path + '/train_knowledge.jsonl') as train_data_file:
        for idx, line in enumerate(train_data_file):
            current_line = json.loads(line.strip())
            train_data.append(current_line)

    with open(file_path + '/train.tables.jsonl') as train_table_file:
        for idx, line in enumerate(train_table_file):
            current_line = json.loads(line.strip())
            train_table[current_line['id']] = current_line

    train_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=train_data,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=dev_data,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader

def get_fields(data, header_tokenization=False, sql_tokenization=False):

    natural_language_utterance = []
    nlu_roberta_encoding = []
    sql_indexing = []
    sql_query = []
    tokenized_sql_query = []
    table_indices = []
    tokenized_headers = []
    headers = []

    for one_data in data:
        natural_language_utterance.append(one_data['question'])
        nlu_roberta_encoding.append(one_data['roberta_enc'])
        sql_indexing.append(one_data['sql'])
        sql_query.append(one_data['query'])
        headers.append(one_data['header'])
        table_indices.append({
            "id" : one_data["table_id"],
            "header": one_data["header"],
            "types" : one_data["types"]
        })

        if sql_tokenization:
            tokenized_sql_query.append(one_data['query_tok'])
        else:
            tokenized_sql_query.append(None)
        
        if header_tokenization:
            tokenized_headers.append(one_data['header_tok'])
        else:
            tokenized_headers.append(None)
        
    return natural_language_utterance,nlu_roberta_encoding,sql_indexing,sql_query,tokenized_sql_query,table_indices,tokenized_headers,headers

# def get_fields_1(t1, tables, no_hs_t=False, no_sql_t=False):
#     nlu1 = t1['question']
#     nlu_t1 = t1['question_tok']
#     tid1 = t1['table_id']
#     sql_i1 = t1['sql']
#     sql_q1 = t1['query']
#     if no_sql_t:
#         sql_t1 = None
#     else:
#         sql_t1 = t1['query_tok']
#     tb1 = tables[tid1]
#     if not no_hs_t:
#         hs_t1 = tb1['header_tok']
#     else:
#         hs_t1 = []
#     hs1 = tb1['header']

#     return nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1

# def get_fields(t1s, tables, no_hs_t=False, no_sql_t=False):

#     nlu, nlu_t, tid, sql_i, sql_q, sql_t, tb, hs_t, hs = [], [], [], [], [], [], [], [], []
#     for t1 in t1s:
#         if no_hs_t:
#             nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t)
#         else:
#             nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t)

#         nlu.append(nlu1)
#         nlu_t.append(nlu_t1)
#         tid.append(tid1)
#         sql_i.append(sql_i1)
#         sql_q.append(sql_q1)
#         sql_t.append(sql_t1)

#         tb.append(tb1)
#         hs_t.append(hs_t1)
#         hs.append(hs1)

#     return nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hs
