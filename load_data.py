import json
import torch
import os
import json
from matplotlib.pylab import *
# def get_data(file_path: str,batch_size: int):
#     '''
#     Gets data from the dataset and creates a data loader

#     Arguments:
#     file_path: The path to the directory in which the dataset is contained
#     batch_size: Batch size to be used for the data loaders

#     Returns:
#     train_data: Training dataset (Natural Language utterances)
#     train_table: Training tables (Table schema and table data) 
#     dev_data: Development dataset (Natural Language utterances) 
#     dev_table: Development tables (Table schema and table data) 
#     train_loader: Training dataset loader
#     dev_loader:  Development dataset loader
#     '''
#     # Loading Dev Files(Development Dataset)
#     dev_data = []
#     dev_table = {}

#     with open(file_path + '/dev_knowledge.jsonl') as dev_data_file:
#         for idx, line in enumerate(dev_data_file):
#             current_line = json.loads(line.strip())
#             dev_data.append(current_line)

#     with open(file_path + '/dev.tables.jsonl') as dev_table_file:
#         for idx, line in enumerate(dev_table_file):
#             current_line = json.loads(line.strip())
#             dev_table[current_line['id']] = current_line
    
#     # Loading Train Files(Training Dataset)
#     train_data = []
#     train_table = {}

#     with open(file_path + '/train_knowledge.jsonl') as train_data_file:
#         for idx, line in enumerate(train_data_file):
#             current_line = json.loads(line.strip())
#             train_data.append(current_line)

#     with open(file_path + '/train.tables.jsonl') as train_table_file:
#         for idx, line in enumerate(train_table_file):
#             current_line = json.loads(line.strip())
#             train_table[current_line['id']] = current_line

#     train_loader = torch.utils.data.DataLoader(
#         batch_size=batch_size,
#         dataset=train_data,
#         shuffle=True,
#         num_workers=4,
#         collate_fn=lambda x: x  # now dictionary values are not merged!
#     )

#     dev_loader = torch.utils.data.DataLoader(
#         batch_size=batch_size,
#         dataset=dev_data,
#         shuffle=True,
#         num_workers=4,
#         collate_fn=lambda x: x  # now dictionary values are not merged!
#     )

#     return train_data, train_table, dev_data, dev_table, train_loader, dev_loader

# def get_fields(tables_data, input_tables, header_tokenization=True, sql_tokenization=True):

#     natural_language_utterance = []
#     tokenized_natural_language_utterance = []
#     sql_indexing = []
#     sql_query = []
#     tokenized_sql_query = []
#     table_indices = []
#     tokenized_headers = []
#     headers = []

#     for table_data in tables_data:
#         natural_language_utterance.append(table_data['question'])
#         tokenized_natural_language_utterance.append(table_data['question_tok'])
#         sql_indexing.append(table_data['sql'])
#         sql_query.append(table_data['query'])
#         headers.append(input_tables[table_data['table_id']]['header'])
#         table_indices.append(table_data['table_id'])

#         if sql_tokenization:
#             tokenized_sql_query.append(table_data['query_tok'])
#         else:
#             tokenized_sql_query.append(None)
        
#         if header_tokenization:
#             tokenized_headers.append(input_tables[table_data['table_id']]['header_tok'])
#         else:
#             tokenized_headers.append([])
        
#     return natural_language_utterance,tokenized_natural_language_utterance,sql_indexing,sql_query,tokenized_sql_query,table_indices,tokenized_headers,headers


def get_data(path_wikisql,batch_size=8):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, False, 12,
                                                                      no_w2i=True, no_hs_tok=False)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, batch_size, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader

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

    path_sql = os.path.join(path_wikisql, mode+'_knowledge.jsonl')
    if no_hs_tok:
        path_table = os.path.join(path_wikisql, mode + '.tables.jsonl')
    else:
        path_table = os.path.join(path_wikisql, mode+'_tok.tables.jsonl')

    data = []
    table = {}
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            if toy_model and idx >= toy_size:
                break

            t1 = json.loads(line.strip())
            data.append(t1)

    with open(path_table) as f:
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

def get_fields_1(t1, tables, no_hs_t=False, no_sql_t=False):
    nlu1 = t1['question']
    nlu_t1 = t1['question_tok']
    tid1 = t1['table_id']
    sql_i1 = t1['sql']
    sql_q1 = t1['query']
    if no_sql_t:
        sql_t1 = None
    else:
        sql_t1 = t1['query_tok']
    tb1 = tables[tid1]
    if not no_hs_t:
        hs_t1 = tb1['header_tok']
    else:
        hs_t1 = []
    hs1 = tb1['header']

    return nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1

def get_fields(t1s, tables, no_hs_t=False, no_sql_t=False):

    nlu, nlu_t, tid, sql_i, sql_q, sql_t, tb, hs_t, hs = [], [], [], [], [], [], [], [], []
    for t1 in t1s:
        if no_hs_t:
            nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t)
        else:
            nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t)

        nlu.append(nlu1)
        nlu_t.append(nlu_t1)
        tid.append(tid1)
        sql_i.append(sql_i1)
        sql_q.append(sql_q1)
        sql_t.append(sql_t1)

        tb.append(tb1)
        hs_t.append(hs_t1)
        hs.append(hs1)

    return nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hs
