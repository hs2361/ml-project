import torch
import os
from seq2sql_model_classes import Seq2SQL_v1
from bert_model_classes import BertConfig
from tokenizer_classes import FullTokenizer
from bert_model_classes import BertModel

def get_bert_model(model_path, bert_model_type = 'uncased_L-12_H-768_A-12', no_pretraining = False,load_pretrained_model = False):
    '''
    get_bert_model
    Arguments:
    model_path: The path to the directory in which the model is contained
    bert_model_type : As name suggest pass bert model type as 'uncased_L-12_H-768_A-12'
    no_pretraining : enter true or false in case of pretrained or not
    load_pretrained_model : want to load pretrained model(true or false)

    Returns:
    model_bert: returns bert model
    bert_tokenizer: returns tokenizer of bert model
    bert_config: returns configuration of bert model

    '''
    #bert_model_types:'uncased_L-12_H-768_A-12',
                    # 'uncased_L-24_H-1024_A-16',
                    # 'cased_L-12_H-768_A-12',
                    #  'cased_L-24_H-1024_A-16',
                    #  'multi_cased_L-12_H-768_A-12'

    # If bert model is cased we have to uncase it i.e. convert all stuff into lowercase
    if bert_model_type == 'cased_L-12_H-768_A-12' or bert_model_type == 'cased_L-24_H-1024_A-16' or bert_model_type == 'multi_cased_L-12_H-768_A-12':
        convert_to_lower_case = False
    else:
        convert_to_lower_case = True

    # File path of general configuration files
    bert_config_file = os.path.join(model_path, f'bert_config_{bert_model_type}.json')
    vocab_file = os.path.join(model_path, f'vocab_{bert_model_type}.txt')
    initial_checkpoint = os.path.join(model_path, f'pytorch_model_{bert_model_type}.bin')

    # Reading the configuration File
    bert_config = BertConfig.from_json_file(bert_config_file)
    bert_config.print_status() #Comment out if we do't want extra lines printed out

    # Loading a BERT model according to configuration file 
    model_bert = BertModel(bert_config)

    # Building the Tokenizer
    bert_tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=convert_to_lower_case)

    # If we don't want to do pretraining
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(initial_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_bert

    # If we have to load a already trained model
    if load_pretrained_model:
        assert model_path != None

        if torch.cuda.is_available():
            res = torch.load(model_path)
        else:
            res = torch.load(model_path, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert

    return model_bert, bert_tokenizer, bert_config


def get_seq2sql_model(bert_hidden_layer_size, number_of_layers = 2,
                    hidden_vector_dimensions = 100,
                    number_lstm_layers = 2,
                    dropout_rate = 0.3,
                    load_pretrained_model=False, model_path=None):
    
    '''
    
    get_seq2sql_model
    Arguments:
    bert_hidden_layer_size: sizes of hidden layers of bert model
    number_of_layers : total number of layers
    hidden_vector_dimensions : dimensions of hidden vectors
    number_lstm_layers : total number of lstm layers
    dropout_rate : value of dropout rate
    load_pretrained_model : want to load pretrained model(true or false)
    model_path : The path to the directory in which the model is contained
    
    Returns:
    model: returns the model
    
    '''

    # number_of_layers = "The Number of final layers of BERT to be used in downstream task."
    # hidden_vector_dimensions : "The dimension of hidden vector in the seq-to-SQL module."
    # number_lstm_layers : "The number of LSTM layers." in seqtosqlmodule

    sql_main_operators = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    sql_conditional_operators = ['=', '>', '<', 'OP']

    number_of_neurons = bert_hidden_layer_size * number_of_layers  # Seq-to-SQL input vector dimenstion

    model = Seq2SQL_v1(iS = number_of_neurons, hS= hidden_vector_dimensions,lS= number_lstm_layers, dr= dropout_rate, n_cond_ops=4,  n_agg_ops=6)
    model = model

    if load_pretrained_model:
        assert model_path != None
        if torch.cuda.is_available():
            res = torch.load(model_path)
        else:
            res = torch.load(model_path, map_location='cpu')
        model.load_state_dict(res['model'])

    return model

def get_optimizers(model, model_bert, fine_tune =False,learning_rate_model=1e-3,learning_rate_bert=1e-5):
    '''
    get_optimizers
    Arguments:
    model: returned model from get_seq2sql_model
    model_bert : returned model from get_bert_model
    fine_tune : want to fine tune(true or false)
    learning_rate_model : learning rate of model (from get_seq2sql_model)
    learning_rate_bert : learning rate of bert model (from get_bert_model)
    
    Returns:
    opt: returns the optimised model (from get_seq2sql_model)
    opt_bert : returns the optimised bert model (from get_bert_model)

    '''


    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=learning_rate_model, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=learning_rate_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=learning_rate_model, weight_decay=0)
        opt_bert = None

    return opt, opt_bert
