import json
import torch

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



