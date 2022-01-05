import pandas as pd

##########################################
#    Obtain data functions
##########################################


def get_data_lp(dataset):
    """
    Get the entity file and literal file for Link Prediction
    """
    if dataset == "WikidataDWD":
        entities = pd.read_csv(f'data/{dataset}/train.txt', sep='\t', usecols=[0, 1, 2])
    else:
        entities = pd.read_csv(f'data/{dataset}/train.txt', sep='\t', header=None, usecols=[0, 1, 2])
    entities.columns = ['node1', 'label', 'node2']

    df = pd.read_csv(f'data/{dataset}/numerical_literals.txt', sep='\t', header=None, usecols=[0, 1, 2])
    df.columns = ['node1', 'label', 'node2']
    df = df[df['node2'].notnull()]
    df = df.reset_index(drop=True)

    return entities, df


def clean_entities(df, dataset):
    if dataset.lower() == "yago15k":
        df[0] = df[0].apply(lambda x: x.split("resource")[-1][1:-1])
        df[1] = df[1].apply(lambda x: x.split("resource")[-1][1:-1])
        df[2] = df[2].apply(lambda x: x.split("resource")[-1][1:-1])
    df.columns = ['node1', 'label', 'node2']
    return df


def clean_numeric(df, dataset):
    if dataset.lower() == "yago15k":
        df[0] = df[0].apply(lambda x: x.split("resource")[-1][1:-1])
        df[1] = df[1].apply(lambda x: x.split("resource")[-1][1:-1])
    elif dataset.lower() == "fb15k237":
        df[1] = df[1].apply(lambda x: x.split("ns")[-1][1:-1])
    df.columns = ['node1', 'label', 'node2']
    df = df[df['node2'].notnull()]
    df = df.reset_index(drop=True)
    return df


def get_data_np(dataset):
    """ Get the entity file and literal file for """
    entities = clean_entities(pd.read_csv(f'data/{dataset}/train_kge', sep='\t', header=None), dataset)
    train = clean_numeric(pd.read_csv(f'data/{dataset}/train_100', sep='\t', header=None), dataset)
    valid = clean_numeric(pd.read_csv(f'data/{dataset}/dev', sep='\t', header=None), dataset)
    test = clean_numeric(pd.read_csv(f'data/{dataset}/test', sep='\t', header=None), dataset)
    return entities, train, valid, test
