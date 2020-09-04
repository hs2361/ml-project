from matplotlib.pylab import *


def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1).sentence:
        for tok in sentence.token:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok

def sent_split(documents):
    words = []
    for sent in sent_tokenize(documents):
      for word in word_tokenize(sent):
        words.append(word)
    return words

def load_jsonl(path_file, toy_data=False, toy_size=4, shuffle=False, seed=1):
    data = []

    with open(path_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_data and idx >= toy_size and (not shuffle):
                break
            t1 = json.loads(line.strip())
            data.append(t1)

    if shuffle and toy_data:
        # When shuffle required, get all the data, shuffle, and get the part of data.
        print(
            f"If the toy-data is used, the whole data loaded first and then shuffled before get the first {toy_size} data")

        python_random.Random(seed).shuffle(data)  # fixed
        data = data[:toy_size]

    return data

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
