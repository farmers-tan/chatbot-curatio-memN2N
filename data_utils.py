from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf

stop_words=set(["a","an","the"])


def load_candidates(data_dir, task_id):
    assert task_id > 0 and task_id < 8
    candidates=[]
    candidates_f=None
    candid_dic={}
    if task_id==6:
        candidates_f='dialog-babi-task6-dstc2-candidates.txt'
    else:
        candidates_f='dialog-babi-candidates-2.txt'
    with open(os.path.join(data_dir,candidates_f)) as f:
        for i,line in enumerate(f):
            candid_dic[line.strip().split(' ',1)[1]] = i
            line=tokenize(line.strip())[1:]
            candidates.append(line)
    # return candidates,dict((' '.join(cand),i) for i,cand in enumerate(candidates))
    return candidates,candid_dic


def load_dialog_task(data_dir, task_id, candid_dic, isOOV):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 8

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else: 
        test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    return train_data, test_data, val_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>':
        return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in stop_words]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result


# def parse_dialogs(lines,candid_dic):
#     '''
#         Parse dialogs provided in the babi tasks format
#     '''
#     data=[]
#     context=[]
#     u=None
#     r=None
#     for line in lines:
#         line=str.lower(line.strip())
#         if line:
#             nid, line = line.split(' ', 1)
#             nid = int(nid)
#             if '\t' in line:
#                 u, r = line.split('\t')
#                 u = tokenize(u)
#                 r = tokenize(r)
#                 # temporal encoding, and utterance/response encoding
#                 u.append('$u')
#                 u.append('#'+str(nid))
#                 r.append('$r')
#                 r.append('#'+str(nid))
#                 context.append(u)
#                 context.append(r)
#             else:
#                 r=tokenize(line)
#                 r.append('$r')
#                 r.append('#'+str(nid))
#                 context.append(r)
#         else:
#             context=[x for x in context[:-2] if x]
#             u=u[:-2]
#             r=r[:-2]
#             key=' '.join(r)
#             if key in candid_dic:
#                 r=candid_dic[key]
#                 data.append((context, u,  r))
#             context=[]
#     return data

def parse_dialogs_per_response(lines,candid_dic):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]
    context=[]
    u=None
    r=None
    for line in lines:
        line=line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if '\t' in line:
                u, r = line.split('\t')
                a = candid_dic[r]
                u = tokenize(u)
                r = tokenize(r)
                # temporal encoding, and utterance/response encoding
                # data.append((context[:],u[:],candid_dic[' '.join(r)]))
                data.append((context[:],u[:],a))
                u.append('$u')
                u.append('#'+str(nid))
                r.append('$r')
                r.append('#'+str(nid))
                context.append(u)
                context.append(r)
            else:
                r=tokenize(line)
                r.append('$r')
                r.append('#'+str(nid))
                context.append(r)
        else:
            # clear context
            context=[]
    return data



def get_dialogs(f,candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)

def vectorize_candidates_sparse(candidates,word_idx, vocab, ivocab, word_vector_size):
    shape=(len(candidates),len(word_idx)+1)
    indices=[]
    values=[]
    for i,candidate in enumerate(candidates):
        for w in candidate:
            #indices.append([i,word_idx[w]])
            indices.append([i, process_word(w, word_idx, vocab, ivocab, word_vector_size, to_return="index")])
            values.append(1.0)
    return tf.SparseTensor(indices,values,shape)

def vectorize_candidates(candidates,word_idx,sentence_size, vocab, ivocab, word_vector_size):
    shape=(len(candidates),sentence_size)
    print(shape)
    C=[]

    for i,candidate in enumerate(candidates):
        lc=max(0,sentence_size-len(candidate))
        #C.append([word_idx[w] if w in word_idx else 0 for w in candidate] + [0] * lc)
        C.append([process_word(w, word_idx, vocab, ivocab, word_vector_size, to_return="index") for w in candidate] + [0] * lc)
        #C.append([[process_word(w, word_idx, vocab, ivocab, word_vector_size, to_return="word2vec")] for w in candidate] + [pad] * lc)

    #array_C = np.vstack(np.array(C))
    return tf.constant(C,shape=shape)


# word_idx is just assigning a number to a word. If a new word comes out of the vocabulary it has
# built over training, it will be 0. So new word cannot be well predicted. Max sentence length is 
# kept and 0's are appended whenever the sentence length is less than max sentence length. Means
# you cant ask a query more than max sentence lenght or it may fail
def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size, vocab, ivocab, word_vector_size, uncertain_word = False, uncertain = -1):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    # Do not sort. Keep data as is from reading of files
    # data.sort(key=lambda x:len(x[0]),reverse=True)
    for i, (story, query, answer) in enumerate(data):
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            #ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
            # If story or query is either the unknown response related or its during interactive mode/test
            if answer == 45 or uncertain_word:
                ss.append([process_word(w, word_idx, vocab, ivocab, word_vector_size, to_return="index", uncertain_word = True, uncertain = uncertain) for w in sentence] + [0] * ls)
            else:
                ss.append([process_word(w, word_idx, vocab, ivocab, word_vector_size, to_return="index") for w in sentence] + [0] * ls)
            # inp_vector = [list(process_word(w, word_idx, vocab, ivocab, word_vector_size, to_return="index")) for w in sentence]

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        #q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq
        if answer == 45 or uncertain_word:
            q = [process_word(w, word_idx, vocab, ivocab, word_vector_size, to_return="index", uncertain_word = True, uncertain = uncertain) for w in query] + [0] * lq
        else:
            q = [process_word(w, word_idx, vocab, ivocab, word_vector_size, to_return="index") for w in query] + [0] * lq

        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(answer))
    return S, Q, A


def load_glove(dim):
    word2vec = {}
    
    print("==> loading glove")
    with open(("./data/glove/glove.6B." + str(dim) + "d.txt"), encoding="utf8") as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = list(map(float, l[1:]))
            
    print("==> glove is loaded")
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector

def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True, uncertain_word=False, uncertain = -1):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        # During training, only after training in-context, we go to out of context and any new word
        # out of vocab is treated as the uncertain word defined before
        # In interactive testing, an unknown word not in vocab is that unknown word learned from train set
        if uncertain_word:
            vocab[word] = uncertain
            word2vec[word] = word2vec[ivocab[uncertain]]
        else:
            next_index = len(ivocab)
            vocab[word] = next_index
            ivocab[next_index] = word

    vec = list(word2vec[word])
    if vec == []:
        create_vector(word, word2vec, word_vector_size, silent)
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = list(word2vec[word])
    return embedding