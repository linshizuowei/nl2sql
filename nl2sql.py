# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import random
import time
import re
from torch.autograd import Variable




op_sql_dict = {0:">", 1:"<", 2:"==", 3:"!="}
agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
conn_sql_dict = {0:"", 1:" and ", 2:" or "}
SOS_TOKEN = u'\ude96'
EOS_TOKEN = u'\udf6c'
UNK_TOKEN = u'\udc4a'
TAB_TOKEN = u'\udc0d'
MAX_LEN = 114
TAR_LEN = 35


def extract_table():
    table_code = {}
    headers = {}
    code = 0
    for tfile in ['train_data/train.tables.json', 'val_data/val.tables.json', 'test_data/test.tables.json']:
        with open(tfile) as fd:
            for n, line in enumerate(fd):
                data = json.loads(line)
                if data['id'] not in table_code:
                    bi = bin(code)[2:]
                    co = '0' * (300 - len(bi)) + bi
                    table_code[data['id']] = co
                    code += 1
                if data['id'] not in headers:
                    headers[data['id']] = data['header']
    return table_code, headers

def load_word_vector():
    word_vector = {}
    word2index = {}
    index2word = {}
    with open('origin_data/nl2sql_char_embedding_baseline') as fd:
        for n, line in enumerate(fd):
            line = line.decode('utf-8').strip().split(' ')
            vec = [float(x) for x in line[1:]]
            word_vector[line[0]] = vec
            index2word[n] = line[0]
            word2index[line[0]] = n
    return word_vector, word2index, index2word


def construct_output_str(data, dtype):
    res = []
    header = table_headers[data['table_id']]
    if dtype == 'agg':
        agg = data['sql']['agg']
        for i in agg:
            res.append(agg_sql_dict[i])
        return res
    elif dtype == 'sel':
        sel = data['sql']['sel']
        for i in sel:
            res.append(header[i])
    elif dtype == 'conds':
        conds = data['sql']['conds']
        for it in conds:
            res.append([header[it[0]], op_sql_dict[it[1]], it[2]])
    elif dtype == 'cond_conn_op':
        res.append(conn_sql_dict[data['sql']['cond_conn_op']])
    return res
 
def construct_output_vector(str_list):
    vector = []
    target_index = []
    for it in str_list:
        if isinstance(it, list):
            for s in it:
                for c in s:
                    vector.append(word_vector.get(c, word_vector[UNK_TOKEN]))
                    target_index.append(word2index.get(c, word2index[UNK_TOKEN]))
        else:
            for c in it:
                vector.append(word_vector.get(c, word_vector[UNK_TOKEN]))
                target_index.append(word2index.get(c, word2index[UNK_TOKEN]))
        vector.append(word_vector[TAB_TOKEN])
        target_index.append(word2index.get(c, word2index[UNK_TOKEN]))
    vector = vector[:-1] + [word_vector[EOS_TOKEN]]
    out_len = len(vector)
    target_index = target_index[:-1] + [word2index[EOS_TOKEN]]
    vector = vector + [word_vector[TAB_TOKEN] for _ in range(TAR_LEN-len(vector))]
    target_index = target_index + [word2index[TAB_TOKEN] for _ in range(TAR_LEN-len(target_index))]
    return vector, target_index, out_len

def construct_output_sql(data):

    sel = data['sql']['sel']
    agg = data['sql']['agg']
    op = data['sql']['cond_conn_op']
    conds = data['sql']['conds']

    ## construct elements selected
    lsel = []
    for i in range(len(sel)):
        if agg[i] != 0:
            lsel.append('%s(col_%s)' % (agg[i], sel[i]+1))
        else:
            lsel.append('col_%s' % sel[i]+1)

    ## construct conditions
    lconds = []
    for it in conds:
        col = it[0]+1
        cond = op_sql_dict[it[1]]
        lconds.append('col_%s%s%s' % (col, cond, it[2]))

    sql_str = "select %s from Table_%s where %s;" % (','.join(lsel), data['table_id'], conn_sql_dict[op].join(lconds))
    return sql_str

def construct_input_str(data):
    ques = re.sub(r'\s', '', data['question'])
    return '%s##%s' % (data['table_id'], ques)

def construct_input_vector(in_str):
    table, ques = in_str.split('##')
    vector = []
    code = []
    for i in table_code[table]:
        code.append(int(i))
    vector.append(code)

    for c in ques:
        vector.append(word_vector.get(c, word_vector[UNK_TOKEN]))

    vector.append(word_vector[EOS_TOKEN])
    input_len = len(vector)
    vector = vector + [word_vector[TAB_TOKEN] for _ in range(MAX_LEN-len(vector))]

    return vector, input_len


def prepare_data(dtype, etype):
    fpre = dtype
    fpath = '%s_data/%s.json' % (fpre, fpre)

    pairs = []
    indexs = []
    with open(fpath) as fd:
        for line in fd:
            data = json.loads(line)
            in_str = construct_input_str(data)
            out_str = construct_output_str(data, etype)
            pairs.append([in_str, out_str])
    if dtype == 'train':
        random.shuffle(pairs)
    return pairs

def prepare_val_data(etype):
    fpath = 'val_data/val.json'
    inputs = []
    outputs = []
    indexs = []
    inlen = []
    outlen = []
    strs = []
    with open(fpath) as fd:
        for line in fd:
            data = json.loads(line)
            in_str = construct_input_str(data)
            out_str = construct_output_str(data, etype)
            strs.append([in_str, out_str])


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, inputs, input_lengths, hidden):
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, input_lengths)
        output, hidden = self.gru(packed, hidden)
        output, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.relu(inputs)
        output, hidden = self.gru(output, hidden)
        output = self.linear(output[0])
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class AttnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attn_size = input_size + hidden_size
        
        self.drop = nn.Dropout()
        self.attn = nn.Linear(self.attn_size, MAX_LEN)
        self.attn_combine = nn.Linear(self.attn_size, input_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden, encoder_outputs, mask):
        inputs = self.drop(inputs)
#        attn_weights = F.softmax(self.attn(torch.cat((inputs[0], hidden[0]), 1)), dim=1)
#        attn_weights = attn_weights.masked_fill(mask==0, 1e-10)
        attn_weights = self.attn(torch.cat((inputs[0], hidden[0]), 1))
        attn_weights = attn_weights.masked_fill(mask==0, -1e2)
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_applied = (torch.bmm((attn_weights.unsqueeze(0)).transpose(0, 1), encoder_outputs.transpose(0, 1))).transpose(0, 1)
        inputs = self.attn_combine(torch.cat((inputs[0], attn_applied[0]), 1)).unsqueeze(0)
        output = self.relu(inputs)
        output, hidden = self.gru(output, hidden)
#        output, hidden = self.gru(inputs, attn_applied)
        output = self.linear(output[0])
        output = self.softmax(output)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.relu(inputs)
        output, hidden = self.lstm(output, hidden)
        output = self.linear(output[0])
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))


def train(train_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    in_pair = [construct_input_vector(pair[0]) for pair in train_pairs]
    input_tensor = tensorFromVec([pair[0] for pair in in_pair])
    input_len = torch.tensor([pair[1] for pair in in_pair])
    tar_pair = [construct_output_vector(pair[1]) for pair in train_pairs]
    target_tensor = tensorFromVec([pair[0] for pair in tar_pair])
    target_index = torch.tensor([pair[1] for pair in tar_pair])
    target_len = [pair[2] for pair in tar_pair]
    mask = create_mask(input_tensor)

    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_outputs = torch.zeros(MAX_LEN, batch_size, encoder.hidden_size, device=device)
    decoder_outputs = torch.zeros(TAR_LEN, batch_size, output_size, device=device)
    loss = 0

    encoder_output, encoder_hidden = encoder(input_tensor, input_len, encoder_hidden)
    for i in range(encoder_output.size()[0]):
        encoder_outputs[i] = encoder_output[i]

    decoder_input = torch.tensor([[word_vector[SOS_TOKEN]] for _ in range(batch_size)], device=device).transpose(0, 1)
    decoder_hidden = encoder_hidden
    use_tf = True if random.random() < teacher_forcing_ratio else False
    if use_tf:
        for i in range(TAR_LEN):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            decoder_outputs[i] = decoder_output
            decoder_input = target_tensor[i].unsqueeze(0)
#            loss += criterion(decoder_output, target_index.transpose(0,1)[i])
    else:
        for i in range(TAR_LEN):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            topv, topi = decoder_output.topk(1)
            decoder_outputs[i] = decoder_output
            decoder_input = torch.tensor([word_vector[index2word[int(j)]] for j in topi.view(1,-1)[0]]).unsqueeze(0)
#            loss += criterion(decoder_output, target_index.transpose(0,1)[i])

#    loss = masked_cross_entropy(decoder_outputs.transpose(0,1).contiguous(), target_index.contiguous(), target_len)
    loss = criterion(decoder_outputs.view(-1, output_size), target_index.transpose(0, 1).contiguous().view(-1))
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
#    return loss.item() / sum(target_len)
    return loss.item()

def trainIters(encoder, decoder, n_iters, print_every=1000, lr=0.1):
    print_loss = 0
    start = time.time()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr, momentum=0.5)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr, momentum=0.5)
    criterion = nn.NLLLoss(ignore_index=word2index[TAB_TOKEN])
#    criterion = nn.CrossEntropyLoss(ignore_index=word2index[TAB_TOKEN])

    for epoch in range(1, n_iters+1):
        if epoch % 5 == 0:
            for opt in [encoder_optimizer, decoder_optimizer]:
                for pg in opt.param_groups:
                    pg['lr'] = pg['lr'] * 0.1
        print '=====>epoch:', epoch, ' lr', encoder_optimizer.param_groups[0]['lr']
        for i in range(1, total_num/batch_size+1):
#        for i in range(10):
            train_pairs = sorted(pairs[(i-1)*batch_size: i*batch_size], key=lambda x: len(x[0]), reverse=True)
            loss = train(train_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss += loss

            if i % print_every == 0:
                print_loss_avg = print_loss / print_every
                print_loss = 0
                if VAL:
                    val_loss = validate(encoder, decoder, criterion)
                    print '=====>epoch:', epoch, ' batch:', i, ' train loss:',print_loss_avg, ' val loss:', val_loss, ' cost: %.2fs'% (time.time() - start)
                else:
                    print '=====>epoch:', epoch, ' batch:', i, ' loss:',print_loss_avg, ' cost: %.2fs'% (time.time()-start)
                start = time.time()


def validate(encoder, decoder, criterion):
    vbatch_size = 128
    valpairs = sorted([random.choice(val_pairs) for _ in range(vbatch_size)], key=lambda x: len(x[0]), reverse=True)
    in_pair = [construct_input_vector(pair[0]) for pair in valpairs]
    input_tensor = tensorFromVec([pair[0] for pair in in_pair])
    input_length = torch.tensor([pair[1] for pair in in_pair])
    tar_pair = [construct_output_vector(pair[1]) for pair in valpairs]
    target_tensor = tensorFromVec([pair[0] for pair in tar_pair])
    target_index = torch.tensor([pair[1] for pair in tar_pair])
    target_len = [pair[2] for pair in tar_pair]
    mask = create_mask(input_tensor)

    encoder_hidden = torch.zeros(1, input_tensor.shape[1], hidden_size, device=device)
    encoder_outputs = torch.zeros(MAX_LEN, vbatch_size, encoder.hidden_size, device=device)
    decoder_outputs = torch.zeros(TAR_LEN, vbatch_size, output_size, device=device)
    loss = 0

    encoder_output, encoder_hidden = encoder(input_tensor, input_length, encoder_hidden)
    for i in range(encoder_output.size()[0]):
        encoder_outputs[i] = encoder_output[i]

    decoder_input = torch.tensor([[word_vector[SOS_TOKEN]] for _ in range(vbatch_size)], device=device).transpose(0, 1)
    decoder_hidden = encoder_hidden
    use_tf = True if random.random() < teacher_forcing_ratio else False
    if use_tf:
        for i in range(TAR_LEN):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            decoder_outputs[i] = decoder_output
            decoder_input = target_tensor[i].unsqueeze(0)
    else:
        for i in range(TAR_LEN):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            decoder_outputs[i] = decoder_output
            topv, topi = decoder_output.topk(1)
            decoder_input = torch.tensor([word_vector[index2word[int(j)]] for j in topi.view(1,-1)[0]]).unsqueeze(0)

    loss = criterion(decoder_outputs.view(-1, output_size), target_index.transpose(0, 1).contiguous().view(-1))
    return loss.item()


def tensorFromVec(vector):
    return torch.tensor(vector, device=device).transpose(0,1)

def tensorFromPair(pair):
    in_tensor = torch.tensor([pair[0]], device=device).transpose(0,1)
    out_tensor = torch.tensor([pair[1]], device=device).transpose(0,1)
    return (in_tensor, out_tensor)


def evaluate(encoder, decoder, data):
    in_vec, input_len = construct_input_vector(data)
    input_len = torch.tensor([input_len])
    in_tensor = torch.tensor([in_vec], device=device).transpose(0,1)
    mask = create_mask(in_tensor)
    with torch.no_grad():
        invec_len = in_tensor.size()[0]
        encoder_hidden = torch.zeros(1, 1, hidden_size, device=device)
        encoder_outputs = torch.zeros(MAX_LEN, 1, encoder.hidden_size, device=device)

        encoder_output, encoder_hidden = encoder(in_tensor, input_len, encoder_hidden)

        encoder_outputs[0] = encoder_output[0]
        decoder_input = torch.tensor([[word_vector[SOS_TOKEN]]], device=device)
        decoder_hidden = encoder_hidden
        decoder_words = []
        while True:
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            topv, topi = decoder_output.data.topk(1)
            topi_vec = word_vector[index2word[int(topi.squeeze().detach())]]
            if topi_vec == word_vector[EOS_TOKEN]:
                decoder_words.append('<EOS>')
                break
            else:
                decoder_words.append(index2word[int(topi.squeeze().detach())])
            if len(decoder_words) >= 20:
                break

            decoder_input = torch.tensor([[word_vector[index2word[int(topi.squeeze().detach())]]]])
    return decoder_words

def random_evaluate(encoder, decoder, num=5):
    print '==========>  predict train data  <=========='
    for i in range(num):
        pair = random.choice(pairs)
        print '===>question', pair[0].split('##')[1].encode('utf-8')
        print '===>%s keyword'%ETYPE, ' '.join(pair[1]).encode('utf-8')
        output = evaluate(encoder, decoder, pair[0])
        print '===>output', ' '.join(output).encode('utf-8'), '\n'

def create_mask(data_tensor):
    mask = []
    for i in range(data_tensor.shape[0]):
        ll = []
        for j in range(data_tensor.shape[1]):
            if torch.equal(data_tensor[i,j],torch.tensor(word_vector[TAB_TOKEN])):
                ll.append(0)
            else:
                ll.append(1)
        mask.append(ll)
    mask = torch.tensor(mask).transpose(0, 1)
    return mask


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):

    """
    Args:
        logits: A Variable containing a FloatTensor of size
        (batch, max_len, num_classes) which contains the
        unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
        (batch, max_len) which contains the index of the true
        class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
        which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
                                                                                """
    length = Variable(torch.LongTensor(length))
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
#    log_probs_flat = F.log_softmax(logits_flat)
    log_probs_flat = logits_flat
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5
ETYPE = 'sel'
VAL = True
total_num = 41522
start = time.time()
###  code table
table_code, table_headers = extract_table()
## load word vector
word_vector, word2index, index2word = load_word_vector()
pairs = prepare_data('train', ETYPE)
if VAL:
    val_pairs = prepare_data('val', ETYPE)
print 'prepare data done. cost', (time.time()-start), 's'
start = time.time()
hidden_size = 256
input_size = 300
output_size = len(word2index)
batch_size = 5
encoder = Encoder(input_size, hidden_size).to(device)
decoder = AttnDecoder(input_size, hidden_size, output_size).to(device)
print '=====>start train'
trainIters(encoder, decoder, 10)
print '=====>end train.  cost %.2fmins\n'%((time.time()-start)/60.0)
random_evaluate(encoder, decoder)
