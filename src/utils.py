# Originally Forked from https://github.com/ExplorerFreda/VGNSL?files=1
# Modified by Noriyuki Kojima (nk654@cornell.edu)

import os, copy
import numpy as np

import collections
from functools import reduce

import torch
import torch.nn as nn


class EmbeddingCombiner(nn.Module):
    def __init__(self, *embeddings):
        super().__init__()
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, input):
        return torch.cat([e(input) for e in self.embeddings], dim=-1)


def tree2list(tokens):
    tree = list()
    list_stack = list()
    list_stack.append(tree)
    stack_top = tree
    for token in tokens:
        if token == '(':
            new_span = []
            stack_top.append(new_span)
            list_stack.append(new_span)
            stack_top = new_span
        elif token == ')':
            list_stack = list_stack[:-1]
            if len(list_stack) != 0:
                stack_top = list_stack[-1]
        else:
            stack_top.append(token)
    return tree


def treelist2dict(tree, d):
    if type(tree) is str:
        return tree
    span_reprs = [treelist2dict(s, d) for s in tree]
    d[' '.join(span_reprs)] = tree
    return ' '.join(span_reprs)


def tree2str(tree):
    if type(tree) is str:
        return tree
    items = [tree2str(item) for item in tree]
    return '( ' + ' '.join(items) + ' )'


def make_embeddings(opt, vocab_size, dim):
    init_embeddings = None
    if hasattr(opt, 'vocab_init_embeddings'):
        try:
            init_embeddings = torch.tensor(torch.load(opt.vocab_init_embeddings))
        except:
            init_embeddings = torch.tensor(np.load(opt.vocab_init_embeddings))

    emb = None
    if opt.init_embeddings_type in ('override', 'partial'):
        emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        if init_embeddings is not None:
            if opt.init_embeddings_type == 'override':
                emb.weight.data.copy_(init_embeddings)
            else:
                assert opt.init_embeddings_type == 'partial'
                emb.weight.data[:, :init_embeddings.size(1)] = init_embeddings
    elif opt.init_embeddings_type == 'partial-fixed':
        partial_dim = opt.init_embeddings_partial_dim
        emb1 = nn.Embedding(vocab_size, partial_dim, padding_idx=0)
        emb2 = nn.Embedding(vocab_size, dim - partial_dim, padding_idx=0)

        if init_embeddings is not None:
            emb1.weight.data.copy_(init_embeddings)
        emb1.weight.requires_grad_(False)

        emb = EmbeddingCombiner(emb1, emb2)
    else:
        raise NotImplementedError()
    return emb


def concat_shape(*shapes):
    output = []
    for s in shapes:
        if isinstance(s, collections.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)


def broadcast(tensor, dim, size):
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim+1:]))


def add_dim(tensor, dim, size):
    return broadcast(tensor.unsqueeze(dim), dim, size)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs"""
    return im.mm(s.t())


def l2norm(x, dim=-1):
    return x / x.norm(2, dim=dim, keepdim=True).clamp(min=1e-6)


def generate_tree(captions, tree_indices, pos, vocab, pad_word='<pad>'):
    words = list(filter(lambda x: x!=pad_word, [{
        '(': '**LP**',
        ')': '**RP**'
    }.get(vocab.idx2word[int(word)], vocab.idx2word[int(word)]) for word in captions[pos]]))
    idx = 0
    while len(words) > 1:
        p = tree_indices[idx][pos]
        words = words[:p] + ['( {:s} {:s} )'.format(words[p], words[p+1])] + words[p+2:]
        idx += 1
    return words[0]


def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def index_one_hot_ellipsis(tensor, dim, index):
    tensor_shape = tensor.size()
    tensor = tensor.view(prod(tensor_shape[:dim]), tensor_shape[dim], prod(tensor_shape[dim+1:]))
    assert tensor.size(0) == index.size(0)
    index = index.unsqueeze(-1).unsqueeze(-1)
    index = index.expand(tensor.size(0), 1, tensor.size(2))
    tensor = tensor.gather(1, index)
    return tensor.view(tensor_shape[:dim] + tensor_shape[dim+1:])


def index_mask(indices, max_length):
    batch_size = indices.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand
    if indices.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    indices_expand = indices.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand == indices_expand

def indices_mask(starts, ends, max_length):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = starts.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand
    if starts.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    starts_expand = starts.unsqueeze(1).expand_as(seq_range_expand).long()
    ends_expand = ends.unsqueeze(1).expand_as(seq_range_expand).long()
    return (seq_range_expand >= starts_expand) * (seq_range_expand <= ends_expand), seq_range_expand == starts_expand


def index_range_ellipsis(x, a, b, dim=1, padding_zero=True):
    assert dim == 1

    batch_size, seq_length = x.size()[:2]
    seg_lengths = b - a
    max_seg_length = seg_lengths.max().item()

    mask = length2mask(seg_lengths, max_seg_length)

    # indices values: [[0, 1, 0, 0, ...], [1, 2, 3, 0, ...], ...]
    base = torch.arange(max_seg_length)
    if torch.cuda.is_available():
        base = base.cuda()
    indices = add_dim(base, 0, batch_size) + a.unsqueeze(-1)
    indices = indices * mask.long()  # shape: [batch_size, max_seg_length]

    # batch_indices values: [[0, 0, 0...], [1, 1, 1, ...], ...]
    base = torch.arange(batch_size)
    if torch.cuda.is_available():
        base = base.cuda()
    batch_indices = add_dim(base, 1, max_seg_length)  # shape: [batch_size, max_seg_length]

    flattened_x = x.reshape(concat_shape(-1, x.size()[2:]))
    flattened_indices = (indices + batch_indices * seq_length).reshape(-1)
    output = flattened_x[flattened_indices].reshape(
        concat_shape(batch_size, max_seg_length, x.size()[2:]))

    if padding_zero:
        output = output * add_dim_as_except(mask.type_as(output), output, 0, 1)

    return output

def add_dim_as_except(tensor, target, *excepts):
    assert len(excepts) == tensor.dim()
    tensor = tensor.clone()
    excepts = [e + target.dim() if e < 0 else e for e in excepts]
    for i in range(target.dim()):
        if i not in excepts:
            tensor.unsqueeze_(i)
    return tensor


def length2mask(lengths, max_length):
    rng = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    lengths = lengths.unsqueeze(-1)
    rng = add_dim_as_except(rng, lengths, -1)
    mask = rng < lengths
    return mask.float()


def prod(values, default=1):
    if len(values) == 0:
        return default
    return reduce(lambda x, y: x * y, values)


def clean_tree(sentence, remove_tag_set={'<start>', '<end>', '<pad>'}):
    for tag in remove_tag_set:
        sentence = sentence.replace(tag, ' ')
    items = sentence.split()
    stack = list()
    for item in items:
        if item != ')':
            stack.append(item)
        else:
            pos = -1
            while stack[pos] != '(':
                pos -= 1
            if pos == -2:
                stack = stack[:-2] + [stack[-1]] # removing single word-bracket case e..g, (cat)
            else:
                stack = stack[:pos] + [' '.join(['('] + stack[pos+1:] + [')'])]
    assert len(stack) == 1
    return stack[0]


def clean_span(span_bounds, length):
    max_len = length - 1
    new_spans = []
    for sp in span_bounds:
        st, end = sp
        if st == 0: #<start>
            st = st+1
        if end == max_len: #<end>
            end = end-1
        # discard conditions
        if end > max_len -1:
            continue
        if st == end:
            continue
        if end < st:
            continue
        new_span = [st-1, end-1]
        if new_span in new_spans:
            continue
        new_spans.append(new_span)
    return new_spans

def rectangulate_splits(trees, seq_length, slack=1):
    batch_size = len(trees)
    splits = torch.ones(batch_size, seq_length-1, 2) * -1 # Set this -1: <pad>
    for i in range(batch_size):
        length = len(trees[i])
        try:
            tree = np.array(trees[i]) + 1
            splits[i, :length, :] = torch.Tensor(tree)
        except:
            print("empty")
    return splits

def extract_spans_by_order(captions_idx, span_bounds, vocab, tree_indices=None):
    if type(span_bounds) == list:
        span_bounds = torch.stack(span_bounds)
    span_bounds = span_bounds.data.cpu().numpy()

    spans = [[] for i in range(len(captions_idx))]
    captions = [[] for i in range(len(captions_idx))]
    all_phrases = []

    # create a table of captions in string
    for i in range(len(captions_idx)):
        for j in range(len(span_bounds)+1):
            word = vocab.idx2word[int(captions_idx[i][j])]
            captions[i].append(word)
    captions = np.array(captions)

    # summarize all phrases
    if tree_indices is not None:
        tree_indices = torch.stack(tree_indices)
        tree_indices = tree_indices.data.cpu().numpy()
        if tree_indices is not None:
            all_phrases = []
            phrases = [list(captions[i]) for i in range(captions.shape[0])]

    # create spans
    for j in range(len(span_bounds)):
        all_pairs =  [None for _ in range(captions.shape[0])]
        for i in range(len(captions_idx)):
            bound = span_bounds[j][i]
            st, end = bound
            if st == -1: # for gold-tree
                continue
            # calculate spans
            span = captions[i][st:end+1]
            if not '<pad>' in span:
                spans[i].append(span)

            if tree_indices is not None:
                # calculate all pairs
                phrase = phrases[i]
                pairs = []
                if tree_indices is not None:
                    for k in range(len(phrase)):
                        if k+1 >= len(phrase) or phrase[k+1] == '<pad>':
                            break
                        else:
                            pairs.append(phrase[k] + " " + phrase[k+1])
                all_pairs[i] = pairs
                # calculate current phrases
                phrase[tree_indices[j,i]] = " ".join(span)
                phrase[tree_indices[j,i]+1:-1] = phrase[tree_indices[j,i]+2:]
                phrase[-1] = '<pad>'
                phrases[i] = phrase

        if tree_indices is not None:
            all_phrases.append(all_pairs)

    if tree_indices is not None:
        return spans, all_phrases
    return spans
