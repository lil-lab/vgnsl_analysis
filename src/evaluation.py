# Originally Forked from https://github.com/ExplorerFreda/VGNSL?files=1
# Modified by Noriyuki Kojima (nk654@cornell.edu)

import os
import pickle

import numpy
import time
import numpy as np
from vocab import Vocabulary
from collections import OrderedDict

import torch

from utils import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s


class ContrastiveSparseReward(nn.Module):
    """ compute contrastive reward """

    def __init__(self, margin=0):
        super(ContrastiveSparseReward, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def setup_scores(self, im, s):
        self.im_feats = im
        self.s = s
        self.score = self.sim(im, s)
        self.batch_size = s.size(0)

    def forward(self, text_feat, index):
        text_feat = text_feat.unsqueeze(0)
        sims = self.sim(self.im_feats, text_feat).squeeze(1)
        scores = self.score.clone()
        scores[:, index] = sims
        # compare every diagonal score to scores in its column
        # caption retrieval, given images and retrieve captions
        reward_s = (-scores[index,:] + sims[index] - self.margin).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval, given caption and retrieve images
        reward_im = (-sims + sims[index] - self.margin).clamp(min=0)
        #print("ContrastiveSparseReward")
        #embed()
        # clear diagonals
        reward_s[index] = 0
        reward_im[index] = 0

        # sum up the reward
        reward_s = reward_s.mean(0)
        reward_im = reward_im.mean(0)
        rewards = reward_s + reward_im

        # calculate uniquness
        sims[index] = 0
        uniq = -sims.mean(0) # Negate for a better interpretation

        return rewards, uniq


class ContrastiveSparseLoss(nn.Module):
    """ compute contrastive loss for VSE """

    def __init__(self, margin=0.2):
        super(ContrastiveSparseLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def setup_scores(self, im, s):
        self.im_feats = im
        self.s = s
        self.score = self.sim(im, s)
        self.batch_size = s.size(0)

    def forward(self, text_feat, index):
        text_feat = text_feat.unsqueeze(0)
        sims = self.sim(self.im_feats, text_feat).squeeze(1)
        scores = self.score.clone()
        scores[:, index] = sims
        diagonal = scores.diag().view(self.batch_size, 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval, given images and retrieve captions
        loss_s = (scores[index,:] - sims[index] + self.margin).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval, given caption and retrieve images
        loss_im = (sims - sims[index] + self.margin).clamp(min=0)

        # clear diagonals
        loss_s[index] = 0
        loss_im[index] = 0

        # sum up the loss
        loss_s = loss_s.mean(0)
        loss_im = loss_im.mean(0)
        loss = loss_s + loss_im

        return loss


def encode_data(model, data_loader, log_step=10, logging=print, vocab=None, stage='dev'):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    spans = {}
    logged = False
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(images, captions, lengths, volatile=True)
        img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
        span_bounds = model_output[:8]

        # output sampled trees
        if (not logged) or (stage == 'test'):
            logged = True
            if stage == 'debug':
                sample_num = 1
            if stage == 'dev' or stage == "analysis":
                sample_num = 5
            for j in range(sample_num):
                logging(generate_tree(captions, tree_indices, j, vocab))

        cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # preserve the spans if analysis mode is enabled
        if stage == "analysis":
            span = extract_spans_by_order(captions, span_bounds, vocab)
            for j, id in enumerate(ids):
                spans[id] = span[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

    if stage == "debug":
        return img_embs, cap_embs, cap_span_features
    elif stage == "analysis":
        return img_embs, cap_embs, cap_span_features, spans

    return img_embs, cap_embs


def model_analysis(model, vocab, opt, *data):
    """Encode all images and captions loadable by `data_loader`
    """
    images, captions, lengths = data[0]
    # switch to evaluate mode
    model.val_start()

    # make sure val logger is used
    lengths = torch.Tensor(lengths).long()
    if torch.cuda.is_available():
        lengths = lengths.cuda()

    # compute the embeddings
    model_output = model.forward_emb(images, captions, lengths, volatile=True)
    all_pharse_features_cpu = model.txt_enc.sem_embeddings
    img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
    span_bounds = model_output[:8]
    cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)

    """
    Tokens
        1. Similarity
    """
    img_feat = img_emb.unsqueeze(2)
    word_embs = l2norm(word_embs)
    words_sim = torch.bmm(word_embs, img_feat)


    """
    Spans
        1. Similarity
    """
    img_feat = img_emb.unsqueeze(2)
    span_feat = torch.stack(cap_span_features, dim=1)
    span_feat = l2norm(span_feat)
    spans_sim = torch.bmm(span_feat, img_feat)


    """
    Phrases
        1. Similarity
        2. Reward
    """
    reward_fast_func = ContrastiveSparseReward(margin=0.2)
    left_loss_fast_func = ContrastiveSparseLoss(margin=0.2)
    right_loss_fast_func = ContrastiveSparseLoss(margin=0.2)

    img_feat = img_emb.unsqueeze(2)
    phrase_sims = []
    phrase_logits = []
    phrase_rewards= []

    # similarity
    for t in range(len(all_pharse_features_cpu)):
        if torch.cuda.is_available():
            text_feat = torch.tensor(all_pharse_features_cpu[t]).cuda()
            phrase_sim = torch.zeros((text_feat.size(0),text_feat.size(1)-1))
            phrase_logit = torch.zeros((text_feat.size(0),text_feat.size(1)-1))

            for k in range(text_feat.size(1)-1):
                syn_feats = torch.cat(
                    (l2norm(text_feat[:,k+1,:]), l2norm(text_feat[:,k,:])),
                    dim=1
                )
                logits = model.txt_enc.syn_score(syn_feats)
                phrase_feat = text_feat[:,k,:] + text_feat[:,k+1,:]
                phrase_feat = l2norm(phrase_feat)
                tmp_sims = torch.bmm(phrase_feat.unsqueeze(1), img_feat)
                phrase_sim[:,k] = tmp_sims[:,0,0]
                phrase_logit[:,k] = logits[:,0]

            phrase_sim = phrase_sim.data.cpu().numpy().copy()
            phrase_sims.append(phrase_sim)
            phrase_logit = phrase_logit.data.cpu().numpy().copy()
            phrase_logits.append(phrase_logit)

    # reward
    batch_size = len(lengths)
    phrase_rewards = torch.zeros((lengths.max(0)[0]-1, batch_size, lengths.max(0)[0] - 1)) # t*bs*pos
    phrase_uniqs = torch.zeros((lengths.max(0)[0]-1, batch_size, lengths.max(0)[0] - 1))
    left_regs = torch.zeros((lengths.max(0)[0]-1, batch_size, lengths.max(0)[0] - 1)) # t*bs*pos
    right_regs = torch.zeros((lengths.max(0)[0]-1, batch_size, lengths.max(0)[0] - 1)) # t*bs*pos

    for i in range(lengths.max(0)[0] - 1):
        curr_imgs = list()
        curr_caps = list()
        curr_left_caps = list()
        curr_right_caps = list()
        indices = list()
        for j in range(batch_size):
            if i < lengths[j] - 1:
                curr_imgs.append(img_emb[j].reshape(1, -1))
                curr_caps.append(cap_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                curr_left_caps.append(left_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                curr_right_caps.append(right_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                indices.append(j)

        curr_img_emb = torch.cat(curr_imgs, dim=0)
        curr_cap_emb = torch.cat(curr_caps, dim=0)
        left_cap_emb = torch.cat(curr_left_caps, dim=0)
        right_cap_emb = torch.cat(curr_right_caps, dim=0)
        reward_fast_func.setup_scores(curr_img_emb, curr_cap_emb)
        left_loss_fast_func.setup_scores(curr_img_emb, left_cap_emb)
        right_loss_fast_func.setup_scores(curr_img_emb, right_cap_emb)

        ct = 0
        for j in range(batch_size): # batch
            if i < lengths[j] - 1: # timestamp
                t = lengths[j] - 2 - i
                phrases_embs = torch.tensor(all_pharse_features_cpu[t][j,:,:]).cuda()
                for k in range(phrases_embs.size(0)-1):
                    new_feature = phrases_embs[k+1,:] + phrases_embs[k,:]
                    new_feature = l2norm(new_feature)
                    reward_fast, uniq = reward_fast_func(new_feature, ct)
                    left_reg = left_loss_fast_func(l2norm(phrases_embs[k,:]), ct)
                    right_reg = right_loss_fast_func(l2norm(phrases_embs[k+1,:]), ct)
                    phrase_rewards[t,j,k] = reward_fast
                    phrase_uniqs[t,j,k] = uniq
                    left_regs[t,j,k] = left_reg
                    right_regs[t,j,k] = right_reg
                ct += 1

    phrase_rewards = (left_regs + 1.0) * phrase_rewards \
        / (opt.lambda_hi * right_regs + 1.0)
    phrase_rewards = phrase_rewards.data.cpu().numpy()
    phrase_uniqs = phrase_uniqs.data.cpu().numpy()

    """
    Captions
        1. Simlarity
    """
    img_feat = img_emb.unsqueeze(2)
    cap_emb = l2norm(cap_emb)
    cap_sim = torch.bmm(cap_emb.unsqueeze(1), img_feat)

    # preserve the embeddings by copying from gpu and converting to numpy
    img_embs = img_emb.data.cpu().numpy().copy()
    cap_embs = cap_emb.data.cpu().numpy().copy()
    words_sims = words_sim.data.cpu().numpy()[:,:,0]
    spans_sims = spans_sim.data.cpu().numpy()[:,:,0]
    cap_sims = cap_sim.data.cpu().numpy()[:,:,0]

    # preserve the spans if analysis midode is enabled
    spans, phrases = extract_spans_by_order(captions, span_bounds, vocab, tree_indices)

    del images, captions

    return img_embs, cap_sims, words_sims, spans, spans_sims, phrases, phrase_sims, phrase_logits, phrase_rewards, phrase_uniqs


def ground_truth_analysis(model, vocab, trees, img_embs, *data):
    images, captions, lengths = data[0]

    # switch to evaluate mode
    model.val_start()

    # numpy array to keep all the embeddings
    spans = {}

    # make sure val logger is used
    lengths = torch.Tensor(lengths).long()
    if torch.cuda.is_available():
        lengths = lengths.cuda()
    if torch.cuda.is_available():
        captions = captions.cuda()

    # compute the embeddings (ignore start and end tokens ....)
    sem_embeddings = model.txt_enc.sem_embedding(captions.cuda())
    seq_length = sem_embeddings.shape[1]

    splits = rectangulate_splits(trees, seq_length)
    text_feat = []

    for j in range(seq_length-1):
        starts = splits[:,j,0]
        ends = splits[:,j,1]
        center_masks, left_masks = indices_mask(starts, ends, max_length=seq_length)
        center_masks = center_masks.float()
        left_masks = left_masks.float()
        if sem_embeddings.is_cuda:
            center_masks = center_masks.cuda()
        new_features = torch.sum(sem_embeddings*center_masks.unsqueeze(2), dim=1)
        new_features = l2norm(new_features)
        text_feat.append(new_features)
        sem_embeddings[center_masks == 1] = 0
        com_embeddings = torch.zeros(sem_embeddings.shape).cuda()
        com_embeddings += new_features.unsqueeze(1)
        com_embeddings[left_masks != 1] = 0
        #com_embeddings[center_masks != 1] = 0 (Buggy version but works better.)
        sem_embeddings = sem_embeddings + com_embeddings

    cap_emb = sem_embeddings[:,1,:]

    # preserve the spans if analysis mode is enabled
    span_bounds = splits.permute(1,0,2).long()
    spans = extract_spans_by_order(captions, span_bounds, vocab)

    # calculate span similarity
    span_feat = torch.stack(text_feat, dim=1).float()
    if span_feat.is_cuda:
        img_feat = torch.tensor(img_embs).unsqueeze(2).float().cuda()
    spans_sim = torch.bmm(span_feat, img_feat)

    # calculate caption similarity
    cap_emb = l2norm(cap_emb)
    cap_sim = torch.bmm(cap_emb.unsqueeze(1), img_feat)

    # preserve the embeddings by copying from gpu and converting to numpy
    cap_embs = cap_emb.data.cpu().numpy().copy()
    spans_sims = spans_sim.data.cpu().numpy()[:,:,0]
    cap_sims = cap_sim.data.cpu().numpy()[:,:,0]

    del images, captions

    return cap_sims, spans, spans_sims


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # compute scores
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def ti_sim(images, captions, npts=None, measure='cosine'):
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])
    sim = []
    for index in range(npts):
        # Get query captions
        queries = captions[5 * index:5 * index + 5]
        # compute scores
        d = numpy.dot(queries, ims[index].T)
        sim = sim + list(d)
    return sim


def test_trees(model_path=None, data_splits='test', custom_gtfile=None, model=None,\
                    data_loader=None, opt=None, return_indicies=False):
    from data import get_eval_loader
    from model import VGNSL

    """ use the trained model to generate parse trees for text """
    # check a model exists
    assert(not((model_path is None) and (model is None)))

    # load  model and options
    if opt is None:
        checkpoint = torch.load(model_path, map_location='cpu')
        opt = checkpoint['opt']

    # load vocabulary used by the model
    vocab = pickle.load(open(os.path.join(opt.data_path, 'vocab.pkl'), 'rb'))
    opt.vocab_size = len(vocab)

    if model is None:
        # construct model
        model = VGNSL(opt)
        # load model state
        model.load_state_dict(checkpoint['model'])
    if data_loader is None:
        data_loader = get_eval_loader(
            opt.data_path, data_splits, vocab, opt.batch_size, opt.workers,
            load_img=False, img_dim=opt.img_dim
        )

    cap_embs = None
    logged = False
    trees = list()
    spans = list()
    for i, data in enumerate(data_loader):
        if len(data) == 3:
            (captions, lengths, ids) = data
        else:
            (images, captions, lengths, ids) = data
        # make sure val logger is used
        model.logger = print
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        if len(data) == 3:
            model_output = model.forward_emb(captions, lengths, volatile=True)
            cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
            span_bounds = model_output[:8]
        else:
            model_output = model.forward_emb(images, captions, lengths, volatile=True)
            img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
            span_bounds = model_output[:8]
        candidate_trees = list()
        for j in range(len(ids)):
            candidate_trees.append(generate_tree(captions, tree_indices, j, vocab))

        appended_trees = ['' for _ in range(len(ids))]
        candidate_spans = ['' for _ in range(len(ids))]
        span_bounds = torch.stack(span_bounds,1)
        span_bounds = span_bounds.data.cpu().numpy()
        for j in range(len(ids)):
            appended_trees[ids[j] - min(ids)] = clean_tree(candidate_trees[j])
            candidate_spans[ids[j] - min(ids)] = clean_span(span_bounds[j, :, :], lengths[j].data.cpu().numpy())
        trees.extend(appended_trees)
        spans.extend(candidate_spans)
        cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)
        if len(data) == 3:
            del captions, cap_emb
        else:
            del images, captions, img_emb, cap_emb

    if custom_gtfile is not None:
        ground_truth = [line.strip() for line in open(custom_gtfile)]
    else:
        ground_truth = [line.strip() for line in open(
            os.path.join(opt.data_path, '{}_ground-truth.txt'.format(data_splits)))]

    if return_indicies:
        return trees, ground_truth, spans
    else:
        return trees, ground_truth


def create_trees(model, data, vocab):
    (images, captions, lengths, ids) = data
    trees = {}
    lengths = torch.Tensor(lengths).long()
    if torch.cuda.is_available():
        lengths = lengths.cuda()
    model_output = model.forward_emb(images, captions, lengths, volatile=True)
    img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
    span_bounds = model_output[:8]
    for i, id in enumerate(ids):
        trees[id] = generate_tree(captions, tree_indices, i, vocab)
    return trees
