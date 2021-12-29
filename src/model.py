import os
from copy import Error
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn import functional
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import make_embeddings, l2norm, cosine_sim, sequence_mask, \
    index_mask, index_one_hot_ellipsis, add_dim


class EncoderImagePrecomp(nn.Module):
    """ image encoder """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """ Xavier initialization for the fully connected layer """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """ extract image feature vectors """
        if type(images) == list:
            features = torch.cat(images, dim=0)
            features = self.fc(features.float())
            assert(not self.no_imgnorm)
            if not self.no_imgnorm:
                features = l2norm(features)
            splits = []
            ct = 0
            for i in range(len(images)):
                splits.append((ct, ct+images[i].shape[0]))
                ct += images[i].shape[0]
            return features, splits
        else:
            features = self.fc(images.float())
            if not self.no_imgnorm:
                features = l2norm(features)
            return features

    def load_state_dict(self, state_dict):
        """ copies parameters, overwritting the default one to
            accept state_dict from Full model """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """ text encoder """
    def __init__(self, opt, vocab_size, semantics_dim):
        super(EncoderText, self).__init__()
        opt.syntax_dim = semantics_dim  # syntax is tied with semantics
        if hasattr(opt, 'bottleneck_dim'):
            self.bottleneck_dim = opt.bottleneck_dim
        else:
            self.bottleneck_dim = 2
        self.vocab_size = vocab_size
        self.semantics_dim = semantics_dim
        self.sem_embedding = make_embeddings(opt, self.vocab_size, self.semantics_dim)

        # combine function
        if hasattr(opt, 'combine_fn'):
            self.combine_fn = opt.combine_fn
        else:
            self.combine_fn = "mean"

        # tagger function ()
        self.tagger_fn = opt.tagger_fn
        if self.tagger_fn == "softmax":
            self.tagger = nn.Sequential(
                nn.Linear(opt.syntax_dim, int(opt.syntax_score_hidden/2)),
                nn.ReLU(),
                nn.Linear(int(opt.syntax_score_hidden/2), self.bottleneck_dim, bias=False),
                nn.Softmax()
            )
        elif self.tagger_fn == "linear":
            self.tagger = nn.Sequential(
                nn.Linear(opt.syntax_dim, self.bottleneck_dim, bias=False)
            )
        elif self.tagger_fn == "deeper":
            self.tagger = nn.Sequential(
                nn.Linear(opt.syntax_dim, int(200)),
                nn.ReLU(),
                nn.Linear(int(200), int(100)),
                nn.ReLU(),
                nn.Linear(int(100), int(50)),
                nn.ReLU(),
                nn.Linear(int(50), self.bottleneck_dim, bias=False)
            )
        elif self.tagger_fn == "deeper_recursive":
            self.tagger_token2tags = nn.Sequential(
                nn.Linear(opt.syntax_dim, int(200)),
                nn.ReLU(),
                nn.Linear(int(200), int(100)),
                nn.ReLU(),
                nn.Linear(int(100), int(50)),
                nn.ReLU(),
                nn.Linear(int(50), self.bottleneck_dim, bias=False)
            )
            self.tagger_tags2tags = nn.Sequential(
                nn.Linear(self.bottleneck_dim*2, self.bottleneck_dim*2),
                nn.ReLU(),
                nn.Linear(self.bottleneck_dim*2, self.bottleneck_dim),
                nn.ReLU(),
                nn.Linear(self.bottleneck_dim, self.bottleneck_dim, bias=False)
            )
        elif self.tagger_fn == "linear_recursive":
            self.tagger_token2tags = nn.Sequential(
                nn.Linear(opt.syntax_dim, int(opt.syntax_score_hidden/2)),
                nn.ReLU(),
                nn.Linear(int(opt.syntax_score_hidden/2), self.bottleneck_dim, bias=False)
            )
            self.tagger_tags2tags = nn.Sequential(
                nn.Linear(self.bottleneck_dim*2, self.bottleneck_dim)
            )
        elif self.tagger_fn == "mean_recursive":
            self.tagger_token2tags = nn.Sequential(
                nn.Linear(opt.syntax_dim, int(opt.syntax_score_hidden/2)),
                nn.ReLU(),
                nn.Linear(int(opt.syntax_score_hidden/2), self.bottleneck_dim, bias=False)
            )
        else:
            self.tagger = nn.Sequential(
                nn.Linear(opt.syntax_dim, int(opt.syntax_score_hidden/2)),
                nn.ReLU(),
                nn.Linear(int(opt.syntax_score_hidden/2), self.bottleneck_dim, bias=False)
            )

        # score function
        if hasattr(opt, 'score_fn'):
            self.score_fn = opt.score_fn
        else:
            self.score_fn = "linear"

        self.syn_score = nn.Sequential(
            nn.Linear(self.bottleneck_dim*2, 1, bias=False)
        )
        if self.score_fn == "linear":
            pass
        elif self.score_fn == "mean":
            self.syn_score[0].weight.data[0] = torch.Tensor([0.5, 0.5])
            self.syn_score[0].weight.requires_grad = False
        elif self.score_fn == "mean_hi":
            self.syn_score[0].weight.data[0] = torch.Tensor([1, 20])
            self.syn_score[0].weight.requires_grad = False
        elif self.score_fn == "max":
            self.syn_score[0].weight.data[0] = torch.Tensor([0, 1])
            self.syn_score[0].weight.requires_grad = False

    def reset_weights(self):
        self.sem_embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths, volatile=False):
        """ sample a tree for each sentence """
        max_select_cnt = int(lengths.max(dim=0)[0].item()) - 1

        tree_indices = list()
        tree_probs = list()
        span_bounds = list()
        features = list()
        left_span_features = list()
        right_span_features = list()

        # closed range: [left_bounds[i], right_bounds[i]]
        left_bounds = add_dim(torch.arange(
            0, max_select_cnt + 1, dtype=torch.long, device=x.device), 0, x.size(0))
        right_bounds = left_bounds

        sem_embeddings = self.sem_embedding(x)
        syn_embeddings = sem_embeddings
        if "recursive" in self.tagger_fn:
            tag_embeddings = torch.zeros((sem_embeddings.shape[0], sem_embeddings.shape[1], self.bottleneck_dim)).cuda().float()

        output_word_embeddings = sem_embeddings * \
            sequence_mask(lengths, max_length=lengths.max()).unsqueeze(-1).float()

        valid_bs = lengths.size(0)

        self.sem_embeddings = [] 

        for i in range(max_select_cnt):
            self.sem_embeddings.append(sem_embeddings.data.cpu().numpy().copy()) 

            seq_length = sem_embeddings.size(1)
            # set invalid positions to 0 prob
            # [0, 0, ..., 1, 1, ...]
            length_mask = 1 - sequence_mask(
                (lengths - 1 - i).clamp(min=0), max_length=seq_length - 1).float()

            undone_mask = 1 - length_mask[:, 0]
            if "recursive" in self.tagger_fn:
                if i == 0:
                    tags = self.tagger_token2tags(l2norm(syn_embeddings))
                    tag_embeddings[:,:,:] = tags
                    tag1 = tags[:,1:]
                    tag2 = tags[:, :-1]
                else:
                    tag1 = tag_embeddings[:,1:]
                    tag2 = tag_embeddings[:,:-1]

                # Predict tags recursively
                if self.tagger_fn == "deeper_recursive" or self.tagger_fn == "linear_recursive":
                    syn_feats = torch.cat(
                        [tag2, tag1],
                        dim=2
                    )
                    tags_combined = self.tagger_tags2tags(syn_feats).squeeze(-1)
                    if len(tags_combined.shape) == 2:
                        tags_combined = tags_combined.unsqueeze(2)
                elif self.tagger_fn == "mean_recursive":
                    if self.combine_fn == "mean":
                        tags_combined = (tag1 + tag2) / 2.
                    elif self.combine_fn == "max":
                        assert(self.bottleneck_dim == 1)
                        tags = torch.cat([tag1, tag2], 2)
                        tags_combined = torch.max(tags, 2)[0]
                        tags_combined = tags_combined.unsqueeze(2)
                    elif self.combine_fn == "min":
                        assert(self.bottleneck_dim == 1)
                        tags = torch.cat([tag1, tag2], 2)
                        tags_combined = torch.min(tags, 2)[0]
                        tags_combined = tags_combined.unsqueeze(2)
                else:
                    raise NotImplementedError
            else:
                tag1 = self.tagger(l2norm(syn_embeddings[:, 1:]))
                tag2 = self.tagger(l2norm(syn_embeddings[:, :-1]))

            syn_feats = torch.cat(
                [tag2, tag1],
                dim=2
            )

            prob_logits = self.syn_score(syn_feats).squeeze(-1)
            prob_logits = prob_logits - 1e10 * length_mask
            probs = F.softmax(prob_logits, dim=1)

            if not volatile:
                sampler = Categorical(probs)
                indices = sampler.sample()
            else:
                values = probs.max(1)[0]
                indices = probs.max(1)[1]
            tree_indices.append(indices)
            tree_probs.append(index_one_hot_ellipsis(probs, 1, indices))

            this_spans = torch.stack([
                index_one_hot_ellipsis(left_bounds, 1, indices),
                index_one_hot_ellipsis(right_bounds, 1, indices + 1)
            ], dim=1)
            this_features = torch.add(
                index_one_hot_ellipsis(sem_embeddings, 1, indices),
                index_one_hot_ellipsis(sem_embeddings, 1, indices + 1)
            )
            this_left_features = index_one_hot_ellipsis(sem_embeddings, 1, indices)
            this_right_features = index_one_hot_ellipsis(sem_embeddings, 1, indices + 1)
            this_features = l2norm(this_features)
            this_left_features = l2norm(this_left_features)
            this_right_features = l2norm(this_right_features)

            span_bounds.append(this_spans)
            features.append(l2norm(this_features) * undone_mask.unsqueeze(-1).float())
            left_span_features.append(this_left_features * undone_mask.unsqueeze(-1).float())
            right_span_features.append(this_right_features * undone_mask.unsqueeze(-1).float())

            # update word embeddings
            left_mask = sequence_mask(indices, max_length=seq_length).float()
            right_mask = 1 - sequence_mask(indices + 2, max_length=seq_length).float()
            center_mask = index_mask(indices, max_length=seq_length).float()
            update_masks = (left_mask, right_mask, center_mask)

            this_features_syn = torch.add(
                index_one_hot_ellipsis(syn_embeddings, 1, indices),
                index_one_hot_ellipsis(syn_embeddings, 1, indices + 1)
            )
            this_features_syn = l2norm(this_features_syn)
            syn_embeddings = self.update_with_mask(syn_embeddings, syn_embeddings, this_features_syn, *update_masks)

            if "recursive" in self.tagger_fn:
                this_features_tags = index_one_hot_ellipsis(tags_combined, 1, indices)
                tag_embeddings = self.update_with_mask(tag_embeddings, tag_embeddings, this_features_tags, *update_masks)

            sem_embeddings = self.update_with_mask(sem_embeddings, sem_embeddings, this_features, *update_masks)
            left_bounds = self.update_with_mask(left_bounds, left_bounds, this_spans[:, 0], *update_masks)
            right_bounds = self.update_with_mask(right_bounds, right_bounds, this_spans[:, 1], *update_masks)

        return features, left_span_features, right_span_features, output_word_embeddings, tree_indices, \
               tree_probs, span_bounds

    @staticmethod
    def update_with_mask(lv, rv, cv, lm, rm, cm):
        if lv.dim() > lm.dim():
            lm = lm.unsqueeze(2)
            rm = rm.unsqueeze(2)
            cm = cm.unsqueeze(2)

        return (lv * lm.to(lv))[:, :-1] + (rv * rm.to(rv))[:, 1:] + (cv.unsqueeze(1) * cm.to(cv))[:, :-1]


class ContrastiveReward(nn.Module):
    """ compute contrastive reward """

    def __init__(self, margin=0, loss_type="sh-loss"):
        super(ContrastiveReward, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.loss_type = loss_type
        assert(self.loss_type == "sh-loss")

    def forward(self, im, s):
        """ return the reward """
        scores = self.sim(im, s)
        diagonal = scores.diag().view(scores.shape[0], 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if self.loss_type == "sh-loss":
            # compare every diagonal score to scores in its column
            # caption retrieval, given images and retrieve captions
            reward_s = (d1 - scores - self.margin).clamp(min=0)
            # compare every diagonal score to scores in its row
            # image retrieval, given caption and retrieve images
            reward_im = (d2 - scores - self.margin).clamp(min=0)
            # clear diagonals
            I = torch.eye(scores.size(0)) > .5
            if torch.cuda.is_available():
                I = I.cuda()
            reward_s = reward_s.masked_fill_(I, 0)
            reward_im = reward_im.masked_fill_(I, 0)

            # sum up the reward
            reward_s = reward_s.mean(1)
            reward_im = reward_im.mean(0)
        elif self.loss_type == "mh-loss":
            # Mask own correspondance
            I = torch.eye(scores.size(0)) > .5
            if torch.cuda.is_available():
                I = I.cuda()
            scores = scores.masked_fill_(I, 0)
            reward_s = (diagonal.squeeze(1) - torch.max(scores,dim=0)[0] - self.margin).clamp(min=0)
            reward_im = (diagonal.squeeze(1) - torch.max(scores,dim=1)[0] - self.margin).clamp(min=0)

        return reward_s + reward_im


class ContrastiveLoss(nn.Module):
    """ compute contrastive loss for VSE """

    def __init__(self, margin=0, loss_type="mh-loss"):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.loss_type = loss_type
        assert(self.loss_type == "mh-loss" or self.loss_type == "sh-loss")

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(scores.shape[0], 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if self.loss_type == "sh-loss":
            loss_s = (self.margin + scores - d1).clamp(min=0)
            loss_im = (self.margin + scores - d2).clamp(min=0)
            I = torch.eye(scores.size(0)) > .5
            if torch.cuda.is_available():
                I = I.cuda()
            loss_s = loss_s.masked_fill_(I, 0)
            loss_im = loss_im.masked_fill_(I, 0)

            loss_s = loss_s.mean(1)
            loss_im = loss_im.mean(0)
        elif self.loss_type == "mh-loss":
            # Mask own correspondance
            I = torch.eye(scores.size(0)) > .5
            if torch.cuda.is_available():
                I = I.cuda()
            scores = scores.masked_fill_(I, 0)
            # TODO do not know which one is which
            loss_s = (self.margin + torch.max(scores,dim=0)[0] - diagonal.squeeze(1)).clamp(min=0)
            loss_im = (self.margin + torch.max(scores,dim=1)[0] - diagonal.squeeze(1)).clamp(min=0)

        return loss_s + loss_im


class VGNSL(object):
    """ the main VGNSL model """

    def __init__(self, opt):
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecomp(
            opt.img_dim, opt.embed_size, opt.no_imgnorm
        )
        self.txt_enc = EncoderText(opt, opt.vocab_size, opt.word_dim)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # bounding boxes
        print("Model initialization .....")
        self.feature_op = opt.feature_op
        self.loss_type = opt.loss_type

        # loss, reward and optimizer
        self.reward_criterion = ContrastiveReward(margin=opt.margin, loss_type="sh-loss")
        self.loss_criterion = ContrastiveLoss(margin=opt.margin, loss_type=self.loss_type)
        self.vse_reward_alpha = opt.vse_reward_alpha
        self.vse_loss_alpha = opt.vse_loss_alpha
        self.lambda_hi = opt.lambda_hi


        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        self.params = params

        self.optimizer = getattr(torch.optim, opt.optimizer)(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.optimizer.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        if len(state_dict) >= 3:
            self.optimizer.load_state_dict(state_dict[2])

    def train_start(self):
        """ switch to train mode """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """ switch to evaluate mode """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            if type(images) == list:
                for i in range(len(images)):
                    images[i] = images[i].cuda()
                captions = captions.cuda()
                with torch.set_grad_enabled(not volatile):
                    img_emb, splits = self.img_enc(images)
                    txt_outputs = self.txt_enc(captions, lengths, volatile)
                return (img_emb,) + txt_outputs + (splits, )
            else:
                images = images.cuda()
                captions = captions.cuda()
                with torch.set_grad_enabled(not volatile):
                    img_emb = self.img_enc(images)
                    txt_outputs= self.txt_enc(captions, lengths, volatile)
                return (img_emb, ) + txt_outputs

    # cap_span_features: list(max_word_size), each element of list contains the feature of bs * feature-dim
    # left_span_features: list(max_word_size), each element of list contains the feature of bs * feature-dim; embedding of X_{j} for consituent made at time
    # right_span_features: list(max_word_size), each element of list contains the feature of bs * feature-dim; embedding of X_{j+1} for consituent made at time t
    def forward_reward(self, base_img_emb, cap_span_features, left_span_features, right_span_features,\
                       word_embs, lengths, span_bounds, base_img_splits=None):
        if base_img_splits is None:
            batch_size = base_img_emb.size(0)
        else:
            batch_size = len(base_img_splits)
        reward_matrix = torch.zeros(batch_size, lengths.max(0)[0]-1).float()
        left_reg_matrix = torch.zeros(batch_size, lengths.max(0)[0]-1).float()
        right_reg_matrix = torch.zeros(batch_size, lengths.max(0)[0]-1).float()
        if torch.cuda.is_available():
            reward_matrix = reward_matrix.cuda()
            right_reg_matrix = right_reg_matrix.cuda()
            left_reg_matrix = left_reg_matrix.cuda()

        matching_loss = 0
        # iterateing through all spans
        for i in range(lengths.max(0)[0] - 1):
            curr_imgs = list()
            curr_caps = list()
            curr_left_caps = list()
            curr_right_caps = list()
            indices = list()
            splits = list()
            for j in range(batch_size):
                if i < lengths[j] - 1:
                    curr_imgs.append(base_img_emb[j].reshape(1, -1))
                    curr_caps.append(cap_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                    curr_left_caps.append(left_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                    curr_right_caps.append(right_span_features[lengths[j] - 2 - i][j].reshape(1, -1))
                    indices.append(j)
                    if base_img_splits is not None:
                        splits.append(base_img_splits[j])

            cap_emb = torch.cat(curr_caps, dim=0)
            left_cap_emb = torch.cat(curr_left_caps, dim=0)
            right_cap_emb = torch.cat(curr_right_caps, dim=0)
            if base_img_splits is None:
                splits = None
                img_emb = torch.cat(curr_imgs, dim=0)
            else:
                img_emb = base_img_emb
            reward = self.reward_criterion(img_emb, cap_emb, splits)
            left_reg = self.loss_criterion(img_emb, left_cap_emb, splits)
            right_reg = self.loss_criterion(img_emb, right_cap_emb, splits)

            for idx, j in enumerate(indices):
                reward_matrix[j][lengths[j] - 2 - i] = reward[idx]
                left_reg_matrix[j][lengths[j] - 2 - i] = left_reg[idx]
                right_reg_matrix[j][lengths[j] - 2 - i] = right_reg[idx]

            this_matching_loss = self.loss_criterion(img_emb, cap_emb, splits)
            matching_loss += this_matching_loss.sum() + left_reg.sum() + right_reg.sum() 

        self.reward_matrix_raw = reward_matrix.clone()
        self.left_reg_matrix_raw = left_reg_matrix.clone()
        self.right_reg_matrix_raw = right_reg_matrix.clone()

        reward_matrix = reward_matrix / (self.lambda_hi * right_reg_matrix + 1.0)
        reward_matrix = self.vse_reward_alpha * reward_matrix

        return reward_matrix, matching_loss

    def train_emb(self, images, captions, lengths, ids=None, epoch=None, *args):
        """ one training step given images and captions """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # 1. compute the embeddings
        # 2. measure accuracy and record loss
        if self.feature_op == "imagenet":
            img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, probs, \
                span_bounds = self.forward_emb(images, captions, lengths)
            cum_reward, matching_loss = self.forward_reward(
                img_emb, cap_span_features, left_span_features, right_span_features, word_embs, lengths,
                span_bounds, None
            )
        else:
            img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, probs, \
                span_bounds, splits = self.forward_emb(images, captions, lengths)
            cum_reward, matching_loss = self.forward_reward(
                img_emb, cap_span_features, left_span_features, right_span_features, word_embs, lengths,
                span_bounds, splits
            )

        probs = torch.cat(probs, dim=0).reshape(-1, lengths.size(0)).transpose(0, 1)
        masks = sequence_mask(lengths - 1, lengths.max(0)[0] - 1).float()
        rl_loss = torch.sum(-masks * torch.log(probs) * cum_reward.detach())

        entropy = torch.sum(-(masks*probs)*torch.log((masks*probs + 1e-6)))

        loss = rl_loss + matching_loss * self.vse_loss_alpha
        loss = loss / cum_reward.shape[0]
        self.logger.update('Loss', float(loss), img_emb.size(0))
        self.logger.update('MatchLoss', float(matching_loss / cum_reward.shape[0]), img_emb.size(0))
        self.logger.update('RL-Loss', float(rl_loss / cum_reward.shape[0]), img_emb.size(0))
        self.logger.update('Cumu-Reward', float(torch.sum(cum_reward.detach()) / cum_reward.shape[0]), img_emb.size(0))
        self.logger.update('Entropy', float(torch.sum(cum_reward.detach()) / torch.sum(masks)), img_emb.size(0))

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        # clean up
        if epoch > 0:
            del cum_reward
            del tree_indices
            del probs
            del cap_span_features
            del span_bounds
