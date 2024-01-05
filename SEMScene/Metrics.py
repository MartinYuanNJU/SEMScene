"""
Calculate recall@k
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Configuration import device


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


# ===== FROM SGM =====
def xattn_score_t2i(images, obj_nums, captions, cap_lens):
    """
    Images: (n_image, max_n_objs, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    obj_nums:(n_obj) list of obj num per image
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)

    cap_lens = torch.tensor(cap_lens, dtype=captions.dtype)
    cap_lens = Variable(cap_lens)
    captions = torch.transpose(captions, 1, 2)
    for i in range(n_image):
        n_obj = obj_nums[i]
        if n_obj == 0:
            img_i = images[i, :, :].unsqueeze(0).contiguous()
        else:
            img_i = images[i, : n_obj, :].unsqueeze(0).contiguous()
        # --> (n_caption , n_region ,d)
        img_i_expand = img_i.repeat(n_caption, 1, 1)
        # --> (n_caption, d, max_n_word)
        dot = torch.bmm(img_i_expand, captions)
        dot = dot.max(dim=1, keepdim=True)[0].squeeze()
        dot = dot.view(n_caption, -1).contiguous()
        dot = dot.sum(dim=1, keepdim=True)
        cap_lens = cap_lens.contiguous().view(-1, 1).to(device)
        dot = dot / (cap_lens + 1e-6)
        dot = torch.transpose(dot, 0, 1)
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 0)

    return similarities


def xattn_score_i2t(images, obj_nums, captions, cap_lens):
    """
    Images: (batch_size, max_n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    obj_nums:(n_obj) list of obj num per image
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)

    obj_nums = torch.tensor(obj_nums, dtype=images.dtype)
    obj_nums = Variable(obj_nums)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        if n_word == 0:
            cap_i = captions[i, :, :].unsqueeze(0).contiguous()
        else:
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        cap_i_expand = cap_i_expand.contiguous()
        cap_i_expand = torch.transpose(cap_i_expand, 1, 2)
        dot = torch.bmm(images, cap_i_expand)
        # if opt.clamp:
        #     dot = torch.clamp(dot, min=0)
        dot = dot.max(dim=2, keepdim=True)[0].squeeze()
        dot = dot.sum(dim=1, keepdim=True)
        obj_nums = obj_nums.contiguous().view(-1, 1).to(device)
        dot = dot / (obj_nums + 1e-6)
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def CosineSimilarity(images_geb, captions_geb):
    similarities = sim_matrix(images_geb, captions_geb)  # n_img, n_caption
    return similarities


class ContrastiveLoss_matrix(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, predicate_score_rate=1, margin=0, max_violation=False, cross_attn='i2t'):
        super(ContrastiveLoss_matrix, self).__init__()
        self.predicate_score_rate = predicate_score_rate
        self.margin = margin
        self.max_violation = max_violation
        self.cross_attn = cross_attn

    def set_max_violation(self, bool_set):
        if bool_set is True:
            self.max_violation = True
        else:
            self.max_violation = False

    def forward(self, im, im_l, s, s_l, pred, pred_l, s_pred, s_pred_l):
        # compute image-sentence score matrix
        if self.cross_attn == 't2i':
            scores1 = xattn_score_t2i(im, im_l, s, s_l)
            scores2 = xattn_score_t2i(pred, pred_l, s_pred, s_pred_l)
            scores = scores1 + self.predicate_score_rate * scores2

        elif self.cross_attn == 'i2t':
            scores1 = xattn_score_i2t(im, im_l, s, s_l)
            scores2 = xattn_score_i2t(pred, pred_l, s_pred, s_pred_l)
            scores = scores1 + self.predicate_score_rate * scores2
        else:
            raise ValueError("unknown first norm type")

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum(), scores


class ContrastiveLoss_sim(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0., max_violation=False):
        super(ContrastiveLoss_sim, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def set_max_violation(self, bool_set):
        if bool_set is True:
            self.max_violation = True
        else:
            self.max_violation = False

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()
