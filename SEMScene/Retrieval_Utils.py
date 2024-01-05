import numpy as np
import torch
from Metrics import xattn_score_t2i, xattn_score_i2t


def shard_xattn_t2i(images, obj_nums, captions, caplens, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    d = np.zeros((len(images), len(captions)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
            for j in range(n_cap_shard):
                cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                im = torch.from_numpy(images[im_start:im_end]).cuda()
                im_l = obj_nums[im_start:im_end]
                s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
                l = caplens[cap_start:cap_end]
                sim = xattn_score_t2i(im, im_l, s, l)
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d


def shard_xattn_i2t(images, obj_nums, captions, caplens, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    d = np.zeros((len(images), len(captions)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
            for j in range(n_cap_shard):
                cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                im = torch.from_numpy(images[im_start:im_end]).cuda()
                im_l = obj_nums[im_start:im_end]
                s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
                l = caplens[cap_start:cap_end]
                sim = xattn_score_i2t(im, im_l, s, l)
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d


def i2t(images, sims, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    results = []
    for index in range(npts):
        result = dict()
        result['id'] = index
        inds = np.argsort(sims[index])[::-1]
        result['top5'] = list(inds[:5])
        result['top1'] = inds[0]
        result['top10'] = list(inds[:10])
        result['ranks'] = []
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            result['ranks'].append((i, tmp))
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

        if rank < 1:
            result['is_top1'] = 1
        else:
            result['is_top1'] = 0
        if rank < 5:
            result['is_top5'] = 1
        else:
            result['is_top5'] = 0

        results.append(result)

    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1), results
    else:
        return (r1, r5, r10, medr, meanr), results


# find images
def t2i(images, sims, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T  # ncap, nimg
    results = []
    for index in range(npts):
        for i in range(5):
            result = dict()
            result['id'] = 5 * index + i
            inds = np.argsort(sims[5 * index + i])[::-1]
            result['top5'] = list(inds[:5])
            result['top10'] = list(inds[:10])
            result['top1'] = inds[0]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

            if ranks[5 * index + i] < 1:
                result['is_top1'] = 1
            else:
                result['is_top1'] = 0

            if ranks[5 * index + i] < 5:
                result['is_top5'] = 1
            else:
                result['is_top5'] = 0
            result['ranks'] = [(index, ranks[5 * index + i])]
            results.append(result)

    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1), results
    else:
        return (r1, r5, r10, medr, meanr), results


def evalrank(img_embs, obj_nums, cap_embs, cap_lens, pred_embs, pred_nums, cap_rel_embs, cap_rel_nums,
             cross_attn='i2t', predicate_score_rate=1, image_idx2id=None, caption_idx2id=None, sim_glo=None,
             info_dict=None):
    """ image_idx2id and caption_idx2id are remained for visualization """
    print('Contrastive Score ...')
    if info_dict['model_name'].lower() == 'graph':
        sims = sim_glo
    else:
        if cross_attn == 't2i':
            sims1 = shard_xattn_t2i(img_embs, obj_nums, cap_embs, cap_lens, shard_size=128)
            sims2 = shard_xattn_t2i(pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, shard_size=128)
            sims = sims1 + predicate_score_rate * sims2
        else:
            sims1 = shard_xattn_i2t(img_embs, obj_nums, cap_embs, cap_lens, shard_size=128)
            sims2 = shard_xattn_i2t(pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, shard_size=128)
            sims = sims1 + predicate_score_rate * sims2

    if info_dict['model_name'].lower() == 'triplet':
        sims += sim_glo

    r, rt, i2t_results = i2t(img_embs, sims, return_ranks=True)
    ri, rti, t2i_results = t2i(img_embs, sims, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.4f" % rsum)
    print("Average i2t Recall: %.4f" % ar)
    print("Image to text: %.4f %.4f %.4f %.4f %.4f" % r)
    print("Average t2i Recall: %.4f" % ari)
    print("Text to image: %.4f %.4f %.4f %.4f %.4f" % ri)

    return rsum, r, ri
