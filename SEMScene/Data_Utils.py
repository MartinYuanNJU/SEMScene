import random
import numpy as np
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from Configuration import info_dict

if info_dict['Datasets'].lower() == 'flickr30k':
    OBJ_FT_DIR = '../data_flickr30k/image/object'  # run extract_visual_features.py to get this
    PRED_FT_DIR = '../data_flickr30k/image/relationship'  # run extract_visual_features.py to get this
elif info_dict['Datasets'].lower() == 'ms-coco':
    OBJ_FT_DIR = '../data_mscoco/image/object'
    PRED_FT_DIR = '../data_mscoco/image/relationship'
else:
    raise ValueError("Incorrect Dataset Name!")


def indexing_sent(sent, word2idx, add_start_end=True):
    words = word_tokenize(sent)
    if add_start_end:
        words = ['<start>'] + words + ['<end>']
    words_idx = []
    for word in words:
        try:
            idx = word2idx[word]
        except:
            idx = word2idx['<unk>']
        words_idx.append(idx)
    return words_idx


def indexing_rels(rels, word2idx, add_start_end=True):
    rels_idx = []
    for rel in rels:
        for idx, word in enumerate(rel):
            if ':' in word:  # rels in sentence has ":"
                word = word.split(':')[0]
                rel[idx] = word
        rel = ' '.join(rel)
        rel = word_tokenize(rel)
        if add_start_end:
            rel = ['<start>'] + rel + ['<end>']
        rel_idx = []
        for word in rel:
            try:
                idx = word2idx[word]
            except:
                idx = word2idx['<unk>']
            rel_idx.append(idx)
        rels_idx.append(rel_idx)
    return rels_idx


def encode_image_sgg_to_matrix(sgg, word2idx_obj, word2idx_pred):
    """
    sgg is dict with rels, bbox, and labels
    word2idx dictionary to encode word into numeric
    Return obj, pred, and edge matrix in which
    obj = [n_obj, 1] indicating index of object in obj_to_idx --> will pass to embedding
    pred = [n_pred, 1] indicating index of predicate in pred_to_idx --> will pass to embedding
    edge = [n_pred, 2] indicating the relations between objects where edge[k] = [i,j] = [obj[i], pred[k], obj[j]] relations
    """

    obj_np = []
    pred_np = []
    edge_np = []

    try:
        sgg_rels = sgg['rels']
    except:
        sgg_rels = sgg['sgg']
    try:
        sgg_labels = sgg['labels']
    except:
        sgg_labels = sgg['bbox']['labels']

    for idx, obj in enumerate(sgg_labels):
        label_to_idx = word2idx_obj[obj]
        obj_np.append(label_to_idx)

    for idx, rel in enumerate(sgg_rels):
        sub_pos = rel[0].split(':')[1]
        pred_label = rel[1]
        obj_pos = rel[2].split(':')[1]
        label_to_idx = word2idx_pred[pred_label]
        pred_np.append(label_to_idx)
        edge_np.append([int(sub_pos), int(obj_pos)])

    obj_np = np.asarray(obj_np, dtype=int)
    pred_np = np.asarray(pred_np, dtype=int)
    edge_np = np.asarray(edge_np, dtype=int)

    return obj_np, pred_np, edge_np


def encode_caption_sgg_to_matrix(sgg, word2idx):
    """
    sgg is dictionary with sent and rels
    sent and rels are lemmatised already
    Return obj, pred, and edge matrix in which
    obj = [n_obj, ] indicating index of object in obj_to_idx --> will pass to embedding
    pred = [n_pred, ] indicating index of predicate in pred_to_idx --> will pass to embedding
    edge = [n_pred, 2] indicating the relations between objects where edge[k] = [i,j] = [obj[i], pred[k], obj[j]] relations
    sent_to_idx: encoded sentence with <start> and <end> token
    """

    obj_np = []
    edge_np = []

    sent_to_idx = indexing_sent(sent=sgg['sent'], word2idx=word2idx, add_start_end=True)  # list

    labels = [x[0] for x in sgg['rels']] + [x[2] for x in sgg['rels']]
    labels = np.unique(np.asarray(labels)).tolist()

    for idx, obj in enumerate(labels):
        try:
            label_to_idx = word2idx[obj]
        except:
            label_to_idx = word2idx['<unk>']
        obj_np.append(label_to_idx)

    for idx, rel in enumerate(sgg['rels']):
        sub, pred_label, obj = rel[0], rel[1], rel[2]
        sub_pos = labels.index(sub)
        obj_pos = labels.index(obj)
        edge_np.append([int(sub_pos), int(obj_pos)])

    pred_np = indexing_rels(rels=sgg['rels'], word2idx=word2idx, add_start_end=True)  # list of list
    # pred: [<start> , sub , pred, obj, <end>]
    len_pred = [len(x) for x in pred_np]  # len of a pred <start> sub, pred (can be multiple words), obj <end>
    obj_np = np.asarray(obj_np, dtype=int)
    edge_np = np.asarray(edge_np, dtype=int)

    return obj_np, pred_np, edge_np, len_pred, sent_to_idx  # obj and edge is numpy array, other is list


'''IMAGE-CAPTION TRIPLET DATASET, for training'''


class PairGraphPrecomputeDataset(Dataset):
    """
    Generate pair of graphs which from image and caption
    """

    def __init__(self, image_sgg, caption_sgg, image_caption_matching, caption_image_matching, word2idx_cap,
                 word2idx_img_obj, word2idx_img_pred, effnet='b0', numb_sample=None, obj_ft_dir=OBJ_FT_DIR,
                 pred_ft_dir=PRED_FT_DIR):
        """
        image_sgg: dictionary of scene graph from images with format image_sgg[image_id]['rels'] and image_sgg[image_id]['labels']
        caption_sgg: dictionary of scene graph from captions with format caption_sgg[cap_id]['rels'] and caption_sgg[cap_id]['sent']
        Note that caption_sgg and image_sgg are all lemmatised
        image_caption_matching: dictionary describes which image matches which caption with format image_caption_matching[image_id] = [cap_id_1, cap_id_2, ...]
        caption_image_matching: reverse dictionary of above caption_image_matching[cap_id] = image_id
        word2idx: dictionary to map words into index for learning embedding
        numb_sample: int indicating number of sample in the dataset
        """

        self.OBJ_FT_DIR = obj_ft_dir
        self.PRED_FT_DIR = pred_ft_dir
        self.effnet = effnet
        self.image_sgg = image_sgg
        self.caption_sgg = caption_sgg
        self.image_caption_matching = image_caption_matching
        self.caption_image_matching = caption_image_matching
        self.list_image_id = list(self.image_caption_matching.keys())
        self.list_caption_id = list(self.caption_image_matching.keys())
        self.numb_sample = numb_sample
        self.word2idx_cap = word2idx_cap
        self.word2idx_img_obj = word2idx_img_obj
        self.word2idx_img_pred = word2idx_img_pred
        self.list_match_pairs = []
        for caption_id in self.list_caption_id:
            image_id = self.caption_image_matching[caption_id]
            self.list_match_pairs.append((image_id, caption_id))
        self.numb_pairs = len(self.list_match_pairs)

        if self.numb_sample is None:
            self.numb_sample = self.numb_pairs

    def create_pairs(self, seed=1509):  # Have to run this function at the beginning of every epoch
        # Shuffle Item
        random.seed(seed)
        print('Creating Pairs of Graphs ...')
        sample_match = self.list_match_pairs.copy()
        if self.numb_sample <= self.numb_pairs:
            random.shuffle(sample_match)
            sample_match = sample_match[0:self.numb_sample]
        else:
            numb_gen = self.numb_sample - self.numb_pairs
            pairs_gen = random.choices(self.list_match_pairs, k=numb_gen)
            sample_match = sample_match + pairs_gen
            random.shuffle(sample_match)
        self.samples = sample_match

    def __getitem__(self, i):
        # Get item
        sample = self.samples[i]
        imgid, capid = sample

        try:
            img_obj_np, img_pred_np, img_edge_np = encode_image_sgg_to_matrix(sgg=self.image_sgg[imgid],
                                                                              word2idx_obj=self.word2idx_img_obj,
                                                                              word2idx_pred=self.word2idx_img_pred)
            cap_obj_np, cap_pred_np, cap_edge_np, cap_len_pred, cap_sent_np = encode_caption_sgg_to_matrix(
                sgg=self.caption_sgg[capid], word2idx=self.word2idx_cap)
        except Exception as e:
            print(e)
            print(f"Error in {sample}")

        result = dict()
        result['image'] = dict()
        result['caption'] = dict()

        # All is numpy array
        result['image']['object'] = img_obj_np
        result['image']['predicate'] = img_pred_np
        result['image']['edge'] = img_edge_np
        result['image']['numb_obj'] = len(img_obj_np)
        result['image']['numb_pred'] = len(img_pred_np)
        result['image']['id'] = imgid
        result['image']['object_ft'] = torch.tensor(
            joblib.load(f"{self.OBJ_FT_DIR}/{imgid[:-4]}.joblib"))  # n_obj, ft_dim
        result['image']['pred_ft'] = torch.tensor(
            joblib.load(f"{self.PRED_FT_DIR}/{imgid[:-4]}.joblib"))  # n_obj, ft_dim
        # All is list
        result['caption']['object'] = cap_obj_np  # [numpy array (numb obj)]
        result['caption']['predicate'] = cap_pred_np  # [list of list]
        result['caption']['edge'] = cap_edge_np  # [numpy array (numb_pred, 2)]
        result['caption']['sent'] = cap_sent_np  # [list]
        result['caption']['numb_obj'] = len(cap_obj_np)  # [scalar]
        result['caption']['len_pred'] = cap_len_pred  # len of each predicate in a caption [list]
        result['caption']['numb_pred'] = len(cap_pred_np)  # number of predicate in a caption [scalar]
        result['caption']['id'] = capid
        # result['caption']['sgg'] = self.caption_sgg[sample[1]] # for debug
        # result['image']['sgg'] = self.image_sgg[sample[0]] # for debug

        result['image']['adj'] = torch.from_numpy(self.image_sgg[imgid]['adj']).long()
        result['caption']['adj'] = torch.from_numpy(self.caption_sgg[capid]['adj']).long()
        return result

    def __len__(self):
        return len(self.samples)


# Collate function for preprocessing batch in dataloader
def pair_precompute_collate_fn(batch):
    """
    image obj, pred, edge is tensor
    others is list
    """
    image_obj = np.array([])
    image_pred = np.array([])
    image_edge = np.array([])
    image_numb_obj = []
    image_numb_pred = []
    image_obj_offset = 0
    image_obj_ft = []
    image_pred_ft = []

    caption_obj = np.array([])
    caption_pred = []
    caption_edge = np.array([])
    caption_numb_obj = []
    caption_numb_pred = []
    caption_len_pred = []
    caption_sent = []
    caption_len_sent = []
    caption_obj_offset = 0

    caption_id = []  # for debug
    image_id = []  # for debug

    img_adj = []
    cap_adj = []
    batch_len = len(batch)
    for ba in batch:
        image_obj = np.append(image_obj, ba['image']['object'])
        image_pred = np.append(image_pred, ba['image']['predicate'])
        for idx_row in range(ba['image']['edge'].shape[0]):
            edge = ba['image']['edge'][idx_row] + image_obj_offset
            image_edge = np.append(image_edge, edge)
        image_obj_offset += ba['image']['numb_obj']
        image_numb_obj += [ba['image']['numb_obj']]
        image_numb_pred += [ba['image']['numb_pred']]
        image_obj_ft.append(ba['image']['object_ft'])
        image_pred_ft.append(ba['image']['pred_ft'])

        # Caption SGG
        caption_obj = np.append(caption_obj, ba['caption']['object'])
        for idx_row in range(ba['caption']['edge'].shape[0]):
            edge = ba['caption']['edge'][idx_row] + caption_obj_offset
            caption_pred += [torch.LongTensor(ba['caption']['predicate'][idx_row])]
            caption_edge = np.append(caption_edge, edge)
        caption_obj_offset += ba['caption']['numb_obj']
        caption_numb_obj += [ba['caption']['numb_obj']]
        caption_numb_pred += [ba['caption']['numb_pred']]
        caption_sent += [torch.LongTensor(ba['caption']['sent'])]
        caption_len_sent += [len(ba['caption']['sent'])]
        caption_len_pred += ba['caption']['len_pred']

        image_id += [ba['image']['id']]
        caption_id += [ba['caption']['id']]
        img_adj.append(ba['image']['adj'])
        cap_adj.append(ba['caption']['adj'])

    image_edge = image_edge.reshape(-1, 2)
    caption_edge = caption_edge.reshape(-1, 2)
    image_obj = torch.LongTensor(image_obj)
    image_pred = torch.LongTensor(image_pred)
    image_edge = torch.LongTensor(image_edge)

    caption_obj = torch.LongTensor(caption_obj)
    caption_edge = torch.LongTensor(caption_edge)

    image_obj_ft = torch.cat(image_obj_ft, dim=0)  # tensor [total_obj, dim]
    image_pred_ft = torch.cat(image_pred_ft, dim=0)  # tensor [total_pred, dim]

    assert image_edge.shape[0] == image_pred.shape[0]
    assert caption_edge.shape[0] == sum(caption_numb_pred)

    img_mask = torch.stack(img_adj)
    max_length = max(caption_numb_pred)
    cap_mask = torch.zeros([batch_len, max_length, max_length])

    for i in range(batch_len):
        temp_num = caption_numb_pred[i]
        if temp_num == 0:
            cap_mask[i, :, :] = torch.eye(max_length)
        else:
            cap_mask[i, :temp_num, :temp_num] = cap_adj[i]

    return image_obj, image_obj_ft, image_pred, image_pred_ft, image_edge, image_numb_obj, image_numb_pred, \
        caption_obj, caption_pred, caption_edge, caption_sent, \
        caption_numb_obj, caption_numb_pred, caption_len_pred, caption_len_sent, img_mask, cap_mask


def make_PairGraphPrecomputeDataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pair_precompute_collate_fn,
                            pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader


'''The following code is for validating and testing'''


# ====== IMAGE DATASET ======
# Only use for validating entire dataset
# Generate image sgg dataset only
class ImagePrecomputeDataset(Dataset):
    def __init__(self, image_sgg, word2idx_obj, word2idx_pred, effnet='b0', numb_sample=None, obj_ft_dir=OBJ_FT_DIR,
                 pred_ft_dir=PRED_FT_DIR):
        self.OBJ_FT_DIR = obj_ft_dir
        self.PRED_FT_DIR = pred_ft_dir
        self.effnet = effnet
        self.image_sgg = image_sgg
        self.list_image_id = list(self.image_sgg.keys())
        self.numb_sample = numb_sample
        self.word2idx_obj = word2idx_obj
        self.word2idx_pred = word2idx_pred
        if self.numb_sample is None or self.numb_sample <= 0 or self.numb_sample > len(self.image_sgg):
            self.numb_sample = len(self.image_sgg)
            assert self.numb_sample == len(self.list_image_id)

    def __len__(self):
        return self.numb_sample

    def __getitem__(self, idx):
        image_id = self.list_image_id[idx]
        img_obj_np, img_pred_np, img_edge_np = encode_image_sgg_to_matrix(sgg=self.image_sgg[image_id],
                                                                          word2idx_obj=self.word2idx_obj,
                                                                          word2idx_pred=self.word2idx_pred)
        result = dict()
        result['id'] = image_id
        result['object'] = img_obj_np
        result['predicate'] = img_pred_np
        result['edge'] = img_edge_np
        result['numb_obj'] = len(img_obj_np)
        result['numb_pred'] = len(img_pred_np)
        result['object_ft'] = torch.tensor(joblib.load(f"{self.OBJ_FT_DIR}/{image_id[:-4]}.joblib"))  # n_obj, ft_dim
        result['pred_ft'] = torch.tensor(joblib.load(f"{self.PRED_FT_DIR}/{image_id[:-4]}.joblib"))  # n_pred, ft_dim
        result['adj'] = torch.from_numpy(self.image_sgg[image_id]['adj']).long()
        return result


def image_precompute_collate_fn(batch):
    image_obj = np.array([])
    image_pred = np.array([])
    image_edge = np.array([])
    image_numb_obj = np.array([])
    image_numb_pred = np.array([])
    image_obj_offset = 0
    image_obj_ft = []
    image_pred_ft = []
    image_id = []
    img_adj = []
    for ba in batch:
        image_obj = np.append(image_obj, ba['object'])
        image_pred = np.append(image_pred, ba['predicate'])
        for idx_row in range(ba['edge'].shape[0]):
            edge = ba['edge'][idx_row] + image_obj_offset
            image_edge = np.append(image_edge, edge)
        image_obj_offset += ba['numb_obj']
        image_numb_obj = np.append(image_numb_obj, ba['numb_obj'])
        image_numb_pred = np.append(image_numb_pred, ba['numb_pred'])
        image_obj_ft.append(ba['object_ft'])
        image_pred_ft.append(ba['pred_ft'])
        image_id += [ba['id']]
        img_adj.append(ba['adj'])
    image_edge = image_edge.reshape(-1, 2)

    image_obj = torch.LongTensor(image_obj)
    image_pred = torch.LongTensor(image_pred)
    image_edge = torch.LongTensor(image_edge)
    image_numb_obj = torch.LongTensor(image_numb_obj)
    image_numb_pred = torch.LongTensor(image_numb_pred)

    image_obj_ft = torch.cat(image_obj_ft, dim=0)
    image_pred_ft = torch.cat(image_pred_ft, dim=0)
    img_mask = torch.stack(img_adj)

    return image_obj, image_obj_ft, image_pred, image_pred_ft, image_edge, image_numb_obj, image_numb_pred, img_mask, \
        image_id


def make_ImagePrecomputeDataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=image_precompute_collate_fn,
                            pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dataloader


# ====== CAPTION DATASET ======
# Only use for validating entire dataset
# Generate caption sgg dataset only (sentence + sgg)
class CaptionDataset(Dataset):
    def __init__(self, caption_sgg, word2idx, numb_sample=None):
        # Do something
        self.caption_sgg = caption_sgg
        self.list_caption_id = list(self.caption_sgg.keys())
        self.numb_sample = numb_sample
        self.word2idx = word2idx
        if self.numb_sample is None or self.numb_sample <= 0 or self.numb_sample > len(self.caption_sgg):
            self.numb_sample = len(self.caption_sgg)
            assert self.numb_sample == len(self.list_caption_id)

    def __len__(self):
        return self.numb_sample

    def __getitem__(self, idx):
        caption_id = self.list_caption_id[idx]
        cap_obj_np, cap_pred_np, cap_edge_np, cap_len_pred, cap_sent_np = encode_caption_sgg_to_matrix(
            sgg=self.caption_sgg[caption_id], word2idx=self.word2idx)

        result = dict()
        result['id'] = caption_id
        result['object'] = cap_obj_np
        result['predicate'] = cap_pred_np
        result['edge'] = cap_edge_np
        result['sent'] = cap_sent_np
        result['numb_obj'] = len(cap_obj_np)
        result['numb_pred'] = len(cap_pred_np)
        result['len_pred'] = cap_len_pred
        result['adj'] = torch.from_numpy(self.caption_sgg[caption_id]['adj']).long()

        return result


def caption_collate_fn(batch):
    caption_obj = np.array([])
    caption_pred = []
    caption_edge = np.array([])
    caption_numb_obj = []
    caption_numb_pred = []
    caption_sent = []
    caption_len_sent = []
    caption_len_pred = []
    cap_id = []
    caption_obj_offset = 0
    cap_adj = []
    batch_len = len(batch)
    for ba in batch:
        caption_obj = np.append(caption_obj, ba['object'])
        for idx_row in range(ba['edge'].shape[0]):
            edge = ba['edge'][idx_row] + caption_obj_offset
            caption_edge = np.append(caption_edge, edge)
            caption_pred += [torch.LongTensor(ba['predicate'][idx_row])]

        caption_obj_offset += ba['numb_obj']
        caption_numb_obj += [ba['numb_obj']]
        caption_numb_pred += [ba['numb_pred']]
        caption_sent += [torch.LongTensor(ba['sent'])]
        caption_len_sent += [len(ba['sent'])]
        caption_len_pred += ba['len_pred']
        cap_adj.append(ba['adj'])
        cap_id.append(ba['id'])
    caption_edge = caption_edge.reshape(-1, 2)

    caption_obj = torch.LongTensor(caption_obj)
    caption_edge = torch.LongTensor(caption_edge)
    caption_numb_obj = torch.LongTensor(caption_numb_obj)
    caption_numb_pred = torch.LongTensor(caption_numb_pred)

    max_length = max(caption_numb_pred)
    cap_mask = torch.zeros([batch_len, max_length, max_length])

    for i in range(batch_len):
        temp_num = caption_numb_pred[i]
        if temp_num == 0:
            cap_mask[i, :, :] = torch.eye(max_length)
        else:
            cap_mask[i, :temp_num, :temp_num] = cap_adj[i]

    return caption_obj, caption_pred, caption_edge, caption_sent, \
        caption_numb_obj, caption_numb_pred, caption_len_pred, caption_len_sent, cap_mask, cap_id


def make_CaptionDataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=caption_collate_fn, pin_memory=pin_memory,
                            num_workers=num_workers, shuffle=shuffle)
    return dataloader
