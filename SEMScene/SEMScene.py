import os
from tqdm.auto import tqdm
from Data_Utils import *
import Models as md
from Metrics import *
from Retrieval_Utils import *
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import time
from tensorboardX import SummaryWriter
from torch.backends import cudnn

cudnn.enabled = True
cudnn.benchmark = True

if info_dict['Datasets'].lower() == 'flickr30k':
    DATA_DIR = '../data_flickr30k/data'

    word2idx_cap = joblib.load(f"{DATA_DIR}/flickr30k_lowered_caps_word2idx_train_val.joblib")
    word2idx_img_obj = joblib.load(f"{DATA_DIR}/flickr30k_lowered_img_obj_word2idx.joblib")
    word2idx_img_pred = joblib.load(f"{DATA_DIR}/flickr30k_lowered_img_pred_word2idx.joblib")

    TOTAL_CAP_WORDS = len(word2idx_cap)
    TOTAL_IMG_OBJ = len(word2idx_img_obj)
    TOTAL_IMG_PRED = len(word2idx_img_pred)

    subset = 'train'
    images_data_train = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data_adj.joblib")
    caps_data_train = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data_adj (1).joblib")
    img_cap_matching_idx_train = joblib.load(f"{DATA_DIR}/image_caption_matching_flickr_{subset}.joblib")
    cap_img_matching_idx_train = joblib.load(f"{DATA_DIR}/caption_image_matching_flickr_{subset}.joblib")

    subset = 'val'
    images_data_val = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data_adj.joblib")
    caps_data_val = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data_adj (1).joblib")
    img_cap_matching_idx_val = joblib.load(f"{DATA_DIR}/image_caption_matching_flickr_{subset}.joblib")
    cap_img_matching_idx_val = joblib.load(f"{DATA_DIR}/caption_image_matching_flickr_{subset}.joblib")

    init_embed_model_weight_cap = joblib.load(f'{DATA_DIR}/init_glove_embedding_weight_lowered_train_val.joblib')
    init_embed_model_weight_cap = torch.FloatTensor(init_embed_model_weight_cap)
    init_embed_model_weight_img_obj = joblib.load(f'{DATA_DIR}/init_glove_embedding_weight_lowered_img_obj.joblib')
    init_embed_model_weight_img_obj = torch.FloatTensor(init_embed_model_weight_img_obj)
    init_embed_model_weight_img_pred = joblib.load(f'{DATA_DIR}/init_glove_embedding_weight_lowered_img_pred.joblib')
    init_embed_model_weight_img_pred = torch.FloatTensor(init_embed_model_weight_img_pred)

elif info_dict['Datasets'].lower() == 'ms-coco':
    DATA_DIR = '../data_mscoco/data'
    subset = 'train'
    images_data_train = joblib.load(f"{DATA_DIR}/mscoco_{subset}2014_images_data_adj.joblib")
    caps_data_train = joblib.load(f"{DATA_DIR}/mscoco_{subset}2014_caps_data_stem_adj.joblib")
    img_cap_matching_idx_train = joblib.load(f"{DATA_DIR}/image_caption_matching_mscoco_{subset}2014.joblib")
    cap_img_matching_idx_train = joblib.load(f"{DATA_DIR}/caption_image_matching_mscoco_{subset}2014.joblib")

    subset = 'val'
    images_data_val = joblib.load(f"{DATA_DIR}/mscoco_{subset}2014_images_data_adj.joblib")
    caps_data_val = joblib.load(f"{DATA_DIR}/mscoco_{subset}2014_caps_data_stem_adj.joblib")
    img_cap_matching_idx_val = joblib.load(f"{DATA_DIR}/image_caption_matching_mscoco_{subset}2014.joblib")
    cap_img_matching_idx_val = joblib.load(f"{DATA_DIR}/caption_image_matching_mscoco_{subset}2014.joblib")

    word2idx_cap_sent = joblib.load(f"{DATA_DIR}/mscoco_sgraf_caps_word2idx.joblib")
    word2idx_cap = joblib.load(f"{DATA_DIR}/mscoco_original_sgm_caps_word2idx.joblib")
    word2idx_img_obj = joblib.load(f"{DATA_DIR}/mscoco_img_obj_word2idx.joblib")
    word2idx_img_pred = joblib.load(f"{DATA_DIR}/mscoco_img_pred_word2idx.joblib")

    TOTAL_SENT_WORDS = len(word2idx_cap_sent)
    TOTAL_CAP_WORDS = len(word2idx_cap)
    TOTAL_IMG_OBJ = len(word2idx_img_obj)
    TOTAL_IMG_PRED = len(word2idx_img_pred)
    print(f'total sent words:{TOTAL_SENT_WORDS}')
    print(f'total cap words:{TOTAL_CAP_WORDS}')
    print(f'Total images objects: {TOTAL_IMG_OBJ}')
    print(f'Total images predicates: {TOTAL_IMG_PRED}')

    init_embed_model_weight_cap = joblib.load(
        f'{DATA_DIR}/mscoco_init_glove_embedding_weight_original_sgm_caps_word2idx.joblib')
    init_embed_model_weight_cap = torch.FloatTensor(init_embed_model_weight_cap)
    init_embed_model_weight_img_obj = joblib.load(
        f'{DATA_DIR}/mscoco_init_glove_embedding_weight_img_obj_word2idx.joblib')
    init_embed_model_weight_img_obj = torch.FloatTensor(init_embed_model_weight_img_obj)
    init_embed_model_weight_img_pred = joblib.load(
        f'{DATA_DIR}/mscoco_init_glove_embedding_weight_img_pred_word2idx.joblib')
    init_embed_model_weight_img_pred = torch.FloatTensor(init_embed_model_weight_img_pred)
else:
    raise ValueError("Incorrect Dataset Name!")


def print_dict(di):
    result = ''
    for key, val in di.items():
        key_upper = key.upper()
        result += f"{key_upper}: {val}\n"
    return result


class Trainer():
    def __init__(self):
        super(Trainer, self).__init__()

        self.info_dict = info_dict
        self.numb_sample = info_dict['numb_sample']
        self.numb_epoch = info_dict['numb_epoch']
        self.unit_dim = 300
        self.numb_gcn_layers = info_dict['numb_gcn_layers']
        self.gcn_hidden_dim = info_dict['gcn_hidden_dim']
        self.gcn_output_dim = info_dict['gcn_output_dim']
        self.gcn_input_dim = info_dict['gcn_input_dim']
        self.activate_fn = info_dict['activate_fn']
        self.grad_clip = info_dict['grad_clip']
        self.use_residual = False
        self.batchnorm = info_dict['batchnorm']
        self.dropout = info_dict['dropout']
        self.batch_size = info_dict['batch_size']
        self.save_dir = info_dict['save_dir']
        self.optimizer_choice = info_dict['optimizer']
        self.learning_rate = info_dict['learning_rate']
        self.weight_decay = info_dict['weight_decay']
        self.lr_decay_factor = info_dict['lr_decay_factor']
        self.model_name = info_dict['model_name']
        self.checkpoint = info_dict['checkpoint']
        self.margin_matrix_loss = info_dict['margin_matrix_loss']
        self.rnn_numb_layers = info_dict['rnn_numb_layers']
        self.rnn_bidirectional = info_dict['rnn_bidirectional']
        if info_dict['model_name'].lower() == 'graph':
            self.rnn_structure = 'GRU'
        else:
            self.rnn_structure = 'LSTM'
        self.visual_backbone = info_dict['visual_backbone']
        self.visual_ft_dim = info_dict['visual_ft_dim']
        self.ge_dim = info_dict['graph_emb_dim']
        self.include_pred_ft = info_dict['include_pred_ft']
        self.device = device
        self.n_heads = info_dict['n_heads']
        # Build datasets for training
        self.datatrain = PairGraphPrecomputeDataset(image_sgg=images_data_train,
                                                    caption_sgg=caps_data_train,
                                                    word2idx_cap=word2idx_cap,
                                                    word2idx_img_obj=word2idx_img_obj,
                                                    word2idx_img_pred=word2idx_img_pred,
                                                    effnet=self.visual_backbone,
                                                    image_caption_matching=img_cap_matching_idx_train,
                                                    caption_image_matching=cap_img_matching_idx_train,
                                                    numb_sample=self.numb_sample)

        # Declare models
        self.image_branch_model = md.ImageModel(word_unit_dim=self.unit_dim, gcn_output_dim=self.gcn_output_dim,
                                                gcn_hidden_dim=self.gcn_hidden_dim,
                                                numb_gcn_layers=self.numb_gcn_layers,
                                                batchnorm=self.batchnorm, dropout=self.dropout,
                                                activate_fn=self.activate_fn,
                                                visualft_structure=self.visual_backbone,
                                                visualft_feature_dim=self.visual_ft_dim,
                                                fusion_output_dim=self.gcn_input_dim,
                                                numb_total_obj=TOTAL_IMG_OBJ, numb_total_pred=TOTAL_IMG_PRED,
                                                init_weight_obj=init_embed_model_weight_img_obj,
                                                init_weight_pred=init_embed_model_weight_img_pred,
                                                include_pred_ft=self.include_pred_ft,
                                                network_structure=info_dict['model_name'].lower())

        self.embed_model_cap = md.WordEmbedding(numb_words=TOTAL_CAP_WORDS, embed_dim=self.unit_dim,
                                                init_weight=init_embed_model_weight_cap, sparse=False)

        self.sent_model = md.SentenceModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim,
                                           numb_layers=self.rnn_numb_layers,
                                           dropout=self.dropout, bidirectional=self.rnn_bidirectional,
                                           structure=self.rnn_structure)

        self.rels_model = md.RelsModel(input_dim=self.unit_dim, hidden_dim=self.gcn_output_dim,
                                       numb_layers=self.rnn_numb_layers,
                                       dropout=self.dropout, bidirectional=self.rnn_bidirectional,
                                       structure=self.rnn_structure)

        self.embed_model_cap = self.embed_model_cap.to(device)
        self.image_branch_model = self.image_branch_model.to(device)
        self.sent_model = self.sent_model.to(device)
        self.rels_model = self.rels_model.to(device)

        # PARAMS & OPTIMIZER
        self.params = []
        self.params += list(filter(lambda p: p.requires_grad, self.image_branch_model.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.embed_model_cap.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.sent_model.parameters()))
        self.params += list(filter(lambda p: p.requires_grad, self.rels_model.parameters()))

        if info_dict['model_name'].lower() == 'triplet':
            self.gcn_model_cap = md.GCN_Network(gcn_input_dim=self.gcn_output_dim, gcn_pred_dim=self.gcn_output_dim,
                                                gcn_output_dim=self.gcn_output_dim, gcn_hidden_dim=self.gcn_hidden_dim,
                                                numb_gcn_layers=self.numb_gcn_layers, batchnorm=self.batchnorm,
                                                dropout=self.dropout, activate_fn=self.activate_fn, use_residual=False)
            self.graph_embed_model = md.GraphEmb(node_dim=self.gcn_output_dim)
            self.semgnn_img = md.SemanticGNN_img(input_dim=self.gcn_output_dim, output_dim=self.gcn_output_dim,
                                                 device=self.device)
            self.semgnn_sent = md.SemanticGNN_sent(input_dim=self.gcn_output_dim, output_dim=self.gcn_output_dim,
                                                   device=self.device)
            self.cross_obj = md.MLP(input_dim=self.gcn_output_dim, hidden_dim=self.gcn_hidden_dim,
                                    output_dim=self.gcn_output_dim * 1,
                                    activate_fn=self.activate_fn, batchnorm=self.batchnorm, dropout=self.dropout,
                                    perform_at_end=True, use_residual=False)
            self.cross_pred = md.MLP(input_dim=self.gcn_output_dim, hidden_dim=self.gcn_hidden_dim,
                                     output_dim=self.gcn_output_dim * 1,
                                     activate_fn=self.activate_fn, batchnorm=self.batchnorm, dropout=self.dropout,
                                     perform_at_end=True, use_residual=False)
            self.transformer_obj = md.MultiHeadAttention(n_heads=self.n_heads, dropout=self.dropout)
            self.transformer_pred = md.MultiHeadAttention(n_heads=self.n_heads, dropout=self.dropout)

            self.gcn_model_cap = self.gcn_model_cap.to(device)
            self.graph_embed_model = self.graph_embed_model.to(device)
            self.semgnn_img = self.semgnn_img.to(device)
            self.semgnn_sent = self.semgnn_sent.to(device)
            self.cross_obj = self.cross_obj.to(device)
            self.cross_pred = self.cross_pred.to(device)
            self.transformer_obj = self.transformer_obj.to(device)
            self.transformer_pred = self.transformer_pred.to(device)

            self.params += list(filter(lambda p: p.requires_grad, self.gcn_model_cap.parameters()))
            self.params += list(filter(lambda p: p.requires_grad, self.graph_embed_model.parameters()))
            self.params += list(filter(lambda p: p.requires_grad, self.semgnn_img.parameters()))
            self.params += list(filter(lambda p: p.requires_grad, self.semgnn_sent.parameters()))
            self.params += list(filter(lambda p: p.requires_grad, self.cross_obj.parameters()))
            self.params += list(filter(lambda p: p.requires_grad, self.cross_pred.parameters()))
            self.params += list(filter(lambda p: p.requires_grad, self.transformer_obj.parameters()))
            self.params += list(filter(lambda p: p.requires_grad, self.transformer_pred.parameters()))

        elif info_dict['model_name'].lower() == 'graph':
            self.gpo_img = md.GPO(d_pe=32, d_hidden=32)
            self.gpo_cap = md.GPO(d_pe=32, d_hidden=32)

            self.gpo_img = self.gpo_img.to(device)
            self.gpo_cap = self.gpo_cap.to(device)

            self.params += list(filter(lambda p: p.requires_grad, self.gpo_img.parameters()))
            self.params += list(filter(lambda p: p.requires_grad, self.gpo_cap.parameters()))

        if self.optimizer_choice.lower() == 'adam':
            self.optimizer = optim.Adam(self.params,
                                        lr=self.learning_rate,
                                        betas=(0.9, 0.999),
                                        eps=1e-08,
                                        weight_decay=self.weight_decay)

        if self.optimizer_choice.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.params,
                                         lr=self.learning_rate,
                                         betas=(0.9, 0.999),
                                         weight_decay=self.weight_decay)

    def adjust_learning_rate(self):
        lr = self.learning_rate * self.lr_decay_factor
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # ---------- WRITE INFO TO TXT FILE ---------
    def extract_info(self):
        try:
            timestampLaunch = self.timestampLaunch
        except:
            timestampLaunch = 'undefined'
        model_info_log = open(f"{self.save_dir}/{self.model_name}-{timestampLaunch}-INFO.log", "w")
        result = f"===== {self.model_name} =====\n"
        result += print_dict(self.info_dict)
        model_info_log.write(result)
        model_info_log.close()

    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.embed_model_cap.load_state_dict(modelCheckpoint['embed_model_cap_state_dict'])
            self.sent_model.load_state_dict(modelCheckpoint['sent_model_state_dict'])
            self.image_branch_model.load_state_dict(modelCheckpoint['image_branch_model_state_dict'])
            self.rels_model.load_state_dict(modelCheckpoint['rels_model_state_dict'])
            if info_dict['model_name'].lower() == 'triplet':
                self.gcn_model_cap.load_state_dict(modelCheckpoint['gcn_model_cap_state_dict'])
                self.graph_embed_model.load_state_dict(modelCheckpoint['graph_embed_model_state_dict'])
                self.semgnn_img.load_state_dict(modelCheckpoint['semgnn_img_model_state_dict'])
                self.semgnn_sent.load_state_dict(modelCheckpoint['semgnn_sent_model_state_dict'])
                self.cross_obj.load_state_dict(modelCheckpoint['cross_obj_model_state_dict'])
                self.cross_pred.load_state_dict(modelCheckpoint['cross_pred_model_state_dict'])
                self.transformer_obj.load_state_dict(modelCheckpoint['transformer_obj_state_dict'])
                self.transformer_pred.load_state_dict(modelCheckpoint['transformer_pred_state_dict'])
            elif info_dict['model_name'].lower() == 'graph':
                self.gpo_img.load_state_dict(modelCheckpoint['gpo_img_state_dict'])
                self.gpo_cap.load_state_dict(modelCheckpoint['gpo_cap_state_dict'])
            self.optimizer.load_state_dict(modelCheckpoint['optimizer_state_dict'])
        else:
            print("Training from scratch")

    # ---------- RUN TRAIN ---------
    def train(self):
        # LOAD PRETRAINED MODEL #
        self.load_trained_model()

        # LOSS FUNCTION #
        loss_matrix = ContrastiveLoss_matrix(margin=self.margin_matrix_loss, predicate_score_rate=1,
                                             max_violation=True, cross_attn='t2i')
        loss_contra = ContrastiveLoss_sim(margin=self.margin_matrix_loss, max_violation=True)
        loss_matrix = loss_matrix.to(device)

        # REPORT #
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%Y%m%d")
        self.timestampLaunch = timestampDate + '-' + timestampTime
        writer = SummaryWriter(f'{self.save_dir}/{self.model_name}-{self.timestampLaunch}/')
        self.extract_info()

        # TRAIN THE NETWORK #
        lossMIN = 100000
        count_change_loss = 0
        if info_dict['model_name'].lower() == 'triplet':
            lr_adjust_epoch_id = {25, 40}
        else:
            lr_adjust_epoch_id = {15, 25}
        for epochID in range(self.numb_epoch):
            print(f"Training {epochID + 1}/{self.numb_epoch}")

            if epochID + 1 in lr_adjust_epoch_id:
                self.adjust_learning_rate()

            lossTrain = self.train_epoch(loss_matrix, loss_contra, writer, epochID + 1)

            with torch.no_grad():
                print('Validating..')
                lossVal, ar_val, ari_val = self.validate_retrieval(images_data_val, caps_data_val)

            lossVal = 6 - lossVal
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            info_txt = f"Epoch {epochID + 1}/{self.numb_epoch} [{timestampEND}]"

            if lossVal < lossMIN:
                count_change_loss = 0
                if lossVal < lossMIN:
                    lossMIN = lossVal
                if info_dict['model_name'].lower() == 'triplet':
                    torch.save({'epoch': epochID + 1,
                                'embed_model_cap_state_dict': self.embed_model_cap.state_dict(),
                                'sent_model_state_dict': self.sent_model.state_dict(),
                                'image_branch_model_state_dict': self.image_branch_model.state_dict(),
                                'rels_model_state_dict': self.rels_model.state_dict(),
                                'gcn_model_cap_state_dict': self.gcn_model_cap.state_dict(),
                                'graph_embed_model_state_dict': self.graph_embed_model.state_dict(),
                                'semgnn_img_model_state_dict': self.semgnn_img.state_dict(),
                                'semgnn_sent_model_state_dict': self.semgnn_sent.state_dict(),
                                'cross_obj_model_state_dict': self.cross_obj.state_dict(),
                                'cross_pred_model_state_dict': self.cross_pred.state_dict(),
                                'transformer_obj_state_dict': self.transformer_obj.state_dict(),
                                'transformer_pred_state_dict': self.transformer_pred.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'best_loss': lossMIN},
                               f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}.pth.tar")

                elif info_dict['model_name'].lower() == 'graph':
                    torch.save({'epoch': epochID + 1,
                                'embed_model_cap_state_dict': self.embed_model_cap.state_dict(),
                                'sent_model_state_dict': self.sent_model.state_dict(),
                                'image_branch_model_state_dict': self.image_branch_model.state_dict(),
                                'rels_model_state_dict': self.rels_model.state_dict(),
                                'gpo_img_state_dict': self.gpo_img.state_dict(),
                                'gpo_cap_state_dict': self.gpo_cap.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'best_loss': lossMIN},
                               f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}.pth.tar")

                else:
                    torch.save({'epoch': epochID + 1,
                                'embed_model_cap_state_dict': self.embed_model_cap.state_dict(),
                                'sent_model_state_dict': self.sent_model.state_dict(),
                                'image_branch_model_state_dict': self.image_branch_model.state_dict(),
                                'rels_model_state_dict': self.rels_model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'best_loss': lossMIN},
                               f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}.pth.tar")

                info_txt = info_txt + f" [SAVE]\nLoss Val: {lossVal}"
            else:
                count_change_loss += 1
                info_txt = info_txt + f"\nLoss Val: {lossVal}"
            print(info_txt)
            info_txt = info_txt + f"\n[i2t] {round(ar_val[0], 4)} {round(ar_val[1], 4)} {round(ar_val[2], 4)}"
            info_txt = info_txt + f"\n[t2i] {round(ari_val[0], 4)} {round(ari_val[1], 4)} {round(ari_val[2], 4)}"
            info_txt = info_txt + f"\nLoss Train: {round(lossTrain, 6)}\n----------\n"

            with open(f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.log", "a") as f_log:
                f_log.write(info_txt)

            writer.add_scalars('Loss Epoch', {'train': lossTrain}, epochID + 1)
            writer.add_scalars('Loss Epoch', {'val': lossVal}, epochID + 1)
            writer.add_scalars('Loss Epoch', {'val-best': lossMIN}, epochID + 1)

            current_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epochID + 1)

            if count_change_loss >= 15:
                print(f'Early stopping: {count_change_loss} epoch not decrease the loss')
                break

        writer.close()

    # ---------- TRAINING 1 EPOCH ---------
    def train_epoch(self, loss_matrix, loss_contra, writer, epochID):
        print(f"Shuffling Training Dataset")
        self.datatrain.create_pairs(seed=1509 + epochID + 100)
        dataloadertrain = make_PairGraphPrecomputeDataLoader(self.datatrain, batch_size=self.batch_size, num_workers=0)
        print(f"Done Shuffling")

        self.embed_model_cap.train()
        self.image_branch_model.train()
        self.sent_model.train()
        self.rels_model.train()
        if info_dict['model_name'].lower() == 'triplet':
            self.gcn_model_cap.train()
            self.graph_embed_model.train()
            self.semgnn_img.train()
            self.semgnn_sent.train()
            self.cross_obj.train()
            self.cross_pred.train()
            self.transformer_obj.train()
            self.transformer_pred.train()
        elif info_dict['model_name'].lower() == 'graph':
            self.gpo_img.train()
            self.gpo_cap.train()

        loss_report = 0
        count = 0
        numb_iter = len(dataloadertrain)
        print(f"Total iteration: {numb_iter}")
        with tqdm(dataloadertrain) as iterator:
            for batch in iterator:
                img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p, \
                    cap_p_o, cap_p_p, cap_p_e, cap_p_s, cap_p_numb_o, cap_p_numb_p, cap_p_len_p, cap_p_len_s, img_mask, cap_mask = batch

                img_p_o_ft = img_p_o_ft.to(device)
                img_p_p_ft = img_p_p_ft.to(device)
                img_p_o = img_p_o.to(device)
                img_p_p = img_p_p.to(device)
                img_p_e = img_p_e.to(device)

                if not self.include_pred_ft:
                    img_p_p_ft = None

                # [Image] GCN Network
                gcn_eb_img_p_o, gcn_eb_img_p_p = self.image_branch_model(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p,
                                                                         img_p_e)

                # [Caption] Padding
                pad_cap_p_s = pad_sequence(cap_p_s, batch_first=True)  # padding sentence
                pad_cap_p_p = pad_sequence(cap_p_p, batch_first=True)  # padding predicates

                # Embedding network (object, predicates in image and caption)
                # [Caption] Embed Sentence and Predicates
                eb_pad_cap_p_s = self.embed_model_cap(pad_cap_p_s.to(device))
                eb_pad_cap_p_p = self.embed_model_cap(pad_cap_p_p.to(device))

                # [Caption] Sentence Model
                batch_size, max_sent_len, input_dim = eb_pad_cap_p_s.shape
                rnn_eb_pad_cap_p_s = self.sent_model(eb_pad_cap_p_s, cap_p_len_s)  # ncap, max sent len, dim

                # Concatenating for batch processing
                rnn_eb_cap_p_rels, rnn_eb_cap_p_rels_nodes = self.rels_model(eb_pad_cap_p_p, cap_p_len_p)
                # total rels, dim

                # img_obj [batch_size, max_obj, dim], img_pred [batch_size, max_pred, dim]
                max_obj_n, max_pred_n, max_rels_n = max(img_p_numb_o), max(img_p_numb_p), max(cap_p_numb_p)
                img_emb = torch.zeros(len(img_p_numb_o), max_obj_n, gcn_eb_img_p_o.shape[1]).to(device)
                pred_emb = torch.zeros(len(img_p_numb_p), max_pred_n, gcn_eb_img_p_p.shape[1]).to(device)
                obj_offset = 0
                for i, obj_num in enumerate(img_p_numb_o):
                    img_emb[i][:obj_num, :] = gcn_eb_img_p_o[obj_offset:obj_offset + obj_num, :]
                    obj_offset += obj_num
                pred_offset = 0
                for i, pred_num in enumerate(img_p_numb_p):
                    pred_emb[i][:pred_num, :] = gcn_eb_img_p_p[pred_offset: pred_offset + pred_num, :]
                    pred_offset += pred_num

                rels_emb = torch.zeros(len(cap_p_numb_p), max_rels_n, rnn_eb_cap_p_rels.shape[1]).to(device)
                rels_offset = 0
                for i, rels_num in enumerate(cap_p_numb_p):
                    rels_emb[i][:rels_num, :] = rnn_eb_cap_p_rels[rels_offset: rels_offset + rels_num, :]
                    rels_offset += rels_num

                loss, sim = loss_matrix(img_emb, img_p_numb_o, rnn_eb_pad_cap_p_s, cap_p_len_s,
                                        pred_emb, img_p_numb_p, rels_emb, cap_p_numb_p)

                if info_dict['model_name'].lower() == 'triplet':
                    # [CAPTION] GCN
                    total_cap_p_numb_o = sum(cap_p_numb_o)
                    total_cap_p_numb_p = sum(cap_p_numb_p)
                    eb_cap_p_o = torch.zeros(total_cap_p_numb_o, self.gcn_output_dim).to(device)
                    eb_cap_p_p = torch.zeros(total_cap_p_numb_p, self.gcn_output_dim).to(device)
                    for idx in range(len(rnn_eb_cap_p_rels_nodes)):
                        edge = cap_p_e[idx]  # subject, object
                        eb_cap_p_o[edge[0]] = rnn_eb_cap_p_rels_nodes[idx, 1, :]  # <start> is 1st token
                        eb_cap_p_o[edge[1]] = rnn_eb_cap_p_rels_nodes[idx, cap_p_len_p[idx] - 2,
                                              :]  # <end> is last token
                        eb_cap_p_p[idx] = torch.mean(rnn_eb_cap_p_rels_nodes[idx, 2:(cap_p_len_p[idx] - 2), :], dim=0)
                    # [Obj EMB]
                    eb_cap_p_o, eb_cap_p_p = self.gcn_model_cap(eb_cap_p_o, eb_cap_p_p, cap_p_e)
                    image_obj_origin = self.graph_embed_model(gcn_eb_img_p_o, gcn_eb_img_p_p, img_p_numb_o,
                                                              img_p_numb_p)  # n_img, dim
                    caption_obj_origin = self.graph_embed_model(eb_cap_p_o, eb_cap_p_p, cap_p_numb_o,
                                                                cap_p_numb_p)  # n_cap, dim

                    batch_size = image_obj_origin.shape[0]
                    image_obj = self.cross_obj(image_obj_origin)
                    caption_obj = self.cross_obj(caption_obj_origin)
                    loss_cross_obj = torch.mean(torch.abs(l1norm(image_obj - caption_obj, dim=1)))

                    img_obj = self.transformer_obj(input_Q=caption_obj.reshape(batch_size, 1, -1),
                                                   input_K=img_emb,
                                                   input_V=img_emb)

                    batch_max_cap_obj = max(cap_p_numb_o)
                    eb_gcn_cap_p_o = torch.zeros([batch_size, batch_max_cap_obj, self.gcn_output_dim]).to(self.device)
                    count_cap_o = 0
                    for idx, _ in enumerate(cap_p_numb_o):
                        if _ > 0:
                            eb_gcn_cap_p_o[idx, :cap_p_numb_o[idx], :] \
                                = eb_cap_p_o[count_cap_o:count_cap_o + cap_p_numb_o[idx], :]
                            count_cap_o += cap_p_numb_o[idx]
                        else:
                            eb_gcn_cap_p_o[idx, 0, :] \
                                = torch.mean(rnn_eb_pad_cap_p_s[idx], dim=0)

                    cap_obj = self.transformer_obj(input_Q=image_obj.reshape(batch_size, 1, -1),
                                                   input_K=eb_gcn_cap_p_o,
                                                   input_V=eb_gcn_cap_p_o)

                    img_obj_origin = self.transformer_obj(input_Q=image_obj.reshape(batch_size, 1, -1),
                                                          input_K=img_emb,
                                                          input_V=img_emb)

                    cap_obj_origin = self.transformer_obj(input_Q=caption_obj.reshape(batch_size, 1, -1),
                                                          input_K=eb_gcn_cap_p_o,
                                                          input_V=eb_gcn_cap_p_o)

                    # [Tri EMB]
                    image_tri_origin = self.semgnn_img(gcn_eb_img_p_p, img_p_numb_p, img_mask)
                    sent_loc_global = torch.mean(rnn_eb_pad_cap_p_s, dim=1)
                    sent_tri_origin = self.semgnn_sent(eb_cap_p_p, cap_p_numb_p, sent_loc_global, cap_mask)

                    image_tri = self.cross_pred(image_tri_origin)
                    sent_tri = self.cross_pred(sent_tri_origin)
                    loss_cross_pred = torch.mean(torch.abs(l1norm((image_tri - sent_tri), dim=1)))
                    SC1 = loss_cross_obj + loss_cross_pred
                    img_tri = self.transformer_pred(input_Q=sent_tri.reshape(batch_size, 1, -1),
                                                    input_K=gcn_eb_img_p_p.reshape(batch_size, -1, self.gcn_output_dim),
                                                    input_V=gcn_eb_img_p_p.reshape(batch_size, -1, self.gcn_output_dim))

                    count_cap_p = 0
                    cap_p = torch.zeros([batch_size, max(cap_p_numb_p), self.gcn_output_dim]).to(self.device)
                    for idx, _ in enumerate(cap_p_numb_p):
                        if _ > 0:
                            cap_p[idx, :cap_p_numb_p[idx], :] \
                                = eb_cap_p_p[count_cap_p:count_cap_p + cap_p_numb_p[idx], :]
                            count_cap_p += cap_p_numb_p[idx]
                        else:
                            cap_p[idx, 0, :] \
                                = torch.mean(rnn_eb_pad_cap_p_s[idx], dim=0)

                    cap_tri = self.transformer_pred(input_Q=image_tri.reshape(batch_size, 1, -1),
                                                    input_K=cap_p,
                                                    input_V=cap_p)

                    img_tri_origin = self.transformer_pred(input_Q=image_tri.reshape(batch_size, 1, -1),
                                                           input_K=gcn_eb_img_p_p.reshape(batch_size, -1,
                                                                                          self.gcn_output_dim),
                                                           input_V=gcn_eb_img_p_p.reshape(batch_size, -1,
                                                                                          self.gcn_output_dim))

                    cap_tri_origin = self.transformer_pred(input_Q=sent_tri.reshape(batch_size, 1, -1),
                                                           input_K=cap_p,
                                                           input_V=cap_p)

                    sim_obj_img = CosineSimilarity(img_obj_origin, img_obj)
                    sim_obj_cap = CosineSimilarity(cap_obj_origin, cap_obj)
                    sim_sem_img = CosineSimilarity(img_tri_origin, img_tri)
                    sim_sem_cap = CosineSimilarity(cap_tri_origin, cap_tri)
                    loss_obj = (loss_contra(sim_obj_img) + loss_contra(sim_obj_cap)) / 2
                    loss_sem = (loss_contra(sim_sem_img) + loss_contra(sim_sem_cap)) / 2
                    SC2 = loss_obj + loss_sem
                    loss_geb = loss_contra(CosineSimilarity(img_obj_origin, cap_obj_origin) +
                                           CosineSimilarity(img_tri_origin, cap_tri_origin))
                    loss = loss + loss_geb + SC1 + SC2

                elif info_dict['model_name'].lower() == 'graph':

                    if epochID == 1:
                        loss_contra.set_max_violation(False)
                    else:
                        loss_contra.set_max_violation(True)
                    img_all = torch.cat([img_emb, pred_emb], dim=1)
                    img_all_len = img_p_numb_o
                    for i in range(len(img_all_len)):
                        img_all_len[i] += img_p_numb_p[i]
                    sent_all = torch.zeros(len(cap_p_numb_p), max_sent_len + max_rels_n, rnn_eb_cap_p_rels.shape[1]).to(
                        device)
                    rels_offset = 0
                    for i, rels_num in enumerate(cap_p_numb_p):
                        sent_all[i][:cap_p_len_s[i], :] = rnn_eb_pad_cap_p_s[i][:cap_p_len_s[i], :]
                        sent_all[i][cap_p_len_s[i]:cap_p_len_s[i] + rels_num, :] = rnn_eb_cap_p_rels[
                                                                                   rels_offset: rels_offset + rels_num,
                                                                                   :]
                        rels_offset += rels_num
                    sent_all_len = cap_p_len_s
                    for i in range(len(sent_all_len)):
                        sent_all_len[i] += cap_p_numb_p[i]

                    pooled_feature_image, pool_weight_img = self.gpo_img(img_all, img_all_len)
                    pooled_feature_cap, pool_weight_cap = self.gpo_cap(sent_all, sent_all_len)
                    sim = CosineSimilarity(pooled_feature_image, pooled_feature_cap)
                    loss = loss_contra(sim)

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    clip_grad_norm(self.params, self.grad_clip)
                self.optimizer.step()

                loss_report += loss.item()
                count += 1
                iterator.set_postfix(Batch_loss=loss.item(), Average_loss=loss_report / count)
                if count % 300 == 0:
                    print(
                        f"Batch Idx: {count + 1} / {len(dataloadertrain)}: Loss Matrix {round(loss_report / count, 6)}")
                    writer.add_scalars('Loss Training Iter', {'loss': loss_report / count},
                                       epochID * np.floor(numb_iter / 20) + np.floor(count / 20))

        return loss_report / count

    def encode_image_sgg(self, image_sgg, batch_size=16):
        i_dts = ImagePrecomputeDataset(image_sgg=image_sgg, word2idx_obj=word2idx_img_obj,
                                       word2idx_pred=word2idx_img_pred, effnet=self.visual_backbone, numb_sample=None)
        i_dtld = make_ImagePrecomputeDataLoader(i_dts, batch_size=batch_size, num_workers=0)

        eb_img_o_all = []
        eb_img_p_all = []
        img_numb_o_all = []
        img_numb_p_all = []
        img_numb_p_all_xx = []
        img_id_all = []
        img_o_all = []
        img_p_all = []
        img_glo_all = []
        img_mask_all = []
        self.image_branch_model.eval()
        if info_dict['model_name'].lower() == 'triplet':
            self.graph_embed_model.eval()
        with torch.no_grad():
            print('Embedding objects and predicates of images ...')
            for batchID, batch in enumerate(i_dtld):
                img_o, img_o_ft, img_p, img_p_ft, img_e, img_numb_o, img_numb_p, img_mask, image_id = batch
                img_o_ft = img_o_ft.to(device)
                img_p_ft = img_p_ft.to(device)
                img_o = img_o.to(device)
                img_p = img_p.to(device)
                img_e = img_e.to(device)
                img_id_all += image_id
                if not self.include_pred_ft:
                    img_p_ft = None

                # [Image] GCN
                gcn_eb_img_o, gcn_eb_img_p = self.image_branch_model(img_o_ft, img_p_ft, img_o, img_p, img_e)

                obj_offset = 0
                pred_offset = 0
                for idx_img in range(len(img_numb_o)):
                    eb_img_o_all.append(gcn_eb_img_o[obj_offset: (obj_offset + img_numb_o[idx_img]), :].data.cpu())
                    eb_img_p_all.append(gcn_eb_img_p[pred_offset: (pred_offset + img_numb_p[idx_img]), :].data.cpu())
                    if info_dict['model_name'].lower() == 'triplet' or info_dict['model_name'].lower() == 'graph':
                        img_o_all.append(gcn_eb_img_o[obj_offset: (obj_offset + img_numb_o[idx_img]), :])
                        img_p_all.append(gcn_eb_img_p[pred_offset: (pred_offset + img_numb_p[idx_img]), :])
                    obj_offset += img_numb_o[idx_img]
                    pred_offset += img_numb_p[idx_img]
                img_numb_o_all += img_numb_o
                img_numb_p_all += img_numb_p
                if info_dict['model_name'].lower() == 'triplet':
                    image_geb = self.graph_embed_model(gcn_eb_img_o, gcn_eb_img_p, img_numb_o, img_numb_p)
                    img_glo_all.append(image_geb)
                    img_numb_p_all_xx.append(img_numb_p)
                    img_mask_all.append(img_mask)

            img_obj_emb = pad_sequence(eb_img_o_all, batch_first=True).data.cpu().numpy()  # nimg, max obj, dim
            img_pred_emb = pad_sequence(eb_img_p_all, batch_first=True).data.cpu().numpy()  # nimg, max pred, dim
            del img_o, img_p, img_e
        return (img_obj_emb, img_pred_emb, img_numb_o_all, img_numb_p_all, img_numb_p_all_xx, img_id_all,
                img_o_all, img_p_all, img_glo_all, img_mask_all)

    def encode_caption_sgg(self, caption_sgg, batch_size=1):
        c_dts = CaptionDataset(caption_sgg=caption_sgg, word2idx=word2idx_cap, numb_sample=None)
        c_dtld = make_CaptionDataLoader(c_dts, batch_size=batch_size, num_workers=0)

        eb_cap_rels_all = []
        eb_cap_sent_all = []
        cap_numb_rels_all = []
        cap_len_sent_all = []
        cap_mask_all = []
        cap_id_all = []
        eb_cap_o_all = []
        eb_cap_numb_o_all = []
        eb_cap_p_all = []
        eb_cap_numb_p_all = []
        cap_obj_all = []

        self.embed_model_cap.eval()
        self.sent_model.eval()
        self.rels_model.eval()
        if info_dict['model_name'].lower() == 'triplet':
            self.gcn_model_cap.eval()
            self.graph_embed_model.eval()

        with torch.no_grad():
            print('Embedding captions data_flickr30k ...')
            for batchID, batch in enumerate(c_dtld):
                cap_o, cap_p, cap_e, cap_s, cap_numb_o, cap_numb_p, cap_len_p, cap_len_s, cap_mask, cap_id = batch
                cap_id_all += cap_id
                eb_cap_numb_o_all.append(cap_numb_o)
                eb_cap_numb_p_all.append(cap_numb_p)
                pad_cap_s_concate = pad_sequence(cap_s, batch_first=True).to(device)  # padding Sentence
                pad_cap_p_concate = pad_sequence(cap_p, batch_first=True).to(device)  # padding Rels

                # Embedding network (object, predicates in image and caption)
                # [Caption] Embed Sentence
                eb_pad_cap_s_concate = self.embed_model_cap(pad_cap_s_concate)
                eb_pad_cap_p_concate = self.embed_model_cap(pad_cap_p_concate)

                # [Caption] Sentence Model
                rnn_eb_pad_cap_s_concate = self.sent_model(eb_pad_cap_s_concate, cap_len_s)
                for idx_sent in range(len(cap_len_s)):
                    eb_cap_sent_all.append(rnn_eb_pad_cap_s_concate[idx_sent, 0:cap_len_s[idx_sent], :].data.cpu())

                rnn_eb_cap_rels, rnn_eb_cap_rels_nodes = self.rels_model(eb_pad_cap_p_concate, cap_len_p)

                pred_offset = 0
                for idx_cap in range(len(cap_numb_p)):
                    eb_cap_rels_all.append(
                        rnn_eb_cap_rels[pred_offset: (pred_offset + cap_numb_p[idx_cap]), :].data.cpu())
                    pred_offset += cap_numb_p[idx_cap]

                cap_numb_rels_all += cap_numb_p
                cap_len_sent_all += cap_len_s

                if info_dict['model_name'].lower() == 'triplet':
                    # [CAPTION] GCN
                    total_cap_numb_o = sum(cap_numb_o)
                    total_cap_numb_p = sum(cap_numb_p)
                    eb_cap_o = torch.zeros(total_cap_numb_o, self.gcn_output_dim).to(device)
                    eb_cap_p = torch.zeros(total_cap_numb_p, self.gcn_output_dim).to(device)
                    for idx in range(len(rnn_eb_cap_rels_nodes)):
                        edge = cap_e[idx]  # subject, object
                        eb_cap_o[edge[0]] = rnn_eb_cap_rels_nodes[idx, 1, :]  # <start> is 1st token
                        eb_cap_o[edge[1]] = rnn_eb_cap_rels_nodes[idx, cap_len_p[idx] - 2, :]  # <end> is last token
                        eb_cap_p[idx] = torch.mean(rnn_eb_cap_rels_nodes[idx, 2:(cap_len_p[idx] - 2), :], dim=0)

                    eb_cap_o, eb_cap_p = self.gcn_model_cap(eb_cap_o, eb_cap_p, cap_e)
                    eb_cap_o_all.append(eb_cap_o)
                    caption_geb = self.graph_embed_model(eb_cap_o, eb_cap_p, cap_numb_o, cap_numb_p)  # n_cap, dim
                    cap_obj_all.append(caption_geb)

                    eb_cap_p_all.append(eb_cap_p)
                    cap_mask_all.append(cap_mask)

            if info_dict['model_name'].lower() == 'graph':
                cap_sent_emb = pad_sequence(eb_cap_sent_all, batch_first=True)
                cap_rels_emb = pad_sequence(eb_cap_rels_all, batch_first=True)
            else:
                cap_sent_emb = pad_sequence(eb_cap_sent_all, batch_first=True).data.cpu().numpy()
                cap_rels_emb = pad_sequence(eb_cap_rels_all, batch_first=True).data.cpu().numpy()

        return (cap_sent_emb, cap_rels_emb, cap_len_sent_all, cap_numb_rels_all, cap_id_all, eb_cap_o_all,
                eb_cap_numb_o_all, eb_cap_p_all, eb_cap_numb_p_all, cap_obj_all, cap_mask_all)

    # ---------- VALIDATE ---------
    def validate_retrieval(self, image_sgg, caption_sgg):

        (img_obj_emb, img_pred_emb, img_numb_o_all, img_numb_p_all, img_numb_p_all_xx, img_id_all, img_o_all, img_p_all,
         img_obj_all, img_mask_all) = self.encode_image_sgg(image_sgg, batch_size=16)

        (cap_sent_emb, cap_rels_emb, cap_len_sent_all, cap_numb_rels_all, cap_id_all, eb_cap_o_all, eb_cap_numb_o_all,
         eb_cap_p_all, eb_cap_numb_p_all, cap_obj_all, cap_mask_all) = self.encode_caption_sgg(caption_sgg,
                                                                                               batch_size=64)

        print(f'{len(img_id_all)} images', ' ', f'{len(cap_id_all)} texts')

        if info_dict['model_name'].lower() == 'triplet':
            self.cross_obj.eval()
            self.transformer_obj.eval()
            self.semgnn_img.eval()
            self.semgnn_sent.eval()
            self.cross_pred.eval()
            self.transformer_pred.eval()

            print('Obj EMB...')
            img_obj = self.cross_obj(torch.cat(img_obj_all, dim=0))
            img_obj = self.transformer_obj(input_Q=img_obj.reshape(len(img_id_all), 1, -1),
                                           input_K=torch.cat(img_o_all, dim=0).reshape(len(img_id_all), 36, -1),
                                           input_V=torch.cat(img_o_all, dim=0).reshape(len(img_id_all), 36, -1))

            batch_max_cap_obj = max(torch.cat(eb_cap_numb_o_all, dim=0))
            eb_gcn_cap_p_o = torch.zeros([len(cap_id_all), batch_max_cap_obj, self.gcn_output_dim]).to(self.device)
            count_cap_o = 0
            cap_p_numb = torch.cat(eb_cap_numb_o_all, dim=0)
            cap_sent_emb_np = torch.from_numpy(cap_sent_emb)
            for idx, _ in enumerate(cap_p_numb):
                if _ > 0:
                    eb_gcn_cap_p_o[idx, :_, :] \
                        = torch.cat(eb_cap_o_all, dim=0)[count_cap_o:count_cap_o + _, :]
                    count_cap_o += _
                else:
                    eb_gcn_cap_p_o[idx, 0, :] \
                        = torch.mean(cap_sent_emb_np[idx], dim=0)

            cap_obj = self.cross_obj(torch.cat(cap_obj_all, dim=0))
            cap_obj = self.transformer_obj(input_Q=cap_obj.reshape(len(cap_id_all), 1, -1),
                                           input_K=eb_gcn_cap_p_o,
                                           input_V=eb_gcn_cap_p_o)

            print(len(img_id_all), len(cap_id_all))
            max_num = 0
            cap_mask_len = 0
            for cap in cap_mask_all:
                max_num = max(max_num, cap.shape[1])
                cap_mask_len += cap.shape[0]

            cap_mask = torch.zeros([cap_mask_len, max_num, max_num])
            count = 0
            for cap in cap_mask_all:
                cap_mask[count:count + cap.shape[0], :cap.shape[1], :cap.shape[1]] = cap
                count += cap.shape[0]
            del cap_mask_all

            print('Tri EMB...')

            img_tri = self.semgnn_img(torch.cat(img_p_all, dim=0), torch.cat(img_numb_p_all_xx, dim=0),
                                      torch.cat(img_mask_all, dim=0).to(device))
            cap_mean = torch.mean(torch.from_numpy(cap_sent_emb), dim=1)
            cap_tri = self.semgnn_sent(torch.cat(eb_cap_p_all, dim=0), torch.cat(eb_cap_numb_p_all, dim=0), cap_mean,
                                       cap_mask.to(device))
            img_tri = self.cross_pred(img_tri)
            img_tri = self.transformer_pred(input_Q=img_tri.reshape(len(img_id_all), 1, -1),
                                            input_K=torch.cat(img_p_all, dim=0).reshape(len(img_id_all), 25, -1),
                                            input_V=torch.cat(img_p_all, dim=0).reshape(len(img_id_all), 25, -1))

            batch_max_cap_pred = max(torch.cat(eb_cap_numb_p_all, dim=0))
            eb_gcn_cap_p_p = torch.zeros([len(cap_id_all), batch_max_cap_pred, self.gcn_output_dim]).to(self.device)
            count_cap_p = 0

            eb_cap_p_numb = torch.cat(eb_cap_numb_p_all, dim=0)
            for idx, cap_p_numb in enumerate(eb_cap_p_numb):
                if cap_p_numb > 0:
                    eb_gcn_cap_p_p[idx, :cap_p_numb, :] \
                        = torch.cat(eb_cap_p_all, dim=0)[count_cap_p:count_cap_p + cap_p_numb, :]
                    count_cap_p += cap_p_numb
                else:
                    eb_gcn_cap_p_p[idx, 0, :] \
                        = torch.mean(cap_sent_emb_np[idx], dim=0)

            cap_tri = self.cross_pred(cap_tri)
            cap_tri = self.transformer_pred(input_Q=cap_tri.reshape(len(cap_id_all), 1, -1),
                                            input_K=eb_gcn_cap_p_p,
                                            input_V=eb_gcn_cap_p_p)

            sim_tri = (CosineSimilarity(img_obj.reshape(len(img_id_all), -1),
                                        cap_obj.reshape(len(cap_id_all), -1)) +
                       CosineSimilarity(img_tri.reshape(len(img_id_all), -1),
                                        cap_tri.reshape(len(cap_id_all), -1))) / 2

            with torch.no_grad():
                score, ar, ari = evalrank(img_obj_emb, img_numb_o_all, cap_sent_emb, cap_len_sent_all,
                                          img_pred_emb, img_numb_p_all, cap_rels_emb, cap_numb_rels_all,
                                          cross_attn='t2i', predicate_score_rate=1,
                                          image_idx2id=img_id_all,
                                          caption_idx2id=cap_id_all,
                                          sim_glo=sim_tri.data.cpu().numpy(),
                                          info_dict=info_dict)

        elif info_dict['model_name'].lower() == 'graph':
            self.gpo_img.eval()
            self.gpo_cap.eval()

            img_o = torch.cat(img_o_all, dim=0).reshape(len(img_id_all), 36, -1)
            img_p = torch.cat(img_p_all, dim=0).reshape(len(img_id_all), 25, -1)
            img_all = torch.cat([img_o, img_p], dim=1)
            del img_o, img_p

            img_all_len = img_numb_o_all
            for i in range(len(img_all_len)):
                img_all_len[i] += img_numb_p_all[i]

            cap_all = torch.zeros(len(cap_len_sent_all), max(cap_len_sent_all) + max(cap_numb_rels_all),
                                  cap_sent_emb.shape[2]).to(device)
            for i in range(len(cap_len_sent_all)):
                cap_all[i][:cap_len_sent_all[i], :] = cap_sent_emb[i][:cap_len_sent_all[i], :]
                if cap_numb_rels_all[i] == 0:
                    continue
                else:
                    cap_all[i][cap_len_sent_all[i]:
                               cap_len_sent_all[i] + cap_numb_rels_all[i], :] = cap_rels_emb[i][:
                                                                                                cap_numb_rels_all[i], :]
            cap_num_all = cap_len_sent_all
            for i in range(len(cap_num_all)):
                cap_num_all[i] += cap_numb_rels_all[i]

            pooled_feature_image, pool_weight_img = self.gpo_img(img_all, img_all_len)
            pooled_feature_cap, pool_weight_cap = self.gpo_cap(cap_all, cap_num_all)

            sim_gra = CosineSimilarity(pooled_feature_image, pooled_feature_cap)
            with torch.no_grad():
                score, ar, ari = evalrank(img_obj_emb, img_numb_o_all, cap_sent_emb, cap_len_sent_all,
                                          img_pred_emb, img_numb_p_all, cap_rels_emb, cap_numb_rels_all,
                                          cross_attn='t2i', predicate_score_rate=1,
                                          image_idx2id=img_id_all,
                                          caption_idx2id=cap_id_all,
                                          sim_glo=sim_gra.data.cpu().numpy(),
                                          info_dict=info_dict)
        else:
            with torch.no_grad():
                score, ar, ari = evalrank(img_obj_emb, img_numb_o_all, cap_sent_emb, cap_len_sent_all,
                                          img_pred_emb, img_numb_p_all, cap_rels_emb, cap_numb_rels_all,
                                          cross_attn='t2i', predicate_score_rate=1,
                                          image_idx2id=img_id_all,
                                          caption_idx2id=cap_id_all,
                                          sim_glo=None,
                                          info_dict=info_dict)

        return score, ar, ari


def run():
    if not os.path.exists(info_dict['save_dir']):
        print(f"Creating {info_dict['save_dir']} folder")
        os.makedirs(info_dict['save_dir'])

    trainer = Trainer()
    trainer.train()  # To make direct testing, delete this statement

    trainer.load_trained_model()
    subset = 'test'
    if info_dict['Datasets'].lower() == 'flickr30k':
        DATA_DIR = '../data_flickr30k/data'
        images_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data_adj.joblib")
        caps_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data_adj (1).joblib")
    elif info_dict['Datasets'].lower() == 'ms-coco':
        DATA_DIR = '../data_mscoco/data'
        images_data = joblib.load(f"{DATA_DIR}/mscoco_{subset}2014_images_data_adj.joblib")
        caps_data = joblib.load(f"{DATA_DIR}/mscoco_{subset}2014_caps_data_stem_adj.joblib")
    else:
        raise ValueError("Incorrect Dataset Name!")

    lossVal, ar_val, ari_val = trainer.validate_retrieval(images_data, caps_data)
    info_txt = f"\n----- SUMMARY (Matrix)-----\nLoss Val: {6 - lossVal}"
    info_txt = info_txt + f"\n[i2t] {round(ar_val[0], 4)} {round(ar_val[1], 4)} {round(ar_val[2], 4)}"
    info_txt = info_txt + f"\n[t2i] {round(ari_val[0], 4)} {round(ari_val[1], 4)} {round(ari_val[2], 4)}"
    print(info_txt)


run()
