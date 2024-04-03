import torch

info_dict = dict()
info_dict['save_dir'] = './Report'
info_dict['numb_sample'] = None  # training sample for 1 epoch
info_dict['Datasets'] = 'Flickr30K'  # Flickr30K, MS-COCO
info_dict['numb_epoch'] = 60  # 60 epochs for triplet; 30 epochs for node, graph
info_dict['numb_gcn_layers'] = 1  # number of gcn layers to be stacked
info_dict['gcn_hidden_dim'] = []  # hidden layer in each gcn layer
info_dict['gcn_output_dim'] = 1024  # final output dim
info_dict['gcn_input_dim'] = 2048  # dims of node and edges
info_dict['batchnorm'] = True
info_dict['batch_size'] = 128
info_dict['dropout'] = 0.4
info_dict['visual_backbone'] = 'b5'  # EfficientNet backbone to extract visual features
info_dict['visual_ft_dim'] = 2048
info_dict['optimizer'] = 'Adam'  # Adam for node, triplet; AdamW for graph
info_dict['learning_rate'] = 4e-4  # 4e-4 for node, triplet; 2e-4 for graph
info_dict['weight_decay'] = 0  # 0 for node, triplet; 1e-3 for graph 
info_dict['lr_decay_factor'] = 0.2  # 0.2 for node, triplet; 0.1 for graph
info_dict['activate_fn'] = 'swish'  # swish, relu, leakyrelu
info_dict['grad_clip'] = 2
info_dict['model_name'] = 'Node'  # Node, Triplet, Graph
info_dict['checkpoint'] = None
info_dict['margin_matrix_loss'] = 0.35  # 0.35 for node, triplet; 0.2 for graph
info_dict['rnn_numb_layers'] = 2
info_dict['rnn_bidirectional'] = True
info_dict['graph_emb_dim'] = info_dict['gcn_output_dim'] * 1
info_dict['n_heads'] = 1  # for triplet
info_dict['include_pred_ft'] = True
info_dict['pretrained'] = 0
'''Editing device here'''
device = torch.device('cuda:0')
