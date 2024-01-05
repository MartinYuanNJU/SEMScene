import joblib
import numpy as np

DATA_DIR = '../data_mscoco/data'
subset = 'train'
images_data_train = joblib.load(f"{DATA_DIR}/mscoco_{subset}2014_images_data.joblib")
list_id = list(images_data_train.keys())

for id in list_id:
    temp = images_data_train[id]
    edge = []
    for idx, rels in enumerate(temp['rels']):
        sub_pos = rels[0].split(':')[1]
        obj_pos = rels[2].split(':')[1]
        edge.append([int(sub_pos), int(obj_pos)])
    edge_np = np.asarray(edge, dtype=int)
    # print(edge_np)
    adj = np.zeros([edge_np.shape[0], edge_np.shape[0]])
    for j in range(adj.shape[0]):
        temp = edge_np[j]
        for k in range(edge_np.shape[0]):
            temp_remain = edge_np[k]
            for x in temp_remain:
                if x in temp:
                    adj[j, k] = 1
                    continue
    images_data_train[id].setdefault('adj', adj)
joblib.dump(images_data_train, f"{DATA_DIR}/mscoco_{subset}2014_image_adj.joblib")