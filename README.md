# SEMScene: Semantic-Consistency Enhanced Multi-Level Scene Graph Matching for Image-Text Retrieval
PyTorch implementation for SEMScene model. SEMScene is a scene-graph based image-text retrieval method. The research paper of SEMScene model has been published on *ACM Transactions on Multimedia Computing, Communications, and Applications (ACM TOMM)* (Volume 20, Issue 8, August 2024) entitled "SEMScene: Semantic-Consistency Enhanced Multi-Level Scene Graph Matching for Image-Text Retrieval".

# Requirements
For all used packages in the model, please refer to the ```requirements.txt```. The Python version is 3.11.4.
# Data 
Except for the uploaded basic data in this repository, the model still need basic data including adjacency matrix based on the connections of predicates and triplets of sentence extracted through leveraging the [SceneGraphParser](https://github.com/vacancy/SceneGraphParser), which can be obtained here: [flickr30k](https://drive.google.com/drive/folders/1W02ub0UtV6wE41v59qa9pfxArKGIMtvv?usp=drive_link) and [mscoco](https://drive.google.com/drive/folders/1c0NpqlR0PypWO2JT3FnRrvKDfZa5M0Wm?usp=drive_link). Please download and place them in the **data_flickr30k/data** and **data_mscoco/data** folders, respectively. Or you can extract them by editing the paths of original files in ```extract_pred_adj.py``` and ```sng_parser_process.ipynb```, then run them. The original files can be download from [here](https://drive.google.com/drive/folders/18OHy--6mqbmNLCeushpbKuEI6xlWSLuf?usp=drive_link). After extracting the triplets of sentence, please implement the **stemming** for them. <br/><br/>
The visual features of objects and predicates are also needed, we follow [LGSGM](https://github.com/m2man/LGSGM) to use **EfficientNet-b5** to extract these features, you can find them here: [flickr30k_visual](https://drive.google.com/drive/folders/1IvlmTZ9wUpOVIr9MzPgWZB5aYTaTD0jn) and [mscoco_visual](https://drive.google.com/drive/folders/1Q1Msy6kV0pzZ7uxrDjDQW34Ta9CucI4i), the files storing all extracted visual features of Flickr30k and MSCOCO are provided by [LGSGM](https://github.com/m2man/LGSGM), many thanks. Please download and place them in the **data_flickr30k** and **data_mscoco**, respectively.
# Training new models from scratch
Please modify the hyper-parameters in **SEMScene/Configuration.py** according to their corresponding comments, and run:
```
python SEMScene/SEMScene.py
```
# Pre-trained model and Evaluation
For limited google drive space, we temporarily upload the pretrained models of Flickr30K, they can be downloaded from [flickr30k_pretrained_model](https://drive.google.com/drive/folders/1weVZduxLwtRn5Q6TBi3n6dBwN9AiUQao?usp=drive_link). Please modify the path in the 24th row ```info_dict['checkpoint'] = None ``` of **SEMScene/Configuration.py** and delete the statement in the 935th row ```trainer.train()``` of **SEMScene/SEMScene.py**, then run the **SEMScene/SEMScene.py** for evaluation.
# Contact
For any issue or comment, you can directly email the authors at **lyk208d80@gmail.com** or **xiangyuan@stu.pku.edu.cn**.
# Reference
If you find our work helpful to your research, please cite our work as:
```
@article{10.1145/3664816,
author = {Liu, Yuankun and Yuan, Xiang and Li, Haochen and Tan, Zhijie and Huang, Jinsong and Xiao, Jingjie and Li, Weiping and Mo, Tong},
title = {SEMScene: Semantic-Consistency Enhanced Multi-Level Scene Graph Matching for Image-Text Retrieval},
year = {2024},
issue_date = {August 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {20},
number = {8},
issn = {1551-6857},
url = {https://doi.org/10.1145/3664816},
doi = {10.1145/3664816},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
month = {aug},
articleno = {261},
numpages = {28}
}
```
