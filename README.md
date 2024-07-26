# SIGRATE-ViT-XAI

This is the official implementation of the paper: Generating Masks from Similarity-based Graphs for Vision Transformer Explainability.

## Abstract

Vision Transformers (ViTs) are becoming the most used architecture to address computer vision tasks due to their outstanding performances. However, the interpretation of the ViT output is still a challenging task, but it is also fundamental to gain trust in these new architectures. In this paper, we want to address this specific task by proposing SIGRATE. Specifically, SIGRATE extracts the embeddings of the image patches and builds the corresponding similarity graph for each attention layer. Then, it generates a set of walks on each similarity graph, where the walk sources are the most relevant and irrelevant patches according to the attention paid by the classification token. Each walk corresponds to the selection of a subset of patches that then generates a binary mask. After that, SIGRATE merges the masks from all the attention layers thanks to the coverage bias formula into a heatmap, which then is reshaped to the same as the input image. Similar to other visual explainability approaches, we tested SIGRATE on two ViT architectures (i.e., ViT and DeiT) and a subset of the ImageNet validation set. The experiments show the promising performances of SIGRATE in terms of Insertion and Deletion metrics. Finally, we report a qualitative analysis to show the capabilities of SIGRATE on three images and an ablation study to evaluate the hyperparameters’ impact on the SIGRATE performance.

![Multilayer creation](./Readme_images/Vit_architecture1.png)

![Heatmap generation](./Readme_images/Vit_architecture2.png)



## Usage

This repository can be directly downloaded and executed locally. The required libraries are displayed in Section [Requirements](#requirements)

In the **Implementation** folder we provide a file called **Usage_example** where the user can run the cells and visualize insertion AUC, deletion AUC and heatmaps of one or more images. 
The code can be used both for ViT and DeiT explaination. In addition, this folder contains the following python files:
- **utils**: contains some functions used for the visualization of AUCs and heatmaps.
- **feature_extractor**: contains the definition of the feature extractor class used by ViT and DeiT models.
- **hook**: contains the definition of the hook classes used by ViT and DeiT models.
- **sigrate**: contains the implementation of the model described in the paper.

The **imgs_idx** file contains the indexes of the images used for testing our approach, which are selected from the ImageNet 2012 validation dataset. To download the dataset you need to login on [ImageNet site](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and click on ILSVRC2012, after this download the 'Development kit (Task 1 & 2)', that contains the ground truth labels, and the 'Validation images (all tasks)'.


## Parameters 
In the **Usage_sample** file the user can modify the following parameters:

### Initialization parameters
**`model`**: represents the model of which we want to make explainability; in this demo this value can be _'vit'_ or _'deit'_;  
**`device`**: device in which the model will be used; this can be _'cpu'_ or _'cuda:0'_;  

### Call Parameters
**`token_ratio`**: percentage of patches which are set to 0 during the binary masks creation;  
**`starting_layer`**: layer from which are calculated the metrics used to construct the masks;  
**`masks_layer`**: number of masks for every layer;  
**`img_path`**: path of the image we want to explain;  
**`label`**: label on which we want to do explaination; this value is optional and if is not provided the model will give the heatmap associated with the predicted class;  




## Requirements <a name="requirements"></a>

In our notebook we used the following libraries:
```
PIL=10.3.0  
transformers=4.30.2
torch=2.2.2
torchvision=0.17.2
numpy=1.26.4
pandas=2.2.2
matplotlib=3.9.0
seaborn=0.13.2
```

## Citation

If you use this model for your research please cite our paper.
