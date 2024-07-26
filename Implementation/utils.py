import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
import seaborn as sns
from sklearn.metrics import auc
import numpy as np
import requests
import torch


def create_mask(patch_list, mask_size, patch_size):
    mask = torch.zeros((mask_size, mask_size), dtype=torch.float32)
    for patch_index in patch_list:
        x = (patch_index // (mask_size // patch_size)) * patch_size
        y = (patch_index % (mask_size // patch_size)) * patch_size
        mask[x:x+patch_size, y:y+patch_size] = 1
    return mask

def download_imagenet_labels(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text.splitlines()


# Function to overlay saliency on the input image
def overlay(image, saliency, alpha=0.7):
    """
    Args:
        image (torch.Tensor): Input image tensor.
        saliency (torch.Tensor): Saliency map tensor.
        alpha (float): Transparency level for overlaying the saliency on the image.

    Displays an overlay of the input image and saliency map using bilinear and nearest interpolation.
    """
    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    image = image.permute(1, 2, 0)
    saliency_bilinear = interpolate(saliency.reshape((1, 1, *saliency.shape)), size=image.shape[:2], mode='bilinear')
    saliency_bilinear = saliency_bilinear.squeeze()
    saliency_patch = interpolate(saliency.reshape((1, 1, *saliency.shape)), size=image.shape[:2], mode='nearest')
    saliency_patch = saliency_patch.squeeze()
    ax[0].imshow(image)
    ax[1].imshow(image)
    ax[1].imshow(saliency_bilinear, alpha=alpha, cmap='jet')
    ax[2].imshow(image)
    ax[2].imshow(saliency_patch, alpha=alpha, cmap='jet')
    plt.show()


# Function to plot AUC (Area Under the Curve) for insertion metric
def plot_auc(scores, op, perc_ins):
    """
    Args:
        scores (list): List of confidence scores.
        op (str): Operation type ('insertion' or 'deletion').
        perc_ins (list): List of percentages of patches.

    Plots the AUC curve for insertion or deletion metric, filling the area under the curve.
    """
    auc_value_ins = auc(perc_ins, scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=perc_ins, y=scores)
    plt.fill_between(perc_ins, scores, color="skyblue", alpha=0.4)
    plt.text(x=np.mean(perc_ins), y=np.mean(scores), s=f"AUC: {auc_value_ins:.2f}",
             fontsize=12, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5))
    plt.title(f"AUC {op} metric")
    plt.xlabel("Percentage of Patches")
    plt.ylabel("Confidence")
    plt.grid()
    plt.show()



def overlay2(image, saliency1, saliency2, title = 'Two_classes', alpha=0.7):
    """
    Args:
        image (torch.Tensor): Input image tensor.
        saliency (torch.Tensor): Saliency map tensor.
        alpha (float): Transparency level for overlaying the saliency on the image.

    Displays an overlay of the input image and saliency map using bilinear and nearest interpolation.
    """
    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    image = image.permute(1, 2, 0)
    saliency1 = interpolate(saliency1.reshape((1, 1, *saliency1.shape)), size=image.shape[:2], mode='bilinear')
    saliency1 = saliency1.squeeze()
    saliency2 = interpolate(saliency2.reshape((1, 1, *saliency2.shape)), size=image.shape[:2], mode='bilinear')
    saliency2 = saliency2.squeeze()
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(image)
    ax[1].imshow(saliency1, alpha=alpha, cmap='jet')
    ax[1].set_title('Class: Elephant')
    ax[2].imshow(image)
    ax[2].imshow(saliency2, alpha=alpha, cmap='jet')
    ax[2].set_title('Class: Zebra')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.show()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.savefig(f'{title}.pdf', bbox_inches='tight')
    plt.show()


def plot_auc_line(scores, perc, images, label_list, saliency_list, title='AUC', figsize_x=10, figsize_y=3, alpha=0.7):
    """
    Args:
        scores (list): List of dictionaries containing insertion and deletion scores.
        perc (list): List of percentages of patches.
        images (list): List of input images.
        label_list (list): List of labels for images.
        saliency_list (list): List of saliency maps for images.

    Plots AUC curves for insertion and deletion metrics for each image, along with the input image.
    """
    num_images = len(scores)
    fig, axs = plt.subplots(num_images, 4, figsize=(figsize_x, figsize_y * num_images))

    # Ensure axs is always a 2D array
    if num_images == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx in range(num_images):
        score = scores[idx]
        image = images[idx]
        label = label_list[idx]
        saliency = saliency_list[idx]

        scores_ins = score['insertion']
        scores_del = score['deletion']

        image = image.permute(1, 2, 0).cpu().numpy()

        auc_value_ins = auc(perc, scores_ins)
        auc_value_del = auc([0] + perc, [0] + scores_del)

        axs[idx, 0].imshow(image)
        axs[idx, 0].set_title(label.title())
        axs[idx, 0].grid(False)
        axs[idx, 0].set_xticks([])
        axs[idx, 0].set_yticks([])

        saliency_bilinear = interpolate(saliency.unsqueeze(0).unsqueeze(0), size=image.shape[:2], mode='bilinear').squeeze().cpu().numpy()

        axs[idx, 1].imshow(image)
        axs[idx, 1].imshow(saliency_bilinear, alpha=alpha, cmap='jet')
        axs[idx, 1].set_title('Heatmap')
        axs[idx, 1].grid(False)
        axs[idx, 1].set_xticks([])
        axs[idx, 1].set_yticks([])

        sns.lineplot(ax=axs[idx, 2], x=perc, y=scores_ins)
        axs[idx, 2].fill_between(perc, scores_ins, alpha=0.4, color='skyblue')
        axs[idx, 2].text(x=np.mean(perc), y=np.mean(scores_ins), s=f"AUC: {auc_value_ins:.2f}",
                         fontsize=11, ha='center', va='center',
                         bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.5})
        axs[idx, 2].set_xlim(0, 100)
        axs[idx, 2].set_ylim(0, 1.05)
        axs[idx, 2].grid()
        axs[idx, 2].set_ylabel('Score')
        axs[idx, 2].set_xlabel('% patches shown')
        axs[idx, 2].tick_params(axis='x', labelsize=8)
        axs[idx, 2].tick_params(axis='y', labelsize=8)
        axs[idx, 2].set_title('Insertion')

        sns.lineplot(ax=axs[idx, 3], x=[0] + perc, y=[scores_del[0]] + scores_del)
        axs[idx, 3].fill_between([0] + perc, [0] + scores_del, alpha=0.4, color='skyblue')
        axs[idx, 3].text(x=np.mean([0] + perc), y=np.mean([0] + scores_del), s=f"AUC: {auc_value_del:.2f}",
                         fontsize=11, ha='center', va='center',
                         bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.5})
        axs[idx, 3].set_xlim(0, 100)
        axs[idx, 3].set_ylim(0, 1.05)
        axs[idx, 3].grid()
        axs[idx, 3].set_ylabel('Score')
        axs[idx, 3].set_xlabel('% patches removed')
        axs[idx, 3].tick_params(axis='x', labelsize=8)
        axs[idx, 3].tick_params(axis='y', labelsize=8)
        axs[idx, 3].set_title('Deletion')

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.savefig(f'{title}.pdf', bbox_inches='tight')
    plt.show()
