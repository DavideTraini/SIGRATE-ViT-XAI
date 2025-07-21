import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
import seaborn as sns
from sklearn.metrics import auc
import numpy as np
import requests
import torch
from matplotlib.patches import Rectangle


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
    parte1, parte2 = title.split("-")
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(image)
    ax[1].imshow(saliency1, alpha=alpha, cmap='jet')
    ax[1].set_title(f'Class: {parte1}')
    ax[2].imshow(image)
    ax[2].imshow(saliency2, alpha=alpha, cmap='jet')
    ax[2].set_title(f'Class: {parte2}')
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


def overlay3(image, saliency, bbxs=None, alpha=0.7):
    """
    Visualizza immagine originale, saliency bilineare, saliency patch con bounding box.

    Args:
        image (torch.Tensor): Immagine originale (C, H, W).
        saliency (torch.Tensor): Mappa di salienza.
        bbxs (list): Lista di bounding box (x1, y1, x2, y2).
        alpha (float): Trasparenza per overlay salienza.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Interpolazioni
    saliency_bilinear = interpolate(saliency.reshape((1, 1, *saliency.shape)), size=image.shape[1:], mode='bilinear').squeeze()
    saliency_patch = interpolate(saliency.reshape((1, 1, *saliency.shape)), size=image.shape[1:], mode='nearest').squeeze()

    titles = ["Original + BBoxes", "Saliency (bilinear)", "Saliency (patch)"]
    overlays = [None, saliency_bilinear, saliency_patch]

    for i in range(3):
        ax[i].imshow(image_np)
        if overlays[i] is not None:
            ax[i].imshow(overlays[i], alpha=alpha, cmap='jet')
        if bbxs:
            for (x1, y1, x2, y2) in bbxs:
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='lime', facecolor='none')
                ax[i].add_patch(rect)
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.tight_layout()
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

        val1 = auc(perc, scores_ins)/100 
      
        val2 = auc([0] + perc, [0] + scores_del)/100
      
        axs[idx, 0].imshow(image)
        axs[idx, 0].set_title(label.title().lower())
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
        axs[idx, 2].text(x=np.mean(perc), y=np.mean(scores_ins), s=f"AUC: {val1:.2f}",
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
        axs[idx, 3].text(x=np.mean([0] + perc), y=np.mean([0] + scores_del), s=f"AUC: {val2:.2f}",
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



def plot_auc_line2(scores, perc, images, label_list, saliency_list, bbxs_scaled_list=None,
                  title='AUC', figsize_x=10, figsize_y=3, alpha=0.7):
    """
    Plots image, saliency heatmap, and insertion/deletion AUC for each image.

    Args:
        scores (list): List of dictionaries with 'insertion' and 'deletion' scores.
        perc (list): List of percentages of patches shown/removed.
        images (list): List of input image tensors (C, H, W).
        label_list (list): List of image labels.
        saliency_list (list): List of saliency maps.
        bbxs_list (list): List of bounding boxes [(x1, y1, x2, y2), ...] for each image.
        original_sizes (list): List of (width, height) tuples for original image sizes.
        title (str): Title for saving figure.
        figsize_x (int): Width of figure.
        figsize_y (int): Height per image.
        alpha (float): Transparency for heatmap overlay.
        val1_init (float): Optional AUC value for insertion.
        val2_init (float): Optional AUC value for deletion.
    """
    num_images = len(scores)
    fig, axs = plt.subplots(num_images, 4, figsize=(figsize_x, figsize_y * num_images))

    if num_images == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx in range(num_images):
        score = scores[idx]
        image = images[idx]
        label = label_list[idx]
        saliency = saliency_list[idx]
        bbxs_scaled = bbxs_scaled_list[idx]

        scores_ins = score['insertion']
        scores_del = score['deletion']

        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

        # Calcolo AUC
        val1 = auc(perc, scores_ins) / 100
        val2 = auc([0] + perc, [0] + scores_del) / 100

        # === Immagine originale ===
        axs[idx, 0].imshow(image_np)
        axs[idx, 0].set_title(label.title().lower())
        axs[idx, 0].grid(False)
        axs[idx, 0].set_xticks([])
        axs[idx, 0].set_yticks([])
        for (x1, y1, x2, y2) in bbxs_scaled:
          rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
          axs[idx, 0].add_patch(rect)

        # === Heatmap saliency ===
        saliency_bilinear = interpolate(saliency.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2], mode='bilinear')
        saliency_bilinear = saliency_bilinear.squeeze().cpu().numpy()

        axs[idx, 1].imshow(image_np)
        axs[idx, 1].imshow(saliency_bilinear, alpha=alpha, cmap='jet')
        axs[idx, 1].set_title('Heatmap')
        axs[idx, 1].grid(False)
        axs[idx, 1].set_xticks([])
        axs[idx, 1].set_yticks([])
        for (x1, y1, x2, y2) in bbxs_scaled:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
            axs[idx, 1].add_patch(rect)

        # === Insertion ===
        sns.lineplot(ax=axs[idx, 2], x=perc, y=scores_ins)
        axs[idx, 2].fill_between(perc, scores_ins, alpha=0.4, color='skyblue')
        axs[idx, 2].text(x=np.mean(perc), y=np.mean(scores_ins), s=f"AUC: {val1:.2f}",
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

        # === Deletion ===
        sns.lineplot(ax=axs[idx, 3], x=[0] + perc, y=[scores_del[0]] + scores_del)
        axs[idx, 3].fill_between([0] + perc, [0] + scores_del, alpha=0.4, color='skyblue')
        axs[idx, 3].text(x=np.mean([0] + perc), y=np.mean([0] + scores_del), s=f"AUC: {val2:.2f}",
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




def plot_auc_single(scores, perc, images, label_list, saliency_list, bbxs_scaled_list=None,
                    title='AUC_single', figsize_x=12, figsize_y=3, alpha=0.7, labelsize = 12, fontsize = 14, fontsize_grid = 12):
    """
    Salva un PDF per ciascuna immagine con:
    - immagine con bbox,
    - heatmap con bbox,
    - grafici AUC di insertion e deletion.

    Args:
        scores, perc, images, label_list, saliency_list: come prima.
        bbxs_scaled_list: bounding boxes già ridimensionati (per immagine 224x224).
        title: prefisso per i file PDF salvati.
    """
    from matplotlib.patches import Rectangle
    from sklearn.metrics import auc
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.nn.functional import interpolate

    num_images = len(scores)

    for idx in range(num_images):
        score = scores[idx]
        image = images[idx]
        label = label_list[idx]
        saliency = saliency_list[idx]
        bbxs_scaled = bbxs_scaled_list[idx] if bbxs_scaled_list else []

        scores_ins = score['insertion']
        scores_del = score['deletion']

        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

        # Calcolo AUC
        val1 = auc(perc, scores_ins) / 100
        val2 = auc([0] + perc, [0] + scores_del) / 100

        # Plot a singola riga
        fig, axs = plt.subplots(1, 4, figsize=(figsize_x, figsize_y))

        # === Immagine originale ===
        axs[0].imshow(image_np)
        axs[0].set_title(label.title().lower(), fontsize=fontsize)
        axs[0].grid(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        for (x1, y1, x2, y2) in bbxs_scaled:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
            axs[0].add_patch(rect)

        # === Heatmap saliency ===
        saliency_bilinear = interpolate(saliency.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2], mode='bilinear')
        saliency_bilinear = saliency_bilinear.squeeze().cpu().numpy()

        axs[1].imshow(image_np)
        axs[1].imshow(saliency_bilinear, alpha=alpha, cmap='jet')
        axs[1].set_title('Heatmap', fontsize=fontsize)
        axs[1].grid(False)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        for (x1, y1, x2, y2) in bbxs_scaled:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
            axs[1].add_patch(rect)

        # === Insertion ===
        sns.lineplot(ax=axs[2], x=perc, y=scores_ins)
        axs[2].fill_between(perc, scores_ins, alpha=0.4, color='skyblue')
        axs[2].text(x=np.mean(perc), y=np.mean(scores_ins), s=f"AUC: {val1:.2f}",
                    fontsize=fontsize, ha='center', va='center',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.5})
        axs[2].set_xlim(0, 100)
        axs[2].set_ylim(0, 1.05)
        axs[2].grid()
        axs[2].set_ylabel('Score', fontsize=fontsize_grid)
        axs[2].set_xlabel('% patches shown', fontsize=fontsize_grid)
        axs[2].tick_params(axis='x', labelsize=10)
        axs[2].tick_params(axis='y', labelsize=10)
        axs[2].set_title('Insertion', fontsize=fontsize)

        # === Deletion ===
        sns.lineplot(ax=axs[3], x=[0] + perc, y=[scores_del[0]] + scores_del)
        axs[3].fill_between([0] + perc, [0] + scores_del, alpha=0.4, color='skyblue')
        axs[3].text(x=np.mean([0] + perc), y=np.mean([0] + scores_del), s=f"AUC: {val2:.2f}",
                    fontsize=fontsize, ha='center', va='center',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.5})
        axs[3].set_xlim(0, 100)
        axs[3].set_ylim(0, 1.05)
        axs[3].grid()
        axs[3].set_ylabel('Score', fontsize=fontsize_grid)
        axs[3].set_xlabel('% patches removed', fontsize=fontsize_grid)
        axs[3].tick_params(axis='x', labelsize=10)
        axs[3].tick_params(axis='y', labelsize=10)
        axs[3].set_title('Deletion', fontsize=fontsize)

        # === Salvataggio ===
        fig.tight_layout()
        file_title = f"{title}_{label.lower().replace(' ', '_')}.pdf"
        fig.savefig(file_title, bbox_inches='tight')
        plt.close(fig)  # chiudi la figura per liberare memoria
