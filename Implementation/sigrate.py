import numpy as np
import math
import random
from transformers import ViTForImageClassification, ViTImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision
import scipy
from torch.nn.functional import interpolate
from hook import VIT_Hook, DEIT_Hook
from feature_extractor import Custom_feature_extractor
from utils import create_mask

from scipy.ndimage import gaussian_filter

import time 
import matplotlib.pyplot as plt


# Definition of the SIGRATE class
class SIGRATE:

    def __init__(self, model, device):

        # Ensure that the specified model is either 'deit' or 'vit'
        assert model == 'deit' or model == 'vit'

        self.device = torch.device(device)

        if model == 'vit':
            # If the model is ViT, initialize the ViTForImageClassification model, Custom_feature_extractor, and VIT_Hook
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.image_processor = Custom_feature_extractor(device, model)
            self.vit_hook = VIT_Hook(self.model)
        else:
            # If the model is DeiT, initialize the DeiTForImageClassificationWithTeacher model, Custom_feature_extractor, and DEIT_Hook
            self.model = DeiTForImageClassificationWithTeacher.from_pretrained(
                'facebook/deit-base-distilled-patch16-224')
            self.image_processor = Custom_feature_extractor(device, model)
            self.vit_hook = DEIT_Hook(self.model)

        self.model.to(self.device)


    def get_saliency(self, img_path, token_ratio, rw_layer, starting_layer = 0, label=False):
        """
        Generates a saliency heatmap for an input image based on embedding similarity and random walks.

        Args:
            img_path (str): Path to the input image file.
            token_ratio (float): The percentage of top nodes to consider for binary masking.
            label (bool or int, optional): If provided, the ground truth label for the image. If not provided,
                                          the predicted label will be used.

        Returns:
            tuple: Tuple containing the saliency heatmap (reshaped) and the image label.
        """

        torch.manual_seed(42)

        # Open and preprocess the input image
        image = Image.open(img_path).convert('RGB')

        processed_image, attentions_scores, emb, predicted_label = self.classify(image, self.vit_hook, self.image_processor, label)

        # Determine the ground truth label
        ground_truth_label = predicted_label if not label else torch.tensor(label)

        starting_nodes = self.get_best_cls(attentions_scores, rw_layer, starting_layer)
      
        # multilayer = self.create_multilayer(attentions_scores, starting_layer)
        multilayer = self.create_multilayer_emb(emb, attentions_scores, starting_layer)
      
        num_layers = multilayer.shape[0]
        total_patches = multilayer.shape[2]

        # Calculate random walks
        masks_array = self.get_masks(multilayer=multilayer, token_ratio=token_ratio,  rw_layer = rw_layer, starting_nodes = starting_nodes)

        
        B = len(masks_array)
        
        mean_len = sum(len(sublist) for sublist in masks_array)/B

        # Obtain model predictions for different sampled token configurations
        confidence_ground_truth_class, top_confidences, top_class_indices = self.vit_hook.classify_with_sampled_tokens(processed_image, masks_array,
                                                                                   ground_truth_label)

        confidence_ground_truth_class = torch.tensor(confidence_ground_truth_class).to(self.device)

        # for elem in masks_array:
        #   mask = create_mask(elem, 224, 16)
        #   fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        #   # resizer(image, return_tensors="pt")['pixel_values'][0].to('cpu')
        #   resizer = ViTImageProcessor(do_rescale=False, do_normalize=False)
        #   result = resizer(image, return_tensors="pt")['pixel_values'][0].to('cpu') * mask.unsqueeze(0).to('cpu')
          
        #   ax.imshow(result.permute(1, 2, 0).to(torch.uint8))
  
        #   plt.tight_layout()  # Assicura che le immagini siano ben posizionate
        #   plt.show()
      
        num_rows = len(masks_array)
        num_cols = max(max(sublist) for sublist in masks_array) + 1  # dimensione massima necessaria
        
        binary_mask_tensor = torch.zeros((num_rows, num_cols), dtype=torch.float32)
        
        for i, indices in enumerate(masks_array):
            binary_mask_tensor[i, indices] = 1

      
        # Calculate the final saliency heatmap
        heatmap_ground_truth_class = binary_mask_tensor * confidence_ground_truth_class.view(-1, 1)

        heatmap_ground_truth_class = torch.sum(heatmap_ground_truth_class, dim=0)
        coverage_bias = torch.sum(binary_mask_tensor, dim=0)
        coverage_bias = torch.where(coverage_bias > 0, coverage_bias, 1)

        heatmap_ground_truth_class = heatmap_ground_truth_class / coverage_bias
        heatmap_ground_truth_class = torch.softmax(heatmap_ground_truth_class, dim = 0)
        heatmap_ground_truth_class = heatmap_ground_truth_class.reshape((14, 14)).to('cpu')

        # threshold = torch.quantile(heatmap_ground_truth_class.reshape(196), 0.20)
        threshold = torch.mean(heatmap_ground_truth_class.reshape(196))
        heatmap = torch.where(heatmap_ground_truth_class < threshold, torch.tensor(0., device=heatmap_ground_truth_class.device), heatmap_ground_truth_class)
        
      
        # ROBA STRANA #
        
        def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
            """Crea un kernel gaussiano 2D"""
            coords = torch.arange(size) - size // 2
            x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
            kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            return kernel
        
        # Heatmap di input: (1, 1, 14, 14)  [batch_size=1, channels=1]
        heatmap = heatmap.reshape(1, 1, 14, 14)
        
        # Crea kernel gaussiano 3x3
        kernel = gaussian_kernel(size=5, sigma=1)
        kernel = kernel.view(1, 1, 5, 5)  # shape: (out_channels, in_channels, H, W)
        
        # Applica convoluzione (padding=1 per mantenere stessa dimensione)
        smoothed = F.conv2d(heatmap, kernel, padding=2)
        
        # Poi prosegui con il tuo codice
        # mean = smoothed.mean()
        # std = smoothed.std()
        # a = 1.0 / std
        # smoothed = torch.sigmoid(a * (smoothed - mean))**4
        # smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        smoothed = smoothed**4
        smoothed = smoothed.reshape((14, 14))

        # threshold = torch.quantile(smoothed.reshape(196), 0.20)
        # # FINE ROBA STRANA
        # percentile = torch.quantile(smoothed, 0.5)
        # smoothed = torch.where(smoothed < mean, torch.tensor(0., device=smoothed.device), smoothed)

        # FINE ROBA STRANA #
        # threshold = torch.mean(smoothed.reshape(196))
        # smoothed = torch.where(smoothed < threshold, torch.tensor(0., device=smoothed.device), smoothed)

        print(mean_len)
      
        return smoothed, ground_truth_label.item()



    def classify(self, image, model, image_processor, class_index):
        inputs = image_processor(images=image, return_tensors="pt")
        logits, attention_scores, embeddings = model.classify_and_capture_outputs(inputs, class_index, output_attentions=True)
        probabilities = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1)
    
        return inputs, attention_scores, embeddings, predicted_class_idx




    def get_best_cls(self, attentions, rw_layer, starting_layer, ):
        att_list = []
        for i in range(len(attentions)):
          att_no_head = torch.max(attentions[i][0], dim = 0)[0]
          if isinstance(self.model, ViTForImageClassification):
              att_no_head_cls = att_no_head[0, 1:] # embeddings without the CLS token
          else:
              att_no_head_cls = att_no_head[0, 2:] # embeddings without the CLS and the distillation token

          att_list.append(att_no_head_cls)
          
        worst_number = int(rw_layer/2)
        top_number = rw_layer - worst_number
      
        attns = torch.stack(att_list, dim=0)
        attns = attns[starting_layer:, :]
        topk_values, topk_indices = torch.topk(attns, k=top_number, dim=1, largest=True, sorted=True)
        worstk_values, worstk_indices = torch.topk(attns, k=worst_number, dim=1, largest=False, sorted=True)
        indices = torch.cat([topk_indices, worstk_indices], dim = 1)
        return indices
        
        
          

    def get_similarity(self, embeddings):
        """
        Creates the similarity matrix of embeddings.

        Args:
            embeddings (torch.Tensor): Embeddings weights of the nodes.
        Returns:
            torch.Tensor: adjacency matrices.
        """
      
        B, N, D = embeddings.shape
        
        # Normalize the vectors along the D dimension
        norm_embeddings = F.normalize(embeddings, p=2, dim=2)  # shape: [B, N, D]
        
        # Calculate the dot product between each pair of vectors
        similarity_matrix = torch.bmm(norm_embeddings, norm_embeddings.transpose(1, 2))  # shape: [B, N, N]

        # # Transform the similarity between 0 and 1
        similarity_matrix = torch.softmax(similarity_matrix, dim = 2)
        
        return similarity_matrix
          


    def create_multilayer_emb(self, embeddings, attentions, starting_layer):
        """
        Creates similarity matrices across multiple layers, combining embeddings.
        Both are normalized to avoid domination by any layer.
    
        Args:
            embeddings (list[Tensor]): [L, 1, T, D] — list of layer activations
            attentions (list[Tensor]): [L, 1, H, T, T] — list of layer attentions
            starting_layer (int): Index of the first layer to consider
    
        Returns:
            torch.Tensor: [L', T, T] similarity matrices from selected layers
        """
    
        embeddings_list = []
        att_no_head = []
    
        for i in range(len(embeddings)):
            emb = embeddings[i][0]  # [T, D]
            att = attentions[i][0]  # [H, T, T]
    
            # Remove special tokens
            if isinstance(self.model, ViTForImageClassification):
                emb = emb[1:]
                att = att[:, 1:, 1:]
            else:
                emb = emb[2:]
                att = att[:, 2:, 2:]
    
            embeddings_list.append(emb)
            att_no_head.append(att.mean(dim=0))  # [T, T]
    
        # Convert to tensors: [L, T, D] or [L, T, T]
        embeddings_tensor = torch.stack(embeddings_list, dim=0)
        att_tensor = torch.stack(att_no_head, dim=0)
    
        # Slice from starting layer
        embeddings_tensor = embeddings_tensor[starting_layer:]
        att_tensor = att_tensor[starting_layer:]
    
        # ---- 🔄 Normalize each layer's embedding and gradient ---- #
        def normalize(tensor):
            # L2 normalization along the feature dimension (dim=-1)
            norm = torch.norm(tensor, dim=-1, keepdim=True) + 1e-6
            return tensor / norm
    
        norm_embeddings = normalize(embeddings_tensor)  # [L, T, D]
        norm_att = normalize(att_tensor)
    
        # ---- ⚙️ Combine with element-wise product ---- #
        scaled_emb = norm_embeddings# * norm_gradients.mean(dim = 2).unsqueeze(-1)  # [L, T, D]
    
        # ---- 🔍 Compute pairwise similarity across tokens ---- #
        similarity_multilayer = self.get_similarity(scaled_emb)  # [L, T, T]

        L, T, _ = similarity_multilayer.shape
        mask = torch.eye(T, device=similarity_multilayer.device).bool()  # [T, T]
        similarity_multilayer[:, mask] = 0
    
        return similarity_multilayer#*norm_att

  
    def crea_matrice_adiacenza(self, B, N, row_mean):
        # Determina la dimensione della griglia
        dim = math.ceil(math.sqrt(N))
        
        # Inizializza la matrice di adiacenza con zeri
        matrice_adiacenza = torch.zeros((N, N), dtype=torch.int, device = row_mean.device)

        val = 1
      
        for i in range(N):
            # Connessione orizzontale (verso destra)
            if (i % dim) < (dim - 1) and (i + 1) < N:
                matrice_adiacenza[i, i + 1] = val
                matrice_adiacenza[i + 1, i] = val
            
            # Connessione verticale (verso il basso)
            if (i + dim) < N:
                matrice_adiacenza[i, i + dim] = val
                matrice_adiacenza[i + dim, i] = val
        
        # Espandi la matrice per includere la dimensione batch
        matrice_adiacenza = matrice_adiacenza.unsqueeze(0).expand(B, N, N)

        matrice_adiacenza = matrice_adiacenza * 1/N # row_mean[:, :, None]
        
        return matrice_adiacenza



    def modify_image(self, operation, heatmap, image, percentage, baseline, device):
        """
        Modifies an image based on the given operation, heatmap, and baseline.

        Args:
            operation (str): The operation to perform ('deletion' or 'insertion').
            heatmap (torch.Tensor): The heatmap indicating pixel importance.
            image (dict): The image dictionary containing 'pixel_values'.
            percentage (float): The percentage of top pixels to consider for modification.
            baseline (str): The baseline image type ('black', 'blur', 'random', or 'mean').
            device: The device on which to perform the operation.

        Returns:
            torch.Tensor: The modified image tensor.
        """
        if operation not in ['deletion', 'insertion']:
            raise ValueError("Operation must be either 'deletion' or 'insertion'.")

        # Finding the top percentage of most important pixels in the heatmap
        num_top_pixels = int(percentage * heatmap.shape[0] * heatmap.shape[1])
        top_pixels_indices = np.unravel_index(np.argsort(heatmap.ravel())[-num_top_pixels:], heatmap.shape)

        # Extract and copy the image tensor
        img_tensor = image['pixel_values'].squeeze(0)
        img_tensor = img_tensor.permute(1, 2, 0)
        modified_image = np.copy(img_tensor.cpu().numpy())

        tensor_img_reshaped = img_tensor.permute(2, 0, 1)

        # Define baseline image based on the specified type
        if baseline == "black":
            img_baseline = torch.zeros(tensor_img_reshaped.shape, dtype=bool).to(device)
        elif baseline == "blur":
            img_baseline = torchvision.transforms.functional.gaussian_blur(tensor_img_reshaped, kernel_size=[15, 15],
                                                                           sigma=[7, 7])
        elif baseline == "random":
            img_baseline = torch.randn_like(tensor_img_reshaped)
        elif baseline == "mean":
            img_baseline = torch.ones_like(tensor_img_reshaped) * tensor_img_reshaped.mean()

        if operation == 'deletion':
            # Replace the most important pixels by applying the baseline values
            darken_mask = torch.zeros(heatmap.shape, dtype=bool).to(device)
            darken_mask[top_pixels_indices] = 1
            modified_image = torch.where(darken_mask > 0, img_baseline, tensor_img_reshaped)

        elif operation == 'insertion':
            # Replace the less important pixels by applying the baseline values
            keep_mask = torch.zeros(heatmap.shape, dtype=bool).to(device)
            keep_mask[top_pixels_indices] = 1
            modified_image = torch.where(keep_mask > 0, tensor_img_reshaped, img_baseline)

        return modified_image



    def calculate_random_walk(self, adj_matrix, walk_length, starting_node):
        walk = torch.full((walk_length + 1,), starting_node, dtype=torch.long)
        current_node = starting_node
        previous_node = -1  # Nessun nodo precedente all'inizio
    
        for i in range(1, walk_length + 1):
            neighbors = adj_matrix[current_node].clone()
    
            # Evita il nodo immediatamente precedente
            if previous_node != -1:
                neighbors[previous_node] = 0
    
            # Se non ci sono più archi validi, termina il walk
            if torch.all(neighbors == 0):
                break
    
            # Scegli il nodo con l'arco più pesante (escluso quello precedente)
            next_node = torch.argmax(neighbors).item()
            walk[i] = next_node
    
            # Dimezza (o moltiplica per 0.8) il peso dell'arco percorso
            adj_matrix[current_node, next_node] *= 0.5
            adj_matrix[next_node, current_node] *= 0.5  # perché il grafo è non orientato
    
            # Aggiorna per il passo successivo
            previous_node = current_node
            current_node = next_node

        return walk



      

    def get_masks(self, multilayer, token_ratio, rw_layer, starting_nodes):
        walk_length = int(multilayer.shape[1] * token_ratio)
        masks = []
    
        for layer in range(multilayer.shape[0]):
            starting_nodes_layer = starting_nodes[layer]
            adj_matrix = multilayer[layer].clone()
            adj_matrix.fill_diagonal_(0)
    
            for current_rw in range(rw_layer):
                starting_node_current_rw = starting_nodes_layer[current_rw].item()
                rw_mask = self.calculate_random_walk(adj_matrix, walk_length, starting_node_current_rw)
                rw_mask = torch.unique(rw_mask)
                masks.append(rw_mask)
    
        return masks





    def get_insertion_deletion(self, patch_perc, heatmap, image, baseline, label):
        """
        Generates confidence scores for insertion and deletion for the specif baseline and every patch_perc.

        Args:
            patch_perc (list): List of patch percentages to consider.
            heatmap (torch.Tensor): Original heatmap.
            image (torch.Tensor): Original image tensor.
            baseline (str): Baseline image type ('black', 'blur', 'random', or 'mean').
            label: True label of the image.

        Returns:
            dict: Dictionary containing confidence scores for 'insertion' and 'deletion' operations.
        """

        # Process the original image
        image = self.image_processor(images=image, return_tensors="pt")

        # Reshape and interpolate the heatmap to match the image size
        heatmap = heatmap.reshape((1, 1, 14, 14))
        gaussian_heatmap = interpolate(heatmap, size=(224, 224), mode='nearest')
        gaussian_heatmap = gaussian_heatmap[0, 0, :, :].to('cpu').detach()

        confidences = {}

        for operation in ['insertion', 'deletion']:
            batch_modified = []
            for percentage in patch_perc:
                modified_image = self.modify_image(operation=operation, heatmap=gaussian_heatmap, image=image,
                                                   percentage=percentage / 100, baseline=baseline, device=self.device)
                batch_modified.append(modified_image)

            batch_modified = torch.stack(batch_modified, dim=0).to(self.device)
            confidences[operation] = self.predict(batch_modified, label)

        return confidences

    def predict(self, obscured_inputs, true_class_index):
        """
        Predicts the class probabilities for the true class for a list of obscured inputs.

        Args:
            obscured_inputs (torch.Tensor): Batch of obscured images.
            true_class_index (int): True class index for the original image.

        Returns:
            list: List of predicted probabilities for the true class in each obscured input.
        """
        outputs = self.model(obscured_inputs)
        probabilities = F.softmax(outputs.logits, dim=1)

        predicted_class_indices = torch.argmax(probabilities, dim=1)

        true_class_probs = probabilities[:, true_class_index]

        return true_class_probs.tolist()
