import numpy as np
import math
from transformers import ViTForImageClassification, ViTImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision
from torch.nn.functional import interpolate
from hook import VIT_Hook, DEIT_Hook
from feature_extractor import Custom_feature_extractor


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


    def get_saliency(self, img_path, token_ratio, masks_layer, starting_layer = 0, label=False):
        """
        Generates a saliency heatmap for an input image based on embedding similarity.

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

        processed_image, attentions_scores, emb, predicted_label = self.classify(image, self.vit_hook, self.image_processor)

        # Determine the ground truth label
        ground_truth_label = predicted_label if not label else torch.tensor(label)

        starting_nodes = self.get_best_cls(attentions_scores, masks_layer, starting_layer)
      
        # multilayer = self.create_multilayer(attentions_scores, starting_layer)
        multilayer = self.create_multilayer_emb(emb, starting_layer)
      
        num_layers = multilayer.shape[0]
        total_patches = multilayer.shape[2]

        # Calculate random masks
        masks_array = self.get_masks(multilayer=multilayer, token_ratio=token_ratio,  masks_layer = masks_layer, starting_nodes = starting_nodes)

        # Convert patch importance list to a tensor
        mask_tensor = torch.stack(masks_array).to(self.device)

        # Obtain model predictions for different sampled token configurations
        confidence_ground_truth_class = self.vit_hook.classify_with_sampled_tokens(processed_image, mask_tensor,
                                                                                   ground_truth_label)

        confidence_ground_truth_class = torch.tensor(confidence_ground_truth_class).to(self.device)
        
        B, _ = mask_tensor.shape
        binary_mask_tensor = torch.zeros((B, total_patches), dtype=torch.int32).to(self.device)
        binary_mask_tensor.scatter_(1, mask_tensor, 1)
      
        # Calculate the final saliency heatmap
        heatmap_ground_truth_class = binary_mask_tensor * confidence_ground_truth_class.view(-1, 1)

        heatmap_ground_truth_class = torch.sum(heatmap_ground_truth_class, dim=0)
        coverage_bias = torch.sum(binary_mask_tensor, dim=0)
        coverage_bias = torch.where(coverage_bias > 0, coverage_bias, 1)

        heatmap_ground_truth_class = heatmap_ground_truth_class / coverage_bias
        heatmap_ground_truth_class_reshape = heatmap_ground_truth_class.reshape((14, 14))


        return heatmap_ground_truth_class_reshape.to('cpu'), ground_truth_label.item()



    def classify(self, image, model, image_processor):
        """
        Classifies an image using the specified model and image processor.

        Args:
            image (torch.Tensor): The input image tensor.
            model (torch.nn.Module): The classification model.
            image_processor (Custom_feature_extractor): The image processor.

        Returns:
            tuple: A tuple containing input features, embedding, and the predicted class index.
        """
        # Process the input image using the provided image processor
        inputs = image_processor(images=image, return_tensors="pt")

        # Forward pass through the model with output_hidden_states
        logits, attentions_scores, embeddings = model.classify_and_capture_outputs(inputs, output_attentions = True)

        # Compute softmax probabilities and predict the class index
        probabilities = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1)

        return inputs, attentions_scores, embeddings, predicted_class_idx

    def get_best_cls(self, attentions, masks_layer, starting_layer):
        """
        Select the best and worst CLS embeddings based on attention scores.

        Parameters:
        attentions (list of torch.Tensor): A list of attention score tensors. Each tensor
                                           has a shape of (num_heads, num_tokens, num_tokens).
                                           The list length corresponds to the number of layers.
        masks_layer (int): Number of masks, used to determine how many top and worst
                        attention scores to select.
        starting_layer (int): The layer from which to start considering attention scores.

        Returns:
        torch.Tensor: A tensor containing the indices of the selected top and worst attention scores.
                      Shape: (num_layers - starting_layer, masks_layer). Each row corresponds to a layer
                      and contains indices of the selected attention scores.
        """

        att_list = []

        # Iterate through the list of attention tensors
        for i in range(len(attentions)):
            # Get the attention scores without the head dimension
            att_no_head = torch.max(attentions[i][0], dim=0)[0]

            # Remove the CLS token (and possibly the distillation token) from the embeddings
            if isinstance(self.model, ViTForImageClassification):
                att_no_head_cls = att_no_head[0, 1:]  # embeddings without the CLS token
            else:
                att_no_head_cls = att_no_head[0, 2:]  # embeddings without the CLS and the distillation token

            # Add the processed attention scores to the list
            att_list.append(att_no_head_cls)

        # Calculate the number of worst and top attention scores to select
        worst_number = int(masks_layer / 2)
        top_number = masks_layer - worst_number

        # Stack the list of attention scores into a tensor
        attns = torch.stack(att_list, dim=0)

        # Consider attention scores starting from the specified layer
        attns = attns[starting_layer:, :]

        # Get the indices of the top attention scores
        topk_values, topk_indices = torch.topk(attns, k=top_number, dim=1, largest=True, sorted=True)

        # Get the indices of the worst attention scores
        worstk_values, worstk_indices = torch.topk(attns, k=worst_number, dim=1, largest=False, sorted=True)

        # Concatenate the indices of the top and worst attention scores
        indices = torch.cat([topk_indices, worstk_indices], dim=1)

        # Return the concatenated indices
        return indices

    def get_similarity(self, embeddings):
        """
        Creates the similarity matrix of embeddings.

        Args:
            embeddings (torch.Tensor): Embeddings weights of the nodes.
        Returns:
            torch.Tensor: adjacency matrices.
        """

        # Normalize the vectors along the D dimension
        norm_embeddings = F.normalize(embeddings, p=2, dim=2)  # shape: [B, N, D]
        
        # Calculate the dot product between each pair of vectors
        similarity_matrix = torch.bmm(norm_embeddings, norm_embeddings.transpose(1, 2))  # shape: [B, N, N]
        
        return similarity_matrix
          

    def create_multilayer_emb(self, embeddings, starting_layer):
        """
        Creates adjacency matrices based on the similarity of the embeddings.

        Args:
            embeddings (torch.Tensor): Embeddings weights across layers and nodes.
        Returns:
            torch.Tensor: Multilayer adjacency matrices.
        """

        embeddings_list = []
      
        for i in range(1, len(embeddings)):
          # Get the embeddings of the first and only image          
          embeddings_image = embeddings[i][0]
          embeddings_list.append(embeddings_image)

        # Stack the aggregated embeddings across layers
        embeddings_tensor = torch.stack(embeddings_list, dim=0)
      
        # Extract relevant embeddings based on model type
        if isinstance(self.model, ViTForImageClassification):
            embeddings_tensor = embeddings_tensor[:, 1:, :].to(self.device)  # embeddings without the CLS token
        else:
            embeddings_tensor = embeddings_tensor[:, 2:, :].to(self.device)  # embeddings without the CLS and the distillation token


        embeddings_tensor = embeddings_tensor[starting_layer:, :, :]

        similarity_multilayer = self.get_similarity(embeddings_tensor)

        return similarity_multilayer


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


    def calculate_masks_layer(self, adj_matrix, masks_length, starting_node):
        """
        Create a masks on a given adjacency matrix.

        Parameters:
        adj_matrix (torch.Tensor): The adjacency matrix of the graph. Shape: (num_nodes, num_nodes).
                                   It represents the connectivity of the nodes in the graph.
        masks_length (int): The length of the masks.
        starting_node (int): The starting node for the masks.

        Returns:
        torch.Tensor: A tensor containing the sequence of nodes visited.
                      Shape: (masks_length + 1,). The first element is the starting node.
        """

        # Number of nodes in the graph
        N = adj_matrix.size(0)

        # Initialize the masks tensor with the starting node. The length is masks_length + 1 to include the starting node.
        masks = torch.full((masks_length + 1,), starting_node, dtype=torch.long)

        # Create a tensor to track visited nodes, initialized to False
        visited = torch.zeros(N, dtype=torch.bool)

        # Mark the starting node as visited
        visited[starting_node] = True

        # Initialize the current node to the starting node
        current_node = starting_node

        # Create the masks
        for i in range(1, masks_length + 1):
            # Get the probabilities (edge weights) for the current node's neighbors
            probabilities = adj_matrix[current_node]

            # Set the probabilities of already visited nodes to 0
            probabilities[visited] = 0

            # Select the next node with the highest probability
            next_node = torch.max(probabilities, dim=0)[1]

            # Add the next node to the masks
            masks[i] = next_node

            # Mark the next node as visited
            visited[next_node] = True

            # Update the current node to the next node
            current_node = next_node

        # Return the sequence of nodes visited
        return masks

    def get_masks(self, multilayer, token_ratio, masks_layer, starting_nodes):
        """
        Generate masks for every layer of the embedding matrix.

        Parameters:
        multilayer (torch.Tensor): A 3D tensor representing the multilayer network.
                                   Shape: (num_layers, num_nodes, num_nodes), where each
                                   slice along the first dimension is an adjacency matrix
                                   for a layer.
        token_ratio (float): Ratio to determine the size of the masks. The size
                             is computed as masks_length = int(num_nodes * token_ratio).
        masks_layer (int): Number of masks for each layer.
        starting_nodes (list of lists): A list containing lists of starting nodes for each
                                        masks in each layer. Shape: (num_layers, masks_layer).

        Returns:
        list: A list of masks generated. Each mask is produced by the
              `calculate_masks_layer` method.
        """

        # Calculate the size of the masks based on the token_ratio
        masks_length = int(multilayer.shape[1] * token_ratio)

        # Initialize an empty list to store the generated masks
        masks = []

        # Iterate through each layer in the multilayer network
        for layer in range(multilayer.shape[0]):
            # Get the starting nodes for the current layer
            starting_nodes_layer = starting_nodes[layer]

            # Get the adjacency matrix for the current layer
            adj_matrix = multilayer[layer]

            # Remove self-loops by setting the diagonal elements to zero
            adj_matrix.fill_diagonal_(0)

            # Create the specified number of masks for the current layer
            for current_rw in range(masks_layer):
                # Get the starting node for the current mask
                starting_node_current_rw = starting_nodes_layer[current_rw]

                # Calculate the mask
                # The `clone()` method ensures the original adjacency matrix is not modified
                rw_mask = self.calculate_masks_layer(
                    adj_matrix.clone(),
                    masks_length,
                    starting_node_current_rw.item()
                )

                # Add the generated mask to the list of masks
                masks.append(rw_mask)

        # Return the list of generated masks
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

        true_class_probs = probabilities[:, true_class_index]

        return true_class_probs.tolist()
