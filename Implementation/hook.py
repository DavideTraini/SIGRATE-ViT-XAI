import torch

import torch.nn.functional as F

class VIT_Hook:
    def __init__(self, model):
        self.model = model
        self.outputs = []
        self.gradients = []

    def sampling_hook(self, token_indices):
        def hook(module, input, output):
            cls_token = output[:, 0:1, :]
            token_embeddings = output[:, 1:, :]
            sampled_tokens = token_embeddings[:, token_indices, :]
            new_output = torch.cat([cls_token, sampled_tokens], dim=1)
            return new_output
        return hook

    def output_hook(self, module, input, output):
        self.outputs.append(output)

    
    def classify_with_sampled_tokens(self, inputs, token_indices_list, class_index):
        """
        Classifies an input image by sampling tokens and returning class probabilities.
    
        Args:
            inputs (dict): Input data for the model.
            token_indices_list (list): List of token indices to be sampled during classification.
            class_index (int): Index of the target class for which the probability will be calculated.
    
        Returns:
            tuple of lists:
                - List of probabilities for the target class.
                - List of maximum softmax confidences.
                - List of predicted class indices.
        """
        true_class_probs = []
        top_confidences = []
        top_class_indices = []
    
        for token_indices in token_indices_list:
            token_indices = token_indices[token_indices != -1]
    
            # Register sampling hook
            hook = self.model.vit.embeddings.dropout.register_forward_hook(self.sampling_hook(token_indices))
    
            # Forward pass
            outputs = self.model(**inputs)
            hook.remove()
    
            # Get predictions
            predictions = outputs.logits.softmax(dim=-1)[0]  # shape: [num_classes]
            true_class_probs.append(predictions[class_index].item())
    
            top_conf, top_class_idx = predictions.max(dim=-1)
            top_confidences.append(top_conf.item())
            top_class_indices.append(top_class_idx.item())
    
        return true_class_probs, top_confidences, top_class_indices


    def classify_and_capture_outputs(self, inputs, class_index=None, output_attentions=False):
        self.outputs = []
    
        def make_capture_and_hook():
            def capture_and_hook(module, input, output):
                self.outputs.append(output)
            return capture_and_hook
    
        hooks = []
        for layer in self.model.vit.encoder.layer:
            hooks.append(layer.intermediate.dense.register_forward_hook(make_capture_and_hook()))
    
        inputs = {k: v.requires_grad_() for k, v in inputs.items()}
    
        outputs = self.model(**inputs, output_attentions=output_attentions)
    
        if class_index is None:
            class_index = outputs.logits.argmax(dim=1).item()
    
        target_logits = outputs.logits[:, class_index]
        target_logits.sum()
    
        for h in hooks:
            h.remove()
    
        # Torna come liste, oppure gestisci il controllo di shape per lo stack
        return outputs.logits, outputs.attentions, self.outputs#, self.gradients



    # def classify_and_capture_outputs(self, inputs, class_index=None, output_attentions=False):
    #     self.outputs = []
    #     self.gradients = []
    
    #     # Funzione per salvare il gradiente in self.gradients
    #     def save_gradient(grad):
    #         self.gradients.append(grad)
    
    #     # Hook: salviamo output e registriamo gradient hook
    #     hooks = []
    #     for layer in self.model.vit.encoder.layer:
    #         def capture_and_hook(module, input, output):
    #             output.retain_grad()  # necessario per .grad
    #             self.outputs.append(output)
    #             output.register_hook(save_gradient)
    
    #         # Crea un hook sul feedforward intermedio (es: MLP)
    #         hooks.append(layer.intermediate.dense.register_forward_hook(capture_and_hook))
    
    #     # Forward pass
    #     inputs = {k: v.requires_grad_() for k, v in inputs.items()}
    #     outputs = self.model(**inputs, output_attentions=output_attentions)
    
    #     # Classe target
    #     if class_index is None:
    #         class_index = outputs.logits.argmax(dim=1).item()
    
    #     # Backward pass sulla logit della classe
    #     target_logits = outputs.logits[:, class_index]
    #     self.model.zero_grad()
    #     target_logits.sum().backward()
    
    #     # Rimozione hook
    #     for h in hooks:
    #         h.remove()
    
    #     # Calcolo GradCAM per ogni layer
    #     cam_per_layer = []
    #     for out, grad in zip(self.outputs, self.gradients):
    #         # out, grad: [batch, seq_len, hidden_dim]
    #         # alpha_k = media sui token (dim 1)
    #         weights = grad.mean(dim=1)  # [batch, hidden_dim]
    
    #         # CAM = somma pesata dei token
    #         cam = (weights.unsqueeze(1) * out).sum(dim=-1)  # [batch, seq_len]
    #         cam = torch.relu(cam)
    #         cam_per_layer.append(cam.detach())  # [batch, seq_len]
    
    #     # cam_per_layer: lista lunga L di tensori [batch, seq_len]
    #     # Stack finale
    #     cam_per_layer = torch.stack(cam_per_layer, dim=0)  # [num_layers, batch, seq_len]
    
    #     return outputs.logits, outputs.attentions, torch.stack(self.outputs, dim=0), cam_per_layer












class DEIT_Hook:
    def __init__(self, model):
        self.model = model
        self.outputs = []
        self.gradients = []

    def sampling_hook(self, token_indices):
        def hook(module, input, output):
            cls_token = output[:, 0:1, :]
            dist_token = output[:, 1:2, :]
            token_embeddings = output[:, 2:, :]
            sampled_tokens = token_embeddings[:, token_indices, :]
            new_output = torch.cat([cls_token, dist_token, sampled_tokens], dim=1)
            return new_output
        return hook

    def output_hook(self, module, input, output):
        self.outputs.append(output)

    def classify_with_sampled_tokens(self, inputs, token_indices_list, class_index):
        """
        Classifies an input image by sampling tokens and returning class probabilities.
    
        Args:
            inputs (dict): Input data for the model.
            token_indices_list (list): List of token indices to be sampled during classification.
            class_index (int): Index of the target class for which the probability will be calculated.
    
        Returns:
            tuple of lists:
                - List of probabilities for the target class.
                - List of maximum softmax confidences.
                - List of predicted class indices.
        """
        true_class_probs = []
        top_confidences = []
        top_class_indices = []
    
        for token_indices in token_indices_list:
            token_indices = token_indices[token_indices != -1]
    
            # Register sampling hook
            hook = self.model.deit.embeddings.dropout.register_forward_hook(self.sampling_hook(token_indices))
    
            # Forward pass
            outputs = self.model(**inputs)
            hook.remove()
    
            # Get predictions
            predictions = outputs.logits.softmax(dim=-1)[0]  # shape: [num_classes]
            true_class_probs.append(predictions[class_index].item())
    
            top_conf, top_class_idx = predictions.max(dim=-1)
            top_confidences.append(top_conf.item())
            top_class_indices.append(top_class_idx.item())
    
        return true_class_probs, top_confidences, top_class_indices

    
    def classify_and_capture_outputs(self, inputs, class_index=None, output_attentions=False):
        self.outputs = []
    
        def make_capture_and_hook():
            def capture_and_hook(module, input, output):
                self.outputs.append(output)
            return capture_and_hook
    
        hooks = []
        for layer in self.model.deit.encoder.layer:
            hooks.append(layer.intermediate.dense.register_forward_hook(make_capture_and_hook()))
    
        inputs = {k: v.requires_grad_() for k, v in inputs.items()}
    
        outputs = self.model(**inputs, output_attentions=output_attentions)
    
        if class_index is None:
            class_index = outputs.logits.argmax(dim=1).item()
    
        target_logits = outputs.logits[:, class_index]
        target_logits.sum()
    
        for h in hooks:
            h.remove()
    
        # Torna come liste, oppure gestisci il controllo di shape per lo stack
        return outputs.logits, outputs.attentions, self.outputs#self.gradients



    
    
    # def classify_and_capture_outputs(self, inputs, class_index=None, output_attentions=False):
    #     self.outputs = []
    #     self.gradients = []
    
    #     # Funzione per salvare il gradiente in self.gradients
    #     def save_gradient(grad):
    #         self.gradients.append(grad)
    
    #     # Hook: salviamo output e registriamo gradient hook
    #     hooks = []
    #     for layer in self.model.vit.encoder.layer:
    #         def capture_and_hook(module, input, output):
    #             output.retain_grad()  # necessario per .grad
    #             self.outputs.append(output)
    #             output.register_hook(save_gradient)
    
    #         # Crea un hook sul feedforward intermedio (es: MLP)
    #         hooks.append(layer.intermediate.dense.register_forward_hook(capture_and_hook))
    
    #     # Forward pass
    #     inputs = {k: v.requires_grad_() for k, v in inputs.items()}
    #     outputs = self.model(**inputs, output_attentions=output_attentions)
    
    #     # Classe target
    #     if class_index is None:
    #         class_index = outputs.logits.argmax(dim=1).item()
    
    #     # Backward pass sulla logit della classe
    #     target_logits = outputs.logits[:, class_index]
    #     self.model.zero_grad()
    #     target_logits.sum().backward()
    
    #     # Rimozione hook
    #     for h in hooks:
    #         h.remove()
    
    #     # Calcolo GradCAM per ogni layer
    #     cam_per_layer = []
    #     for out, grad in zip(self.outputs, self.gradients):
    #         # out, grad: [batch, seq_len, hidden_dim]
    #         # alpha_k = media sui token (dim 1)
    #         weights = grad.mean(dim=1)  # [batch, hidden_dim]
    
    #         # CAM = somma pesata dei token
    #         cam = (weights.unsqueeze(1) * out).sum(dim=-1)  # [batch, seq_len]
    #         cam = torch.relu(cam)
    #         cam_per_layer.append(cam.detach())  # [batch, seq_len]
    
    #     # cam_per_layer: lista lunga L di tensori [batch, seq_len]
    #     # Stack finale
    #     cam_per_layer = torch.stack(cam_per_layer, dim=0)  # [num_layers, batch, seq_len]
    
    #     return outputs.logits, outputs.attentions, torch.stack(self.outputs, dim=0), cam_per_layer

