from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

from matdeeplearn.common.registry import registry


@registry.register_loss("TorchLossWrapper")
class TorchLossWrapper(nn.Module):
    def __init__(self, loss_fn="l1_loss"):
        super().__init__()
        self.loss_fn = getattr(F, loss_fn)

    def forward(self, predictions: torch.Tensor, target: Batch):    
        return self.loss_fn(predictions["output"], target.y)


@registry.register_loss("ForceLoss")
class ForceLoss(nn.Module):
    def __init__(self, weight_energy=1.0, weight_force=0.1):
        super().__init__()
        self.weight_energy = weight_energy
        self.weight_force = weight_force

    def forward(self, predictions: torch.Tensor, target: Batch):  
        combined_loss = self.weight_energy*F.l1_loss(predictions["output"], target.y) + self.weight_force*F.l1_loss(predictions["pos_grad"], target.forces)
        return combined_loss


@registry.register_loss("ForceStressLoss")
class ForceStressLoss(nn.Module):
    def __init__(self, weight_energy=1.0, weight_force=0.1, weight_stress=0.1):
        super().__init__()
        self.weight_energy = weight_energy
        self.weight_force = weight_force
        self.weight_stress = weight_stress

    def forward(self, predictions: torch.Tensor, target: Batch):  
        combined_loss = self.weight_energy*F.l1_loss(predictions["output"], target.y) + self.weight_force*F.l1_loss(predictions["pos_grad"], target.forces) + self.weight_stress*F.l1_loss(predictions["cell_grad"], target.stress)
        return combined_loss
        

@registry.register_loss("DOSLoss")
class DOSLoss(nn.Module):
    def __init__(
        self,
        loss_fn="l1_loss",
        scaling_weight=0.05,
        cumsum_weight=0.005,
        features_weight=0.15,
    ):
        super().__init__()
        self.loss_fn = getattr(F, loss_fn)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling_weight = scaling_weight
        self.cumsum_weight = cumsum_weight
        self.features_weight = features_weight

    def forward(self, predictions: tuple[torch.Tensor, torch.Tensor], target: Batch):
        out, scaling = predictions

        dos_loss = self.loss_fn(out, target.scaled)
        scaling_loss = self.loss_fn(scaling, target.scaling_factor)

        output_cumsum = torch.cumsum(out, axis=1)
        dos_cumsum = torch.cumsum(target.scaled, axis=1)
        dos_cumsum_loss = self.loss_fn(output_cumsum, dos_cumsum)

        scaled_dos = out * scaling.view(-1, 1).expand_as(out)
        x = torch.linspace(-10, 10, 400).to(scaled_dos)
        features = self.get_dos_features(x, scaled_dos)
        features_loss = self.loss_fn(target.features, features.to(self.device))

        loss_sum = (
            dos_loss
            + scaling_loss * self.scaling_weight
            + dos_cumsum_loss * self.cumsum_weight
            + features_loss * self.features_weight
        )

        return loss_sum

    def get_dos_features(self, x, dos):
        """get dos features"""
        dos = torch.abs(dos)
        dos_sum = torch.sum(dos, axis=1)

        center = torch.sum(x * dos, axis=1) / dos_sum
        x_offset = (
            torch.repeat_interleave(x[np.newaxis, :], dos.shape[0], axis=0)
            - center[:, None]
        )
        width = torch.diagonal(torch.mm((x_offset**2), dos.T)) / dos_sum
        skew = torch.diagonal(torch.mm((x_offset**3), dos.T)) / dos_sum / width**1.5
        kurtosis = (
            torch.diagonal(torch.mm((x_offset**4), dos.T)) / dos_sum / width**2
        )

        # find zero index (fermi level)
        zero_index = torch.abs(x - 0).argmin().long()
        ef_states = torch.sum(dos[:, zero_index - 20 : zero_index + 20], axis=1) * abs(
            x[0] - x[1]
        )
        return torch.stack((center, width, skew, kurtosis, ef_states), axis=1)


@registry.register_loss("DistillationLoss")
class DistillationLoss(nn.Module):
    def __init__(self,
            weight_energy=1.0, 
            weight_force=0.1,
            weight_stress=0.1, 
            weight_distillation=50.0, 
            distill_fns = "node2node", 
            use_mae = True, 
            use_huber = False, 
            preprocess_teacher_features = False,
            attention_weight = False
        ):
            super().__init__()
            self.weight_energy = weight_energy
            self.weight_force = weight_force
            self.weight_stress = weight_stress
            self.distill_fns = distill_fns
            self.use_mae = use_mae
            self.use_huber = use_huber
            self.weight_distillation = weight_distillation
            self.preprocess_teacher_features = preprocess_teacher_features
            self.attention_weight = attention_weight

    def forward(self, predictions: torch.Tensor, target: Batch):  
        total_loss = self.weight_energy*F.l1_loss(predictions["s_out"]["output"], target.y) + \
                     self.weight_force*F.l1_loss(predictions["s_out"]["pos_grad"], target.forces) + \
                     self.weight_stress*F.l1_loss(predictions["s_out"]["cell_grad"], target.stress)

        distill_fns = self.distill_fns.split("_")
        if self.attention_weight:
            attention_types = {'node2node':'n2n', 'edge2node':'e2n', 'edge2edge':'e2e', 'vec2vec':'v2v'}
            attention_scores = []

            for loss_name in distill_fns:
                attn_type = attention_types[loss_name]
                k_attention = predictions["t_out"][f'{attn_type}_attention']
                q_attention = predictions["s_out"][f'{attn_type}_attention']

                elementwise_product = k_attention * q_attention
                elementwise_product = torch.sum(elementwise_product, dim=1)
                attention_scores.append(elementwise_product)


            attention_scores = torch.stack(attention_scores, dim=1)
            softmax_scores = F.softmax(attention_scores, dim=1)
        
        for loss_idx, loss_name in enumerate(distill_fns):
            method_name = f"_{loss_name}_distill_loss"
            loss_method = getattr(self, method_name)

            current_loss = loss_method(predictions, target)
            if self.attention_weight:
                current_loss = current_loss*softmax_scores[:,loss_idx]
                current_loss = torch.mean(current_loss)

            total_loss += self.weight_distillation * current_loss

        return total_loss
    
    def _node2node_distill_loss(self, out_batch, target):
        if self.preprocess_teacher_features:
            target = target.embedding
            n2n_mappings = []

            for batch in target:
                for item in batch:
                    n2n_mapping = item['n2n_mapping']
                    if n2n_mapping.device.type == 'cuda':
                        n2n_mappings.append(n2n_mapping)
                    else:
                        n2n_mappings.append(n2n_mapping.to('cuda'))
            target = torch.cat(n2n_mappings, dim=0)
        else:
            target = out_batch["t_out"]["n2n_mapping"]

        reduction = 'none' if self.attention_weight else 'mean'

        # Dynamic loss function selection
        if self.use_mae:
            current_loss = torch.nn.functional.l1_loss(
                out_batch["s_out"]["n2n_mapping"],
                target,
                reduction=reduction
            )
        elif self.loss_function == 'huber':
            current_loss = torch.nn.functional.huber_loss(
                out_batch["s_out"]["n2n_mapping"],
                target,
                delta=self.huber_delta,
                reduction=reduction
            )
        else:  # Default to MSE
            current_loss = torch.nn.functional.mse_loss(
                out_batch["s_out"]["n2n_mapping"],
                target,
                reduction=reduction
            )
        if reduction == 'none':
            current_loss = torch.mean(current_loss, dim=1)
        return current_loss

    # def _node2node_distill_loss(self, out_batch, target):
    #     if self.preprocess_teacher_features:
    #         target = target.embedding
    #         n2n_mappings = []

    #         for batch in target:
    #             for item in batch:
    #                 n2n_mapping = item['n2n_mapping']
    #                 if n2n_mapping.device.type == 'cuda':
    #                     n2n_mappings.append(n2n_mapping)
    #                 else:
    #                     n2n_mappings.append(n2n_mapping.to('cuda'))
    #         target = torch.cat(n2n_mappings, dim=0)
    #     else:
    #         target = out_batch["t_out"]["n2n_mapping"]

    #     if self.use_mae and self.attention_weight==False:
    #         return torch.nn.functional.l1_loss(
    #             out_batch["s_out"]["n2n_mapping"],
    #             target,
    #         )
    #     elif self.use_mae and self.attention_weight:
    #         return torch.nn.functional.l1_loss(
    #             out_batch["s_out"]["n2n_mapping"],
    #             target,
    #             reduction='none'
    #         )

    #     elif self.use_huber and self.attention_weight==False:
    #         return torch.nn.functional.huber_loss(
    #             out_batch["s_out"]["n2n_mapping"],
    #             target,
    #             delta=self.huber_delta,
    #         )
    #     elif self.use_huber and self.attention_weight:
    #         return torch.nn.functional.huber_loss(
    #             out_batch["s_out"]["n2n_mapping"],
    #             target,
    #             delta=self.huber_delta,
    #             reduction='none'
    #         )
    #     elif self.attention_weight:
    #         return torch.nn.functional.mse_loss(
    #             out_batch["s_out"]["n2n_mapping"],
    #             target,
    #             reduction='none'
    #         )
    #     else:
    #         return torch.nn.functional.mse_loss(
    #             out_batch["s_out"]["n2n_mapping"],
    #             target,
    #         )

    def _edge2node_distill_loss(self, out_batch, target):
        if self.preprocess_teacher_features:
            target = target.embedding
            n2n_mappings = []

            for batch in target:
                for item in batch:
                    n2n_mapping = item['e2n_mapping']
                    if n2n_mapping.device.type == 'cuda':
                        n2n_mappings.append(n2n_mapping)
                    else:
                        n2n_mappings.append(n2n_mapping.to('cuda'))
            target = torch.cat(n2n_mappings, dim=0)
        else:
            target = out_batch["t_out"]["e2n_mapping"]

        reduction = 'none' if self.attention_weight else 'mean'

        # Dynamic loss function selection
        if self.use_mae:
            current_loss = torch.nn.functional.l1_loss(
                out_batch["s_out"]["e2n_mapping"],
                target,
                reduction=reduction
            )
        elif self.loss_function == 'huber':
            current_loss = torch.nn.functional.huber_loss(
                out_batch["s_out"]["e2n_mapping"],
                target,
                delta=self.huber_delta,
                reduction=reduction
            )
        else:  # Default to MSE
            current_loss = torch.nn.functional.mse_loss(
                out_batch["s_out"]["e2n_mapping"],
                target,
                reduction=reduction
            )

        if reduction == 'none':
            current_loss = torch.mean(current_loss, dim=1)
        return current_loss

    def _edge2edge_distill_loss(self, out_batch, target):
        if self.preprocess_teacher_features:
            target = target.embedding
            e2e_mappings = []

            for batch in target:
                for item in batch:
                    e2e_mapping = item['e2e_mapping']
                    if e2e_mapping.device.type == 'cuda':
                        e2e_mappings.append(e2e_mapping)
                    else:
                        e2e_mappings.append(e2e_mapping.to('cuda'))
            target = torch.cat(e2e_mappings, dim=0)
        else:
            target = out_batch["t_out"]["e2e_mapping"]
        
        reduction = 'none' if self.attention_weight else 'mean'

        if self.use_mae:
            current_loss = torch.nn.functional.l1_loss(
                out_batch["s_out"]["e2e_mapping"],
                target,
                reduction=reduction
            )
        elif self.loss_function == 'huber':
            current_loss = torch.nn.functional.huber_loss(
                out_batch["s_out"]["e2e_mapping"],
                target,
                delta=self.huber_delta,
                reduction=reduction
            )
        else:  # Default to MSE
            current_loss = torch.nn.functional.mse_loss(
                out_batch["s_out"]["e2e_mapping"],
                target,
                reduction=reduction
            )

        if reduction == 'none':
            current_loss = torch.mean(current_loss, dim=1)
        return current_loss


    def _vec2vec_distill_loss(self, out_batch, target):
        if self.preprocess_teacher_features:
            target = target.embedding
            v2v_mappings = []

            for batch in target:
                for item in batch:
                    v2v_mapping = item['v2v_mapping']
                    if v2v_mapping.device.type == 'cuda':
                        v2v_mappings.append(v2v_mapping)
                    else:
                        v2v_mappings.append(v2v_mapping.to('cuda'))
            target = torch.cat(v2v_mappings, dim=0)
        else:
            target = out_batch["t_out"]["v2v_mapping"]
        
        reduction = 'none' if self.attention_weight else 'mean'

        if self.use_mae:
            current_loss = torch.nn.functional.l1_loss(
                out_batch["s_out"]["v2v_mapping"],
                target,
                reduction=reduction
            )
        elif self.loss_function == 'huber':
            current_loss = torch.nn.functional.huber_loss(
                out_batch["s_out"]["v2v_mapping"],
                target,
                delta=self.huber_delta,
                reduction=reduction
            )
        else:  # Default to MSE
            current_loss = torch.nn.functional.mse_loss(
                out_batch["s_out"]["v2v_mapping"],
                target,
                reduction=reduction
            )
    
        if reduction == 'none':
            current_loss = torch.mean(current_loss, dim=1)
        return current_loss
