from typing import Tuple, List
import torch
import numpy as np
import os
import argparse
import copy # Needed for deepcopy of network for backward flow

# Assuming model.py, loss.py, and utils.py are in the same package
from .model import Neural_Prior
from .loss import my_chamfer_fn
from .utils import EarlyStopping # Timers and GeneratorWrap are not strictly needed for the API's core predict method


class SceneFlowPredictor:
    def __init__(self,
                 device: str = 'cpu',
                 iters: int = 5000,
                 lr: float = 0.008,
                 hidden_units: int = 128,
                 layer_size: int = 8,
                 act_fn: str = 'relu',
                 optimizer_type: str = 'adam',
                 weight_decay: float = 0, # Default from optimization.py
                 early_patience: int = 100,
                 backward_flow: bool = True
                ):
        """
        Initializes the SceneFlowPredictor for runtime optimization of Neural_Prior.
        The Neural_Prior model is optimized per-scene during the predict call, not loaded from pre-trained weights.

        Args:
            device (str): The device to run the optimization on ('cpu' or 'cuda').
            iters (int): Number of iterations to optimize the model for each prediction.
            lr (float): Learning rate for the optimizer.
            hidden_units (int): Number of hidden units (filter_size) in the Neural_Prior.
            layer_size (int): Number of layers in the Neural_Prior.
            act_fn (str): Activation function for the Neural_Prior (e.g., 'relu', 'sigmoid').
            optimizer_type (str): Type of optimizer to use ('adam' or 'sgd').
            weight_decay (float): Weight decay for the optimizer.
            early_patience (int): Patience for early stopping during optimization.
            backward_flow (bool): Whether to include backward flow loss in the optimization.
        """
        self.device = torch.device(device)

        # Store optimization hyperparameters
        # Using a Namespace for consistency with original optimization.py but
        # can also be directly stored as attributes.
        self.options = argparse.Namespace(
            iters=iters,
            lr=lr,
            hidden_units=hidden_units,
            layer_size=layer_size,
            act_fn=act_fn,
            optimizer=optimizer_type,
            weight_decay=weight_decay,
            early_patience=early_patience,
            backward_flow=backward_flow,
        )

        # Instantiate a fresh Neural_Prior model for this scene
        self.net = Neural_Prior(dim_x=3,
                                filter_size=self.options.hidden_units,
                                act_fn=self.options.act_fn,
                                layer_size=self.options.layer_size).to(self.device)

    def fit(self, pc1: torch.Tensor, pc2: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Predicts scene flow between two point clouds by optimizing a Neural_Prior model
        for the given scene.

        Args:
            pc1 (torch.Tensor): First point cloud, shape (N, 3) or (B, N, 3).
            pc2 (torch.Tensor): Second point cloud, shape (N, 3) or (B, N, 3).
                                 Should be on the same device as the predictor or compatible.

        Returns:
            torch.Tensor: Predicted scene flow vectors, shape (N, 3) or (B, N, 3).
                          This is the flow that warps pc1 to best match pc2,
                          on the same device as the predictor.
        """
        if pc1.shape != pc2.shape:
            raise ValueError("Point clouds must have the same shape.")
        if pc1.shape[-1] != 3:
            raise ValueError("Point clouds must have 3 coordinates (x, y, z).")

        # Ensure input tensors are on the correct device and have a batch dimension
        pc1_tensor = pc1.float().to(self.device)
        pc2_tensor = pc2.float().to(self.device)

        if pc1_tensor.dim() == 2: # (N, 3) -> (1, N, 3)
            pc1_tensor = pc1_tensor.unsqueeze(0)
        if pc2_tensor.dim() == 2: # (N, 3) -> (1, N, 3)
            pc2_tensor = pc2_tensor.unsqueeze(0)

        # Setup optimizer
        for param in self.net.parameters():
            param.requires_grad = True

        params_to_optimize = [{'params': self.net.parameters(), 'lr': self.options.lr, 'weight_decay': self.options.weight_decay}]
        
        if self.options.backward_flow:
            net_inv = copy.deepcopy(self.net)
            # Ensure net_inv parameters are also marked for optimization
            params_to_optimize.append({'params': net_inv.parameters(), 'lr': self.options.lr, 'weight_decay': self.options.weight_decay})
            
        if self.options.optimizer == "sgd":
            optimizer = torch.optim.SGD(params_to_optimize, momentum=0) # momentum is 0 in optimization.py for SGD, if not specified in options
        elif self.options.optimizer == "adam":
            optimizer = torch.optim.Adam(params_to_optimize, lr=self.options.lr, weight_decay=self.options.weight_decay) # Adam's weight_decay is often 0
        else:
            raise ValueError(f"Unknown optimizer: {self.options.optimizer}")

        early_stopping = EarlyStopping(patience=self.options.early_patience, min_delta=0.0001)

        best_flow_output = None
        best_loss_val = float('inf')

        total_losses = []

        # Optimization loop
        for epoch in range(self.options.iters):
            optimizer.zero_grad()

            flow_pred_1 = self.net(pc1_tensor)
            pc1_deformed = pc1_tensor + flow_pred_1
            # In optimization.py, normals are passed as None when calling my_chamfer_fn in solver.
            loss_chamfer_1, _ = my_chamfer_fn(pc2_tensor, pc1_deformed, None, None) 
            
            loss_chamfer = loss_chamfer_1
            if self.options.backward_flow:
                flow_pred_1_prime = net_inv(pc1_deformed)
                pc1_prime_deformed = pc1_deformed - flow_pred_1_prime
                loss_chamfer_1_prime, _ = my_chamfer_fn(pc1_prime_deformed, pc1_tensor, None, None)
                loss_chamfer += loss_chamfer_1_prime

            loss = loss_chamfer
            total_losses.append(loss.item())

            if loss.item() < best_loss_val:
                best_loss_val = loss.item()
                # Store the flow at the current best loss
                # Detach to avoid memory issues, keep as torch.Tensor on the device
                best_flow_output = (pc1_deformed - pc1_tensor).detach().squeeze(0) # Remove batch dimension

            if early_stopping.step(loss):
                break

            loss.backward()
            optimizer.step()
        
        # If no improvement was found or early stopping immediately,
        # return the flow from the last state
        if best_flow_output is None:
            with torch.no_grad():
                final_flow_pred = self.net(pc1_tensor)
                best_flow_output = (pc1_tensor + final_flow_pred - pc1_tensor).detach().squeeze(0)

        return best_flow_output, total_losses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the scene flow for a given point cloud using the optimized Neural_Prior model.
        Args:
            x (torch.Tensor): Input point cloud, shape (N, 3) or (B, N, 3).
        Returns:
            torch.Tensor: Predicted scene flow vectors, shape (N, 3) or (B, N, 3).
        """
        return self.net(x.to(self.device))


# Example usage (for testing purposes, not part of the library API itself)
if __name__ == '__main__':
    # This block will only run when the script is executed directly, not when imported.
    # Create dummy point clouds
    num_points = 2048 # Match typical usage in optimization.py
    # Create dummy point clouds as torch.Tensor
    pc1_dummy = torch.rand(num_points, 3, dtype=torch.float32)
    pc2_dummy = torch.rand(num_points, 3, dtype=torch.float32)

    print(f"PC1 shape: {pc1_dummy.shape}")
    print(f"PC2 shape: {pc2_dummy.shape}")

    try:
        # Initialize predictor with default optimization parameters
        # For a real scenario, you might want to adjust iters, lr, etc.
        predictor = SceneFlowPredictor(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            iters=1000, # Reduced iterations for quicker example run
            early_patience=50 # Reduced patience for quicker example run
        )

        # Pass torch.Tensor directly to predict_scene_flow
        scene_flow = predictor.predict_scene_flow(pc1_dummy, pc2_dummy)
        print(f"Predicted scene flow type: {type(scene_flow)}")
        print(f"Predicted scene flow shape: {scene_flow.shape}")
        print(f"Sample predicted scene flow:\n{scene_flow[:5]}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
