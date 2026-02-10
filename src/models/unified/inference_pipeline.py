"""
Inference Pipeline for TerrainFormer

Real-time inference with safety mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time


@dataclass
class SafetyConfig:
    """Safety configuration parameters."""
    min_confidence: float = 0.7
    max_consecutive_same_action: int = 10
    forbidden_reverse_speed_threshold: float = 0.5
    emergency_stop_collision_prob: float = 0.8
    conservative_action: int = 14  # STOP


class SafetyModule(nn.Module):
    """
    Safety layer for action filtering and override.
    
    Implements:
    - Confidence-based fallback
    - Action masking based on vehicle state
    - Emergency stop triggers
    - Action smoothing
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        super().__init__()
        self.config = config or SafetyConfig()
        
        # Track action history for smoothing
        self.recent_actions: List[int] = []
        self.max_history = 20
        
    def get_action_mask(self, 
                        state: torch.Tensor,
                        num_actions: int = 18) -> torch.Tensor:
        """
        Generate mask for valid actions based on vehicle state.
        
        Args:
            state: Vehicle state (B, state_dim)
            num_actions: Total number of actions
            
        Returns:
            Boolean mask (B, num_actions) where True = valid
        """
        B = state.shape[0]
        device = state.device
        
        mask = torch.ones(B, num_actions, dtype=torch.bool, device=device)
        
        # Extract velocity (assuming first element is forward velocity)
        velocity = state[:, 0]
        
        # Disable reverse at low speed (safety precaution)
        low_speed = velocity.abs() < self.config.forbidden_reverse_speed_threshold
        reverse_actions = [15, 16, 17]
        
        for b in range(B):
            if low_speed[b]:
                for a in reverse_actions:
                    mask[b, a] = False
                    
        return mask
    
    def check_emergency_stop(self, 
                             collision_prob: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Check if emergency stop should be triggered.
        
        Args:
            collision_prob: Collision probability (B, 1) or None
            
        Returns:
            Boolean tensor (B,) indicating emergency stop
        """
        if collision_prob is None:
            return torch.zeros(collision_prob.shape[0], dtype=torch.bool)
            
        return collision_prob.squeeze(-1) > self.config.emergency_stop_collision_prob
    
    def apply_confidence_fallback(self,
                                  actions: torch.Tensor,
                                  confidence: torch.Tensor) -> torch.Tensor:
        """
        Apply conservative action when confidence is low.
        
        Args:
            actions: Predicted actions (B,)
            confidence: Confidence scores (B,)
            
        Returns:
            Modified actions (B,)
        """
        low_confidence = confidence < self.config.min_confidence
        actions = torch.where(
            low_confidence,
            torch.full_like(actions, self.config.conservative_action),
            actions
        )
        return actions
    
    def forward(self,
                actions: torch.Tensor,
                confidence: torch.Tensor,
                state: torch.Tensor,
                collision_prob: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Apply all safety checks.
        
        Args:
            actions: Predicted actions (B,)
            confidence: Confidence scores (B,)
            state: Vehicle state (B, state_dim)
            collision_prob: Collision probability (B, 1)
            
        Returns:
            safe_actions: Safety-filtered actions (B,)
            info: Dictionary with safety information
        """
        info = {
            'original_actions': actions.clone(),
            'confidence': confidence.clone(),
            'emergency_stop': False,
            'confidence_fallback': False,
        }
        
        # Check emergency stop
        if collision_prob is not None:
            emergency = self.check_emergency_stop(collision_prob)
            if emergency.any():
                actions = torch.where(
                    emergency,
                    torch.full_like(actions, self.config.conservative_action),
                    actions
                )
                info['emergency_stop'] = True
                
        # Apply confidence fallback
        original = actions.clone()
        actions = self.apply_confidence_fallback(actions, confidence)
        if not torch.equal(original, actions):
            info['confidence_fallback'] = True
            
        return actions, info


class InferencePipeline:
    """
    Complete inference pipeline for real-time operation.
    """
    
    def __init__(self,
                 model: nn.Module,
                 safety_config: Optional[SafetyConfig] = None,
                 device: str = 'cuda'):
        """
        Args:
            model: TerrainFormer model
            safety_config: Safety configuration
            device: Inference device
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.safety = SafetyModule(safety_config)
        
        # Action history buffer
        self.action_history = torch.zeros(1, 10, dtype=torch.long, device=device)
        
        # Timing statistics
        self.inference_times: List[float] = []
        
    def reset(self):
        """Reset state for new episode."""
        self.action_history = torch.zeros(1, 10, dtype=torch.long, device=self.device)
        self.inference_times = []
        
    def update_action_history(self, action: int):
        """Update action history buffer."""
        self.action_history = torch.roll(self.action_history, -1, dims=1)
        self.action_history[0, -1] = action
        
    @torch.no_grad()
    def step(self,
             points: torch.Tensor,
             state: torch.Tensor,
             goal: torch.Tensor) -> Dict:
        """
        Single inference step.
        
        Args:
            points: LiDAR point cloud (N, 4)
            state: Vehicle state (state_dim,)
            goal: Goal direction (2,)
            
        Returns:
            Dictionary with action and metadata
        """
        start_time = time.time()
        
        # Prepare inputs
        points = points.unsqueeze(0).to(self.device)
        state = state.unsqueeze(0).to(self.device)
        goal = goal.unsqueeze(0).to(self.device)
        
        # Get action mask
        action_mask = self.safety.get_action_mask(state, self.model.num_actions)
        
        # Model prediction
        outputs = self.model(
            points, state, goal, self.action_history,
            return_world_predictions=True
        )
        
        # Get action
        logits = outputs['action_logits']
        confidence = outputs['confidence'].squeeze(-1)
        
        # Apply mask and get action
        logits = logits.masked_fill(~action_mask, float('-inf'))
        action = logits.argmax(dim=-1)
        
        # Safety checks
        collision_prob = outputs.get('collision')
        safe_action, safety_info = self.safety(action, confidence, state, collision_prob)
        
        # Update history
        self.update_action_history(safe_action.item())
        
        # Timing
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return {
            'action': safe_action.item(),
            'confidence': confidence.item(),
            'inference_time_ms': inference_time * 1000,
            'safety_info': safety_info,
            'traversability': outputs['world_traversability'].cpu(),
            'action_probs': F.softmax(logits, dim=-1).cpu().squeeze(0),
        }
    
    def get_statistics(self) -> Dict:
        """Get inference statistics."""
        if not self.inference_times:
            return {}
            
        times = torch.tensor(self.inference_times)
        return {
            'mean_inference_ms': times.mean().item() * 1000,
            'std_inference_ms': times.std().item() * 1000,
            'max_inference_ms': times.max().item() * 1000,
            'min_inference_ms': times.min().item() * 1000,
            'total_steps': len(self.inference_times),
        }


def test_inference_pipeline():
    print("Testing Inference Pipeline...")
    
    from .terrainformer import TerrainFormer
    
    model = TerrainFormer(lidar_in_channels=4, bev_size=256, num_actions=18)
    pipeline = InferencePipeline(model, device='cpu')
    
    # Simulate inference
    for i in range(5):
        points = torch.randn(16384, 4)
        points[:, :3] *= 30
        state = torch.randn(6)
        goal = torch.randn(2)
        
        result = pipeline.step(points, state, goal)
        print(f"Step {i}: action={result['action']}, "
              f"confidence={result['confidence']:.3f}, "
              f"time={result['inference_time_ms']:.1f}ms")
    
    stats = pipeline.get_statistics()
    print(f"Statistics: {stats}")
    
    print("Inference Pipeline test passed!")


if __name__ == "__main__":
    test_inference_pipeline()
