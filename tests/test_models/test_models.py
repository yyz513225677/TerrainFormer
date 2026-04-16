"""Tests for model components."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_pointnet2():
    from models.lidar_encoder import PointNet2Backbone
    
    model = PointNet2Backbone(in_channels=4)
    points = torch.randn(2, 16384, 4)
    point_feat, global_feat = model(points)
    
    assert point_feat.shape[0] == 2
    assert global_feat.shape == (2, 1024)
    print("PointNet2 test passed!")


def test_world_model():
    from models.world_model import WorldModel
    
    model = WorldModel(bev_channels=64, embed_dim=512)
    bev = torch.randn(2, 64, 256, 256)
    outputs = model(bev)
    
    assert 'traversability' in outputs
    assert 'global_feature' in outputs
    print("World Model test passed!")


def test_decision_transformer():
    from models.decision import DecisionTransformer
    
    model = DecisionTransformer(world_model_dim=512, hidden_dim=384, num_actions=18)
    
    world_global = torch.randn(2, 512)
    world_latent = torch.randn(2, 64, 512)
    state = torch.randn(2, 6)
    goal = torch.randn(2, 2)
    action_history = torch.randint(0, 18, (2, 10))
    
    outputs = model(world_global, world_latent, state, goal, action_history)
    
    assert outputs['action_logits'].shape == (2, 18)
    print("Decision Transformer test passed!")


def test_terrainformer():
    from models import TerrainFormer
    
    model = TerrainFormer(lidar_in_channels=4, bev_size=256, num_actions=18)
    
    points = torch.randn(2, 16384, 4)
    points[:, :, :3] *= 30
    state = torch.randn(2, 6)
    goal = torch.randn(2, 2)
    action_history = torch.randint(0, 18, (2, 10))
    
    outputs = model(points, state, goal, action_history)
    
    assert 'action_logits' in outputs
    assert outputs['action_logits'].shape == (2, 18)
    
    actions, confidence = model.predict(points, state, goal, action_history)
    assert actions.shape == (2,)
    
    print("TerrainFormer test passed!")


if __name__ == '__main__':
    print("Running model tests...")
    test_pointnet2()
    test_world_model()
    test_decision_transformer()
    test_terrainformer()
    print("\nAll tests passed!")
