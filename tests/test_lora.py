import torch
import torch.nn as nn
from trainer.methods.lora.injection import LoRAInjectionManager
from trainer.methods.lora.exporter import convert_trainer_to_kohya_format

def test_lora_injection_and_export():
    # Construct mock model components resembling diffusers/transformers structures
    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(128, 128)
            self.to_k = nn.Linear(128, 128)
            self.to_v = nn.Linear(128, 128)
            self.to_out = nn.Sequential(nn.Linear(128, 128))

    class MockUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttention()

    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)

    unet = MockUNet()
    te1 = MockEncoder()
    te2 = MockEncoder()

    # Initialize injection manager
    manager = LoRAInjectionManager(
        rank=16,
        alpha=8.0,
        unet_targets=["to_q", "to_k", "to_v"],
        text_encoder_1_targets=["q_proj"],
        text_encoder_2_targets=["k_proj"]
    )

    manager.inject_lora(unet, prefix="unet", targets=manager.unet_targets)
    manager.inject_lora(te1, prefix="te1", targets=manager.te1_targets)
    manager.inject_lora(te2, prefix="te2", targets=manager.te2_targets)

    # Check that layers are wrapped properly
    assert hasattr(unet.attn.to_q, "lora_down")
    assert hasattr(unet.attn.to_q, "lora_up")
    assert not hasattr(unet.attn.to_out[0], "lora_down") # should not be injected

    assert hasattr(te1.q_proj, "lora_down")
    assert not hasattr(te1.k_proj, "lora_down") # should not be injected

    assert not hasattr(te2.q_proj, "lora_down") # should not be injected
    assert hasattr(te2.k_proj, "lora_down")

    # Get optimizable parameters
    params = manager.get_lora_parameters()
    # Down + Up for 3 unet layers, 1 te1 layer, 1 te2 layer = 5 layers * 2 parameters = 10 tensors
    assert len(params) == 10

    # Verify state dict key format
    state_dict = manager.get_lora_state_dict()
    assert "unet.attn.to_q.lora_down.weight" in state_dict
    assert "te1.q_proj.lora_down.weight" in state_dict

    # Verify exporter Kohya formatting conversion
    kohya_dict = convert_trainer_to_kohya_format(state_dict, alpha=8.0)
    assert "lora_unet_attn_to_q.lora_down.weight" in kohya_dict
    assert "lora_unet_attn_to_q.alpha" in kohya_dict
    assert kohya_dict["lora_unet_attn_to_q.alpha"].item() == 8.0

    assert "lora_te1_q_proj.lora_down.weight" in kohya_dict
    assert "lora_te2_k_proj.lora_down.weight" in kohya_dict
