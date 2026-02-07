import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from models.detector import DiffNet

def test_diffnet():
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffNet(num_classes=80, img_size=(224, 224)).to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("\nOutput check:")
    print(f"Enhanced image shape: {outputs['enhanced'].shape}")
    print(f"Depth map shape: {outputs['depth'].shape}")
    print(f"Recovered image shape: {outputs['recovered'].shape}")
    
    if outputs['detections'] is not None:
        print(f"Number of detections: {len(outputs['detections'][0])}")
    
    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_diffnet()
