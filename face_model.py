import torch
import torch.nn as nn
import torchvision.models as models

def get_face_model(num_classes=7, small=False):
    """Return a model. If small=True, return a tiny CNN for fast CPU runs (useful for quick tests).

    Otherwise return ResNet18 without pretrained weights.
    """
    if small:
        # Very small CNN for fast iterations on CPU
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        return model

    # Avoid downloading pretrained weights by default to prevent hangs on slow/blocked networks.
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_face_model()
    print(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("Output shape:", output.shape)
