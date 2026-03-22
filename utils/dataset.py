import torch
from torchvision import datasets, transforms

def get_dataloader(data_path, batch_size=32, img_size=128):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,        # Faster loading
        pin_memory=True       # Important for GPU
    )

    return loader, dataset.classes