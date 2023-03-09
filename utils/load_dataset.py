from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(train_kwargs, test_kwargs):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader


def get_cifar10_dataloader(train_kwargs, test_kwargs):
    train_loader = None
    test_loader = None
    return train_loader, test_loader
