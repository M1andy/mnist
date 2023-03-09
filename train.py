import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR

from models import ConvNet
from utils import parse_args, test
from utils import get_cifar10_dataloader, get_mnist_dataloader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic(False)
    # np.random.seed(seed)
    # random.seed(seed)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        acc = (output.argmax(1) == target).sum().item() / len(target)
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.3f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    acc,
                )
            )
            wandb.log({"Train Loss": loss.item(), "Train Accuracy": acc})


def main():
    args = parse_args()
    wandb_config = args
    wandb.init(
        project="Mnist-Classfication",
        tags=["ConvNet", "Mnist"],
        name=f"convnet-batch{args.batch_size}-lr{args.lr}",
        config=wandb_config,
    )
    setup_seed(args.seed)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}

    if torch.cuda.is_available() and args.cuda > 0:
        device = torch.device(f"cuda:{args.cuda}")

        # update cuda args
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        device = torch.device("cpu")

    if args.dataset == "mnist":
        train_loader, test_loader = get_mnist_dataloader(train_kwargs, test_kwargs)
    elif args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_dataloader(train_kwargs, test_kwargs)
    else:
        raise NotImplementedError

    model = ConvNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_accuracy = test(model, device, test_loader)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            wandb.run.summary["best_accuracy"] = best_accuracy
            if args.save_model:
                torch.save(model.state_dict(), "./ckpts/mnist_cnn.pt")

        scheduler.step()


if __name__ == "__main__":
    main()
