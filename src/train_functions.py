# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
    accuracies: list[float] = []
    model.train()

    for inputs, targets in train_data:
        # move inputs and targets to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        # zero gradients from previous steps
        optimizer.zero_grad()

        # forward pass
        outputs, _ = model(inputs)
        batch_loss = loss(outputs, targets)

        # backward pass and optimization
        batch_loss.backward()
        optimizer.step()

        # track loss and accuracy for the current batch
        losses.append(batch_loss.item())
        accuracies.append((outputs.argmax(dim=1) == targets).float().mean().item())

    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    model.eval()
    losses = []
    accuracies = []

    with torch.no_grad():
        for inputs, targets in val_data:
            # move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # forward pass
            outputs, _ = model(inputs)
            batch_loss = loss(outputs, targets)

            # track loss and accuracy for the current batch
            losses.append(batch_loss.item())
            accuracies.append((outputs.argmax(dim=1) == targets).float().mean().item())

    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)


def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.

    Returns:
        average accuracy.
    """

    model.eval()
    accuracies = []

    with torch.no_grad():
        for inputs, targets in test_data:
            # move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # compute predictions and track accuracy
            outputs, _ = model(inputs)
            accuracies.append((outputs.argmax(dim=1) == targets).float().mean().item())

    return np.mean(accuracies)
