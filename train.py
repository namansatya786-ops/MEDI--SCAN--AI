import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor

from mediscan.model import SkinLesionCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MediScan research CNN.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing one subfolder per class.")
    parser.add_argument("--output", type=Path, default=Path("models/skin_lesion_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def stratified_split(
    dataset: ImageFolder,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    indices_by_class: list[list[int]] = [[] for _ in dataset.classes]
    for index, class_index in enumerate(dataset.targets):
        indices_by_class[class_index].append(index)

    undersized_classes = [
        dataset.classes[index]
        for index, indices in enumerate(indices_by_class)
        if len(indices) < 2
    ]
    if undersized_classes:
        names = ", ".join(undersized_classes)
        raise SystemExit(f"Each class needs at least two images for training and validation. Too few: {names}.")

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in indices_by_class:
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_indices = [indices[position] for position in permutation]
        validation_count = min(
            len(indices) - 1,
            max(1, round(len(indices) * validation_fraction)),
        )
        validation_indices.extend(shuffled_indices[:validation_count])
        train_indices.extend(shuffled_indices[validation_count:])

    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise SystemExit("epochs and batch size must be positive; image size must be at least 32.")
    if not args.data_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.data_dir}")

    try:
        dataset = ImageFolder(
            args.data_dir,
            transform=Compose([Resize((args.image_size, args.image_size)), ToTensor()]),
        )
    except FileNotFoundError as error:
        raise SystemExit(
            "Could not find usable image classes. Add supported images to each class subfolder."
        ) from error
    if len(dataset.classes) < 2:
        raise SystemExit("Training data must contain at least two class subfolders.")

    train_set, validation_set = stratified_split(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    training_size = len(train_set)

    model = SkinLesionCNN(num_classes=len(dataset.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss / training_size:.4f} "
            f"validation_accuracy={correct / total:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": dataset.classes,
            "image_size": args.image_size,
        },
        args.output,
    )
    print(f"Saved research model to {args.output}")


if __name__ == "__main__":
    main()
