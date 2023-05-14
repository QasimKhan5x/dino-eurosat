import argparse

import timm
import torch
import torchvision.transforms as transforms
from evaluation import compute_knn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from utils import Head, MultiCropWrapper


def main():
    parser = argparse.ArgumentParser(
        "DINO training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--device", type=str, choices=("cpu", "cuda"), default="cuda"
    )
    parser.add_argument("-c", "--n-crops", type=int, default=4)
    parser.add_argument("-o", "--out-dim", type=int, default=1024)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()
    print(vars(args))
    # Parameters
    vit_name, dim = "vit_small_patch16_224_in21k", 384

    device = torch.device(args.device)

    n_workers = 48

    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    eurosat = ImageFolder("../data/EuroSAT_RGB")
    sizes = [int(len(eurosat) * 0.7), int(len(eurosat) * 0.3)]
    dataset_train_plain, dataset_val_plain = random_split(eurosat, sizes)

    dataset_train_plain.dataset.transform = transform_plain
    dataset_val_plain.dataset.transform = transform_plain

    data_loader_train_plain = DataLoader(
        dataset_train_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )
    data_loader_val_plain = DataLoader(
        dataset_val_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )

    # Neural network related
    student_vit = timm.create_model(vit_name, pretrained=args.pretrained)
    # student = torch.load("selfsupervised2/best_model.pth")
    student = MultiCropWrapper(
        student_vit,
        Head(
            dim,
            args.out_dim,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    student = student.to(device)
    student.eval()

    # KNN
    current_acc = compute_knn(
        student.backbone,
        data_loader_train_plain,
        data_loader_val_plain,
    )
    print("knn accuracy:", current_acc)


if __name__ == "__main__":
    main()

    # imagenet = 90.16%
    # ssl = 92.76%
    # imagenet + ssl = 97.23%
