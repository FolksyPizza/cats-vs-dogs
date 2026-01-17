import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm


def build_transforms(img_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.25),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return train_tfms, val_tfms


def parse_args():
    parser = argparse.ArgumentParser(description="Cats vs Dogs training")
    parser.add_argument("--train-dir", type=str, default="/ssd/datasets/train")
    parser.add_argument("--val-dir", type=str, default="/ssd/datasets/val")
    parser.add_argument("--model", type=str, default="tf_efficientnetv2_s")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.4)
    parser.add_argument("--cutmix", type=float, default=0.1)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup-pct", type=float, default=0.08)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.log_dir,
        gradient_accumulation_steps=args.grad_accum_steps,
    )

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        accelerator.init_trackers("cats_vs_dogs", config=vars(args))

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_tfms, val_tfms = build_transforms(args.img_size)
    train_dataset = ImageFolder(args.train_dir, transform=train_tfms)
    val_dataset = ImageFolder(args.val_dir, transform=val_tfms)
    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )

    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=num_classes,
        drop_path_rate=args.drop_path,
    )
    model.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    use_mixup = args.mixup > 0.0 or args.cutmix > 0.0
    mixup_fn = None
    if use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            label_smoothing=args.label_smoothing,
            num_classes=num_classes,
        )
        train_criterion = SoftTargetCrossEntropy()
    else:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_pct,
        anneal_strategy="cos",
    )

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = torch.tensor(0.0, device=accelerator.device)
        epoch_count = torch.tensor(0.0, device=accelerator.device)

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for images, labels in progress:
            images = images.to(accelerator.device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(accelerator.device, non_blocking=True)

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            with accelerator.accumulate(model):
                outputs = model(images)
                loss = train_criterion(outputs, labels)
                accelerator.backward(loss)
                if args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss_sum += loss.detach() * images.size(0)
            epoch_count += images.size(0)
            global_step += 1

        accelerator.wait_for_everyone()
        train_loss = accelerator.gather_for_metrics(epoch_loss_sum).sum() / accelerator.gather_for_metrics(
            epoch_count
        ).sum()

        model.eval()
        val_loss_sum = torch.tensor(0.0, device=accelerator.device)
        val_correct = torch.tensor(0.0, device=accelerator.device)
        val_count = torch.tensor(0.0, device=accelerator.device)

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(
                    accelerator.device, non_blocking=True, memory_format=torch.channels_last
                )
                labels = labels.to(accelerator.device, non_blocking=True)
                outputs = model(images)
                loss = val_criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                val_loss_sum += loss.detach() * images.size(0)
                val_correct += (preds == labels).sum()
                val_count += images.size(0)

        accelerator.wait_for_everyone()
        val_loss = accelerator.gather_for_metrics(val_loss_sum).sum() / accelerator.gather_for_metrics(
            val_count
        ).sum()
        val_acc = accelerator.gather_for_metrics(val_correct).sum() / accelerator.gather_for_metrics(
            val_count
        ).sum()

        if accelerator.is_main_process:
            accelerator.log(
                {
                    "train/loss": train_loss.item(),
                    "val/loss": val_loss.item(),
                    "val/acc": val_acc.item(),
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                unwrapped = accelerator.unwrap_model(model)
                ckpt_path = os.path.join(args.save_dir, "best.pt")
                torch.save({"model": unwrapped.state_dict(), "epoch": epoch + 1}, ckpt_path)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        last_path = os.path.join(args.save_dir, "last.pt")
        torch.save({"model": unwrapped.state_dict(), "epoch": args.epochs}, last_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()
