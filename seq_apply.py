import os
import json
import tqdm
import torch
import click
from collections import defaultdict
import lightning.pytorch as pl
from pytorchvideo.transforms import Normalize, Permute, RandAugment
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
import sklearn.metrics as skm

from seq_dataset import SyntaxDataset
from seq_model import SyntaxLightningModule


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-w", "--weights-root", type=click.Path(exists=True), required=True, help="path to models weights.")
@click.option("--fold", type=int, required=True, help="fold number.")
@click.option("-nc", "--num-classes", type=int, default=2, help="num of classes of dataset.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(256, 256), help="frame per clip.")
@click.option("--max-epochs", type=int, default=30, help="max epochs.")
@click.option("--num-workers", type=int, default=16)
@click.option("--seed", type=int, default=42, help="random seed.")
def main(
    dataset_root,
    weights_root,
    fold,
    num_classes,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    seed,
):
    metrics = defaultdict(list)
    for variant in ("mean", "lstm_mean", "bert_mean"):
        run_fold(
            dataset_root,
            weights_root,
            fold,
            variant,
            num_classes,
            frames_per_clip,
            video_size,
            max_epochs,
            num_workers,
            seed,
            metrics
        )
    print(metrics)
    print(json.dumps(metrics))
    with open("folds_metrics_n_fold60exp.json", "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


def run_fold(
    dataset_root,
    weights_root,
    fold,
    variant,
    num_classes,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    seed,
    metrics
):
    left_bin_prob, left_bin, left_syntax, left_sids = run_fold_artery(
        dataset_root,
        weights_root,
        fold,
        "left",
        variant,
        num_classes,
        frames_per_clip,
        video_size,
        max_epochs,
        num_workers,
        seed,
        metrics
    )

    right_bin_prob, right_bin, right_syntax, right_sids = run_fold_artery(
        dataset_root,
        weights_root,
        fold,
        "right",
        variant,
        num_classes,
        frames_per_clip,
        video_size,
        max_epochs,
        num_workers,
        seed,
        metrics
    )

    assert len(left_sids) == len(right_sids)
    assert left_sids == right_sids

    src_path = f"rnn_folds/step2_rnn_fold{fold:02d}_test.json"
    full_src_path = os.path.join(dataset_root, src_path)
    dst_path = f"rnn_folds/step2_rnn_fold{fold:02d}_test.{variant}.json"
    full_dst_path = os.path.join(dataset_root, dst_path)

    dataset = json.load(open(full_src_path))

    assert len(dataset) == len(left_sids) == len(right_sids)
    syntax_true = []
    syntax_pred = []
    for rec, sid, l_prob, l_bin, l_syntax, r_prob, r_bin, r_syntax in zip(dataset, left_sids, left_bin_prob, left_bin, left_syntax, right_bin_prob, right_bin, right_syntax):
        assert rec["study_id"] == sid
        rec["prediction"] = {
            "left_prob": l_prob,
            "left_bin": l_bin,
            "left_syntax": l_syntax,
            "right_prob": r_prob,
            "right_bin": r_bin,
            "right_syntax": r_syntax,
            "syntax": l_syntax + r_syntax,
        }
        syntax_true.append(rec["syntax"])
        syntax_pred.append(l_syntax + r_syntax)
    r2 = skm.r2_score(syntax_true, syntax_pred)
    print("SYNTAX R2", r2)
    metrics[f"{variant}_syntax_r2"].append(r2)
    with open(full_dst_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


def run_fold_artery(
    dataset_root,
    weights_root,
    fold,
    artery,
    variant,
    num_classes,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    seed,
    metrics
):
    print("variant", variant)
    VARIANTS = "mean_out, mean, lstm_mean, lstm_last, gru_mean, gru_last, bert_mean, bert_cls".split(", ")
    assert variant in VARIANTS

    Artery = artery.capitalize()

    model_paths = {
        "pre_best": f"{weights_root}/{Artery}BinSyntax_R3D_fold{fold:02d}_{variant}_pre_best.pt",
        "post_best": f"{weights_root}/{Artery}BinSyntax_R3D_fold{fold:02d}_{variant}_post_best.pt",
    }

    for path in model_paths.values():
        if not os.path.isfile(path):
            print(path)

    pl.seed_everything(seed)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=video_size, antialias=True),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    src_path = f"rnn_folds/step2_rnn_fold{fold:02d}_test.json"

    test_set = SyntaxDataset(
        root=dataset_root,
        meta = src_path,
        train = False,
        length = frames_per_clip,
        label = f"syntax_{artery}",
        artery = artery,
        inference = True,
        transform=test_transform,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=1, #batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    model = SyntaxLightningModule(
        num_classes=num_classes,
        lr=1e-5,
        variant=variant,
        weight_decay=0.001,
        max_epochs=max_epochs,
    )

    model.load_state_dict(torch.load(model_paths["post_best"]))
    model.cuda()
    model.eval()

    Y = []
    Y_syntax = []
    P_bin_prob = []
    P_bin = []
    P_syntax = []
    sids = []

    with torch.no_grad():
        for x, [y], [t], [_weight_], [sid] in tqdm.tqdm(test_dataloader):
            if len(x.shape) == 1:
                bin_prob = 0.0
                val_syntax = 0.0
            else:
                x = x.cuda()
                [pred] = model(x)
                bin_logit, val_log = pred
                bin_prob = float(torch.sigmoid(bin_logit).cpu())
                val_syntax = max(0.0, float(torch.exp(val_log).cpu()) - 1)
                bin = round(bin_prob)
            y_syntax = max(0, float(torch.exp(t)) - 1)
            Y.append(y)
            Y_syntax.append(y_syntax)
            P_bin_prob.append(bin_prob)
            P_bin.append(bin)
            P_syntax.append(val_syntax)
            sids.append(sid)

    m = {
        "auc": skm.roc_auc_score(Y, P_bin_prob),
        "f1": skm.f1_score(Y, P_bin),
        "acc": skm.accuracy_score(Y, P_bin),
        "r2": skm.r2_score(Y_syntax, P_syntax),
    }

    print()
    print(variant, fold, artery)
    print(json.dumps(m, indent=4))
    for k, v in m.items():
        metrics[f"{variant}_{artery}_{k}"].append(v)

    return P_bin_prob, P_bin, P_syntax, sids



if __name__ == "__main__":
    main()
