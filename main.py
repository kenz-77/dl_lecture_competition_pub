import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.fft import fft

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier, TransformerClassifier
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    # Preprocessing function
    def augment(data):
        # 元のデータを保持
        original_data = data.copy()

        # 時間シフト
        shift = np.random.randint(1, data.shape[0])
        augmented_data = np.roll(data, shift, axis=0)

        # ガウスノイズ追加
        noise = np.random.randn(*data.shape) * 0.1
        augmented_data += noise

        # 元のデータと増強されたデータの両方を返す
        return np.vstack((original_data, augmented_data))

    def extract_features(data):
        # FFT
        fft_data = fft(data, axis=0)
        # 絶対値を取って振幅スペクトルを取得
        amplitude = np.abs(fft_data)
        return amplitude

    def preprocess_data(data, lowcut=1, highcut=40.0, fs=1000.0, order=5, augment_flg=False):
        # Bandpass filter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        data = filtfilt(b, a, data, axis=0)
        
        # Normalize data
        data = (data - np.mean(data)) / np.std(data)

        if augment_flg:
          # Data augmentation
          data = augment(data)

        # Feature extraction
        data = extract_features(data)

        return data

    
    train_set = ThingsMEGDataset("train", args.data_dir, transform=lambda x: preprocess_data(x, augment_flg=True))
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir, transform=lambda x: preprocess_data(x))
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir, transform=lambda x: preprocess_data(x))
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    print('Finished data load')

    # ------------------
    #       Model
    # ------------------
    """
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)
    
    model = TransformerClassifier(
        num_classes=train_set.num_classes, 
        seq_len=train_set.seq_len, 
        in_channels=train_set.num_channels
    ).to(args.device)
    """
    model = TransformerClassifier(
        num_classes=train_set.num_classes, 
        seq_len=train_set.seq_len, 
        in_channels=train_set.num_channels
    ).to(args.device)
    
    
    # ------------------
    #     Optimizer
    # ------------------
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)  # L2正則化を追加


    # ------------------
    #     Scheduler
    # ------------------
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    patience = args.patience
    early_stop_counter = 0
    
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        mean_val_acc = np.mean(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {mean_val_acc:.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": mean_val_acc})
        
        if mean_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = mean_val_acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            cprint("Early stopping.", "red")
            break

    print('Finished training')
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
