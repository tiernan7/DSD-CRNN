from pyexpat import model
from src.data import dataset
from src.data.data_loader import get_data_loaders, load_data
from src.models.loss_fn import RangeNormalizedMAE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ray import tune
from src.models.crnn import CRNN
from pathlib import Path
import tempfile
from ray.tune import Checkpoint


def train(config, dataset, null_choice, report = 100):
    # global constants pulled from the *base dataset*
    if null_choice == 'full':
        B = dataset.get_B_null_full().detach().cpu().numpy()         # for 
        extra_dims = B.shape[1] - dataset.get_B_null_true().detach().cpu().numpy().shape[1]
    elif null_choice == 'true':
        B = dataset.get_B_null_true().detach().cpu().numpy()         # for
    else:
        raise ValueError(f"Invalid null_choice: {null_choice}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_ind = dataset.get_output_index()
    out_ind = int(out_ind.item()) if torch.is_tensor(out_ind) else int(out_ind)
    
    num_reactions = config["num_reactions"]
    epochs = config["epochs"]
    lr = config["lr"]
    n_train = config["n_train"]
    n_val = config["n_val"]
    patience = config.get("patience", 200)
    min_delta = config.get("min_delta", 0.0)
    warmup = config.get("warmup", 0)
    
    model = CRNN(B, num_reactions).to(device)

    train_ds, val_ds, test_ds = load_data(dataset, n_train, n_val)

    train_loader,  val_loader, test_loader = get_data_loaders(
    train_ds, val_ds, test_ds,
    batch_size=n_train//2 if n_train >=2 else 1,
)

    # Loss function and optimizer
    loss_fn = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-8)

    t_train = dataset.get_t().to(device).float()

    bad_epochs = 0
    best_val = float("inf")
    train_loss = []
    val_loss = []

    # Training loss
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            c0 = batch["c0"].to(device).float()
            if null_choice == 'full':
                # Add columns of zeros to c0 to match full null space
                c0 = torch.cat([c0, torch.zeros(c0.shape[0], extra_dims, device=c0.device)], dim=1)
            y_true = batch["y"].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            y_full = model(t_train, c0)
            y_pred = y_full[:, :, out_ind] # (B, T, 1)
            loss_value = loss_fn(y_pred, y_true)
            loss_value.backward()
            optimizer.step()

            epoch_train_loss += loss_value.item()
        epoch_train_loss /= len(train_loader)
        train_loss.append(epoch_train_loss)


        # Validation loss
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0

            for batch_idx, batch in enumerate(val_loader):
                c0 = batch["c0"].to(device).float()
                if null_choice == 'full':
                    # Add columns of zeros to c0 to match full null space
                    c0 = torch.cat([c0, torch.zeros(c0.shape[0], extra_dims, device=c0.device)], dim=1)
                y_true = batch["y"].to(device).float()

                y_full = model(t_train, c0)
            # (B, T, N)
                y_pred = y_full[:, :, out_ind] # (B, T, 1)
                loss_value = loss_fn(y_pred, y_true)
                epoch_val_loss += loss_value.item()
            epoch_val_loss /= len(val_loader)
            val_loss.append(epoch_val_loss)
            improved = (best_val - epoch_val_loss) > min_delta
            if improved:
                best_val = epoch_val_loss
                bad_epochs = 0
            else:
                if epoch >= warmup:
                    bad_epochs += 1

            if epoch >= warmup and bad_epochs >= patience:
                # report one last time so Tune sees the final metric
                print(f"Early stopping at epoch {epoch}")
                print(f"Final training loss: {epoch_train_loss:.6f}, validation loss: {epoch_val_loss:.6f}")
                model.eval()
                model.cpu()
                return train_loss, val_loss, model, test_loader


        if epoch % report == 0:
            print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
            
    #final metrics
    print(f"Final training loss: {epoch_train_loss:.6f}, validation loss: {epoch_val_loss:.6f}")
    model.eval()
    model.cpu()
    return train_loss, val_loss, model, test_loader


def CRNN_tune(config, dataset):
    epochs = config["epochs"]
    num_reactions = config["num_reactions"]

    lr = config["lr"]

    patience = config.get("patience", 200)
    min_delta = config.get("min_delta", 0.0)
    warmup = config.get("warmup", 0)

    best_val = float("inf")
    bad_epochs = 0


    # global constants pulled from the *base dataset*
    A_control = dataset.get_A().detach().cpu().numpy()         # for scipy.null_spa
    out_ind = dataset.get_output_index()
    out_ind = int(out_ind.item()) if torch.is_tensor(out_ind) else int(out_ind)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CRNN(A_control, num_reactions).to(device)

    # Loss function and optimizer
    loss_fn = torch.nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-8)

    t_train = dataset.get_t().to(device).float()

    # Load checkpoint if resuming training
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            checkpoint_state = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint_state["epoch"] + 1
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            best_val = checkpoint_state.get("best_val", float("inf"))
            bad_epochs = checkpoint_state.get("bad_epochs", 0)
    else:
        start_epoch = 0

    
    # load dataset and construct data loaders
    train_ds, val_ds, test_ds = load_data(dataset, n_train=1, n_val=1)

    train_loader, val_loader, test_loader = get_data_loaders(
        train_ds, val_ds, test_ds,
        batch_size= 1,
    )
    
    # Training loss
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            c0 = batch["c0"].to(device).float()
            y_true = batch["y"].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            try:
                y_full = model(t_train, c0)
            except AssertionError as e:
                if "underflow in dt" in str(e):
                    tune.report({"train_loss": 1e30, "val_loss": 1e30})
                    return
                raise
            y_pred = y_full[:, :, out_ind] # (B, T, 1)
            loss_value = loss_fn(y_pred, y_true)
            loss_value.backward()
            optimizer.step()

            epoch_train_loss += loss_value.item()
        epoch_train_loss /= len(train_loader)


        # Validation loss
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0

            for batch_idx, batch in enumerate(val_loader):
                c0 = batch["c0"].to(device).float()
                y_true = batch["y"].to(device).float()

                try:
                    y_full = model(t_train, c0)
                except AssertionError as e:
                    if "underflow in dt" in str(e):
                        tune.report({"train_loss": 1e30, "val_loss": 1e30})
                        return
                    raise             # (B, T, N)
                y_pred = y_full[:, :, out_ind] # (B, T, 1)
                loss_value = loss_fn(y_pred, y_true)
                epoch_val_loss += loss_value.item()
            epoch_val_loss /= len(val_loader)
            improved = (best_val - epoch_val_loss) > min_delta
            if improved:
                best_val = epoch_val_loss
                bad_epochs = 0
            else:
                if epoch >= warmup:
                    bad_epochs += 1

            if epoch >= warmup and bad_epochs >= patience:
                # report one last time so Tune sees the final metric
                tune.report({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "early_stop": 1})
                return
            
            metrics = {
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "early_stop": 0
                }

        if epoch % 100 == 0 or improved:
            # Save checkpoint and report metrics
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "bad_epochs": bad_epochs,
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
                torch.save(checkpoint_data, checkpoint_path)
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
           
    # save checkpoint -> checkpoint_obj
                tune.report(metrics, checkpoint=checkpoint)
        else:
            tune.report(metrics)
            


def test_accuracy(net, dataset, device="cpu"):
    train_ds, val_ds, test_ds = load_data(dataset)

    train_loader, test_loader, val_loader = get_data_loaders(
        train_ds, val_ds, test_ds,
        batch_size= 1,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            image_batch, labels = data
            image_batch, labels = image_batch.to(device), labels.to(device)
            outputs = net(image_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total