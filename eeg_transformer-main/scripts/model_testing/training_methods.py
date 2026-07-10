import os
import numpy as np
import torch
import torch.nn as nn
import copy
import gc

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model_classes import TemporalTransformer
import data_loader as data

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total # (loss | acc)

def train_and_save_model_5fold(X_all, y_all, device, save_path="best_model.pth",
                               epochs=30, batch_size=8, lr=1e-4, d_model=32, n_head=4):
    """
    Trains Temporal Transformer using 5-Fold Cross Validation.
    Finds the best model and saves it.
    Returns the best model and a testing group for captum analysis.
    """

    # Data shape X_all (N, channels, time)
    n_channels = X_all.shape[1]

    print("=" * 60)
    print(f"Training starting (5-Fold CV)")
    print(f"Data: {len(X_all)} samples, {n_channels} channels, D_MODEL: {d_model}")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    best_global_acc = 0.0
    best_model_state = None
    best_test_loader_state = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        # Creating collections for current fold:
        train_ds = data.SimpleEEGDataset(X_all[train_idx], y_all[train_idx])
        test_ds = data.SimpleEEGDataset(X_all[test_idx], y_all[test_idx]) #test_idx?

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # In every fold initialize an empty model
        model = TemporalTransformer(
            input_size=n_channels,
            d_model=d_model,
            nhead=n_head,
            num_classes=2,
            feature_method='raw'
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_fold_acc = 0.0
        best_fold_weights = None

        # Training loop for this fold:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            # Track the best accuracy in the fold
            if test_acc > best_fold_acc:
                best_fold_acc = test_acc
                # Save deep copy
                best_fold_weights = copy.deepcopy(model.state_dict())
                is_best = "*"
            else:
                is_best = ""

            if epoch % 5 == 0 or is_best == "*":
                print(
                    f"  Ep {epoch:02d}/{epochs}: TrLoss: {train_loss:.4f}, TrAcc: {train_acc:.2%} | ValAcc: {test_acc:.2%} {is_best}")

        print(f"-> Best in fold {fold + 1}: {best_fold_acc:.2%}")
        fold_results.append(best_fold_acc)

        # Comparing best acc in fold with previous folds accuracy
        if best_fold_acc > best_global_acc:
            best_global_acc = best_fold_acc
            best_model_state = copy.deepcopy(best_fold_weights)
            best_test_loader_state = test_loader

        # Cleanup on memory
        del model
        del optimizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # End
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    print("\n" + "=" * 60)
    print("Training finished (5-FOLD CV)")
    print(f"Results for folds: {[f'{a:.2%}' for a in fold_results]}")
    print(f"average accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"Highest test-set accuracy: {best_global_acc:.2%}")
    print("=" * 60)

    # Recreating the best model state:
    final_model = TemporalTransformer(
        input_size=n_channels,
        d_model=d_model,
        nhead=n_head,
        num_classes=2,
        feature_method='raw'
    ).to(device)
    final_model.load_state_dict(best_model_state)

    # --- Zapisywanie na dysk ---
    torch.save(best_model_state, save_path)
    print(f"\n[SUKCES] Wagi najlepszego modelu zapisano pomyślnie w pliku: {save_path}")

    return final_model, best_test_loader_state


def train_cross_individual(subject_data, model_class, device, epochs=30, batch_size=32, lr=1e-3, d_model=128, nhead=8):
    """
    Performs a 5-fold cross-individual validation.
    Tracks and returns the best performing model across all folds, along with its specific test_loader.
    """
    all_subjects = np.array(list(subject_data.keys()))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_global_acc = 0.0
    best_global_model = None
    best_global_test_loader = None
    fold_accuracies = []

    print(f"\n--- Starting 5-Fold Cross-Individual Validation (Device: {device}) ---")

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_subjects)):
        print(f"Fold {fold + 1}/5")

        train_subjects = all_subjects[train_idx]
        test_subjects = all_subjects[test_idx]

        # Gather training data for current fold
        X_train_list, y_train_list = [], []
        for subj in train_subjects:
            X, y = subject_data[subj]
            X_train_list.append(X)
            y_train_list.append(y)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Gather testing data for current fold
        X_test_list, y_test_list = [], []
        for subj in test_subjects:
            X, y = subject_data[subj]
            X_test_list.append(X)
            y_test_list.append(y)

        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # Create Datasets and DataLoaders
        train_dataset = data.SimpleEEGDataset(X_train, y_train, is_train=True)
        test_dataset = data.SimpleEEGDataset(X_test, y_test, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Input size for the Model
        num_channels = X_train.shape[1] # For temporal transformer forward is applied as squeeze(1).permute(2, 0, 1), so the layer expects Channels as input size
        num_time_points = X_train.shape[2]

        model_name = model_class.__name__
        if "Temporal" in model_name:
            # Temporal: Time is the sequence, so the model needs to embed Channels (electrodes)
            model_input_size = num_channels
            print(f"[INFO] Configuring {model_name} -> input_size = {model_input_size} (Channels)")

        elif "Spatial" in model_name:
            # Spatial: Channels are the sequence, so the model needs to embed time points
            model_input_size = num_time_points
            print(f"[INFO] Configuring {model_name} -> input_size = {model_input_size} (Time points)")

        model = model_class(
            input_size=model_input_size,
            d_model=d_model,
            nhead=nhead,
            num_classes=2,
            feature_method='raw'
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_fold_acc = 0.0
        best_fold_state = None
        total_batches = len(train_loader)

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(x_batch)

                loss = criterion(outputs, y_batch)
                loss.backward()

                # added gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * x_batch.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)

                print(f"Batch [{batch_idx + 1:03d}/{total_batches}]", flush=True)

            # Evaluation for current epoch
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(x_batch)
                    preds = outputs.argmax(dim=1)
                    test_correct += (preds == y_batch).sum().item()
                    test_total += y_batch.size(0)

            train_acc = train_correct / train_total
            test_acc = test_correct / test_total

            # Save the best model state inside the current fold
            if test_acc > best_fold_acc:
                best_fold_acc = test_acc
                best_fold_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1:02d}/{epochs}] "
                      f"Loss: {train_loss / train_total:.4f} "
                      f"Train Acc: {train_acc * 100:.2f}% "
                      f"Test Acc: {test_acc * 100:.2f}%", flush=True)

        print(f"\nBest Test Accuracy for Fold {fold + 1}: {best_fold_acc * 100:.2f}%")
        fold_accuracies.append(best_fold_acc)

        # Check if model from this fold beat best model, if so assign it as best
        if best_fold_acc > best_global_acc:
            best_global_acc = best_fold_acc

            # Reconstruct the best model
            model.load_state_dict(best_fold_state)
            best_global_model = copy.deepcopy(model)
            best_global_test_loader = test_loader

    avg_accuracy = np.mean(fold_accuracies)
    print(f"\n--- Training finished (5-FOLD Cross-Individual) ---")
    print(f"Average 5-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Best Global Fold Accuracy: {best_global_acc * 100:.2f}%")

    # Returning the best model and its specific test_loader for further analysis
    return best_global_model, best_global_test_loader

def train_cross_individual_opt(subject_data, model_class, device, epochs=30, batch_size=32, lr=1e-3, d_model=128, nhead=8, dataset_name="GigaDB"):
    all_subjects = np.array(list(subject_data.keys()))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_global_acc = 0.0
    best_global_state = None  # Keep only weights (state_dict), not whole model
    best_global_test_loader = None
    fold_accuracies = []

    scaler = torch.cuda.amp.GradScaler()
    checkpoints_base_dir = f"./saved_model_states/checkpoints_{dataset_name}"
    os.makedirs(checkpoints_base_dir, exist_ok=True)
    print(f"\n--- Starting 5-Fold Cross-Individual Validation (Device: {device}) ---")

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_subjects)):
        print(f"Fold {fold + 1}/5")

        train_subjects = all_subjects[train_idx]
        test_subjects = all_subjects[test_idx]

        # Creating datasets based on references not concatinaded data*
        train_dataset = data.LazySubjectDataset(subject_data, train_subjects)
        test_dataset = data.LazySubjectDataset(subject_data, test_subjects)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True       # Faster Data transfer with pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Datasize based on the first patient
        sample_subj = train_subjects[0]
        num_channels = subject_data[sample_subj][0].shape[1]
        num_time_points = subject_data[sample_subj][0].shape[2]

        model_name = model_class.__name__
        if "Temporal" in model_name:
            model_input_size = num_channels
            print(f"[INFO] Configuring {model_name} -> input_size = {model_input_size} (Channels)")
        elif "Spatial" in model_name:
            model_input_size = num_time_points
            print(f"[INFO] Configuring {model_name} -> input_size = {model_input_size} (Time points)")

        model = model_class(
            input_size=model_input_size,
            d_model=d_model,
            nhead=nhead,
            num_classes=2,
            feature_method='raw'
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_fold_acc = 0.0
        best_fold_state = None
        total_batches = len(train_loader)

        fold_checkpoint_dir = os.path.join(checkpoints_base_dir, f"fold_{fold + 1}")
        os.makedirs(fold_checkpoint_dir, exist_ok=True)

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * x_batch.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)

            # Evaluation
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(x_batch)
                    preds = outputs.argmax(dim=1)
                    test_correct += (preds == y_batch).sum().item()
                    test_total += y_batch.size(0)

            train_acc = train_correct / train_total
            test_acc = test_correct / test_total

            epoch_checkpoint_path = os.path.join(fold_checkpoint_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_checkpoint_path)

            if test_acc > best_fold_acc:
                best_fold_acc = test_acc
                # Copyting only dictionary of weights
                best_fold_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1:02d}/{epochs}] "
                      f"Loss: {train_loss / train_total:.4f} "
                      f"Train Acc: {train_acc * 100:.2f}% "
                      f"Test Acc: {test_acc * 100:.2f}%", flush=True)

        print(f"\nBest Test Accuracy for Fold {fold + 1}: {best_fold_acc * 100:.2f}%")
        fold_accuracies.append(best_fold_acc)

        if best_fold_acc > best_global_acc:
            best_global_acc = best_fold_acc
            best_global_state = best_fold_state
            best_global_test_loader = test_loader

        # Memory cleanup after fold
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Reconstructing the best model based on weights to return model
    best_global_model = model_class(
        input_size=model_input_size,
        d_model=d_model,
        nhead=nhead,
        num_classes=2,
        feature_method='raw'
    ).to(device)
    best_global_model.load_state_dict(best_global_state)

    avg_accuracy = np.mean(fold_accuracies)
    print(f"\n--- Training finished (5-FOLD Cross-Individual) ---")
    print(f"Average 5-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Best Global Fold Accuracy: {best_global_acc * 100:.2f}%")

    return best_global_model, best_global_test_loader


# Wrapper handles forward, backward and optimizer.zero_grad()
class EEGLightningWrapper(pl.LightningModule):
    def __init__(self, model_class, input_size, d_model, nhead, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model_class'])  # logging hparams
        self.lr = lr

        self.model = model_class(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_classes=2,
            feature_method='raw'
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        preds = outputs.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Logowanie postępów (widoczne m.in. w pasku postępu w konsoli)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        preds = outputs.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer


def train_cross_individual_torch(subject_data, model_class, epochs=30, batch_size=32, lr=1e-3, d_model=128, nhead=8, dataset_name="GigaDB"):
    """
    Performs a 5-fold cross-individual validation using torch lightning libraries.
    Tracks and returns the best performing model across all folds, along with its specific test_loader.
    """

    all_subjects = np.array(list(subject_data.keys()))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    checkpoints_base_dir = f"./lightning_checkpoints_{dataset_name}"
    fold_accuracies = []

    print("\n--- Starting PyTorch Lightning 5-Fold Validation ---")

    # Setting the dimensions based on the first patient
    sample_subj = all_subjects[0]
    num_channels = subject_data[sample_subj][0].shape[1]
    num_time_points = subject_data[sample_subj][0].shape[2]

    model_name = model_class.__name__
    model_input_size = num_channels if "Temporal" in model_name else num_time_points

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_subjects)):
        fold_num = fold + 1
        print(f"\n================ Fold {fold_num}/5 ================")

        train_subjects = all_subjects[train_idx]
        test_subjects = all_subjects[test_idx]

        # Inicjalizacja Dataloaderów (Zakładam istnienie Twojej klasy LazySubjectDataset)
        train_loader = DataLoader(
            data.LazySubjectDataset(subject_data, train_subjects),
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            data.LazySubjectDataset(subject_data, test_subjects),
            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        fold_dir = os.path.join(checkpoints_base_dir, f"fold_{fold_num}")

        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_dir,
            filename="best-epoch{epoch:02d}-val_acc{val_acc:.2f}",
            monitor="val_acc",  # Saves the best model based on validation accuracy
            mode="max",
            save_top_k=1,  # Keeps only 1 best
            save_last=True,  # If True - saves "last.ckpt" after every epoch
            every_n_epochs=1  # Checks every epoch
        )

        # --- Initialization of Model and Trainer
        lightning_model = EEGLightningWrapper(model_class, model_input_size, d_model, nhead, lr)

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed",  # Adequate to torch.cuda.amp!
            gradient_clip_val=1.0,  # Adequate to clip_grad_norm_!
            callbacks=[checkpoint_callback],
            logger=False,
            enable_model_summary=True
        )

        # --- Resume logic after stop (trying to counter colab shutting down)
        last_ckpt_path = os.path.join(fold_dir, "last.ckpt")
        resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None

        if resume_ckpt:
            print(f"Fount previous checkpoint. Trying to resume training last session (Fold: {fold_num})")
        trainer.fit(lightning_model, train_loader, test_loader, ckpt_path=resume_ckpt)

        # Results from best model each fold
        best_model_path = checkpoint_callback.best_model_path
        best_score = checkpoint_callback.best_model_score

        if best_score is not None:
            fold_acc = best_score.item()
            fold_accuracies.append(fold_acc)
            print(f"Najlepsze Val Acc dla Fold {fold_num}: {fold_acc * 100:.2f}% (Zapisano w: {best_model_path})")

    avg_accuracy = np.mean(fold_accuracies)
    print(f"\n--- Trening PyTorch Lightning Zakończony ---")
    print(f"Średnia 5-Fold Accuracy: {avg_accuracy * 100:.2f}%")

    return fold_accuracies, checkpoints_base_dir