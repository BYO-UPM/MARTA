import torch
import numpy as np
from models.pt_models import (
    finetuning_model,
    Selec_embedding,
    Selec_model_two_classes,
)
from losses.pt_losses import SimCLR, GE2E_Loss
import timm
import timm.scheduler
import itertools
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
import wandb
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.utils import cluster_acc, nmi
from tqdm import tqdm
import time
from imblearn.under_sampling import RandomUnderSampler


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, "label array must be 1D"
        n_batches = int(len(y) / batch_size)
        self.batch_size = batch_size
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for _, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y) // self.batch_size


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0, path_save="model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.path_save = path_save

    def early_stop(self, model, optimizer, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # ------------------------------------------
            # Additional information
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                },
                self.path_save,
            )
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EarlyStopping_2metrics:
    def __init__(self, patience=1, min_delta=0, path_save="model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_performance_metric = np.inf
        self.min_validation_loss = np.inf
        self.path_save = path_save

    def early_stop(self, model, optimizer, performance_metric, validation_loss, epoch):
        if (performance_metric < self.min_performance_metric) or (
            (performance_metric == self.min_performance_metric)
            and (validation_loss < self.min_validation_loss)
        ):
            if performance_metric < self.min_performance_metric:
                self.min_performance_metric = performance_metric
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
            self.counter = 0
            # ------------------------------------------
            # Additional information
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                },
                self.path_save,
            )
        elif performance_metric > (self.min_performance_metric + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def Training_CL(
    model_cl,
    training_generator,
    val_generator,
    training_epochs=120,
    cooldown_epochs=30,
    batch_size=4,
    lr_cl=0.00001,
    project_name="model",
    save_path="model_checkpoints",
    patience=8,
    wandb=[],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cl.to(device)

    loss_function_lp = torch.nn.CrossEntropyLoss()
    loss_function_cl = SimCLR(batch_size=batch_size)

    num_epochs = training_epochs + cooldown_epochs

    optimizer_cl = torch.optim.Adam(
        itertools.chain(
            model_cl.embedding.parameters(), model_cl.projection_head.parameters()
        ),
        lr=lr_cl,
    )
    optimizer_lp = torch.optim.Adam(
        itertools.chain(
            model_cl.embedding.parameters(), model_cl.linear_probe.parameters()
        ),
        lr=lr_cl,
    )
    scheduler_cl = timm.scheduler.CosineLRScheduler(
        optimizer_cl, t_initial=training_epochs
    )
    scheduler_lp = timm.scheduler.CosineLRScheduler(
        optimizer_lp, t_initial=training_epochs
    )

    path_save = os.path.join(save_path, project_name + ".pt")
    early_stopping = EarlyStopping(
        patience=patience, min_delta=0.02, path_save=path_save
    )

    for epoch in range(num_epochs):
        num_steps_per_epoch = len(training_generator)
        num_updates = epoch * num_steps_per_epoch
        running_loss = 0.0
        model_cl.train()  # Optional when not using Model Specific layer
        for batch in training_generator:
            (inputs1a, inputs1b), targets = batch

            if torch.cuda.is_available():
                inputs1a = inputs1a.cuda()
                inputs1b = inputs1b.cuda()
                targets = targets.cuda()
                loss_function_cl = loss_function_cl.cuda()

            projectionXa = model_cl.forward(inputs1a)
            projectionXb = model_cl.forward(inputs1b)
            loss_cl = loss_function_cl(projectionXa, projectionXb)

            if wandb:
                wandb.log({"loss_cl": loss_cl})

            loss_cl.backward()
            optimizer_cl.step()
            scheduler_cl.step_update(num_updates=num_updates)

            # print statistics
            running_loss += loss_cl.item()
            optimizer_cl.zero_grad()

            # --------------------------------------------------
            outputs = model_cl.calculate_linear_probe(inputs1a)
            loss_lp = loss_function_lp(outputs, targets)

            if wandb:
                wandb.log({"loss_lp": loss_lp})

            loss_lp.backward()
            optimizer_lp.step()
            scheduler_lp.step_update(num_updates=num_updates)

            optimizer_lp.zero_grad()

        valid_loss = 0.0
        model_cl.eval()
        for data, labels in val_generator:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            target = model_cl.calculate_linear_probe(data)
            loss = loss_function_lp(target, labels)
            valid_loss = valid_loss + loss.item()

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {running_loss / len(training_generator)} \t\t Validation Loss: {valid_loss / len(val_generator)}"
        )

        scheduler_cl.step(epoch + 1)
        scheduler_lp.step(epoch + 1)
        if early_stopping.early_stop(
            model_cl, optimizer_cl, valid_loss / len(val_generator), epoch
        ):
            break
    return path_save


def Training_fine_tunning_ge2e(
    model,
    training_generator,
    val_generator,
    project_name="model",
    save_path="model_checkpoints",
    class_weights=[1, 1],
    training_epochs=30,
    cooldown_epochs=10,
    lr_ft=0.00001,
    patience=10,
    wandb=[],
    lam=0.6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # defining the loss function
    class_weights = torch.FloatTensor(class_weights).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    cl_loss = GE2E_Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    path_save = os.path.join(save_path, project_name + ".pt")
    early_stopping = EarlyStopping_2metrics(
        patience=patience, min_delta=0.05, path_save=path_save
    )

    num_epochs = training_epochs + cooldown_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        acc_r = 0.0
        model.train()  # Optional when not using Model Specific layer
        for batch in training_generator:
            inputs, targets = batch

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            output_train = model(inputs)
            output_emb = model.forward_emb(inputs)

            loss_clasi = loss_function(output_train, targets)
            loss_ge2e = cl_loss(output_emb, targets)
            loss = lam * loss_clasi + (1 - lam) * loss_ge2e

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            optimizer.zero_grad()

            pred = np.argmax(output_train.cpu().detach().numpy(), axis=1)
            acc = accuracy_score(targets.cpu().detach().numpy(), pred, normalize=True)
            acc_r += acc

        acc_r = acc_r / len(training_generator)
        running_loss = running_loss / len(training_generator)

        if wandb:
            wandb.log({"train_acc": acc_r})
            wandb.log({"loss_ft": running_loss})

        # ---------------- validation-------------------------------------
        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer

        f1_r = 0.0
        for data, labels in val_generator:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            output_val = model(data)
            loss_val = loss_function(output_val, labels)
            valid_loss += loss_val.item()

            pred = np.argmax(output_val.cpu().detach().numpy(), axis=1)

            if len(pred.shape) == 0:
                pred = np.expand_dims(np.array(pred), axis=0)
            f1 = f1_score(labels.cpu().detach().numpy(), pred)
            f1_r += f1
        f1_r = f1_r / len(val_generator)
        valid_loss = valid_loss / len(val_generator)
        if wandb:
            wandb.log({"val_f1": f1_r})
            wandb.log({"val_loss_ft": valid_loss})

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {running_loss} \t\t Validation Loss: {valid_loss}"
        )

        scheduler.step()

        if early_stopping.early_stop(model, optimizer, -f1_r, valid_loss, epoch):
            break

    return path_save


def Training_fine_tunning(
    model,
    training_generator,
    val_generator,
    project_name="model",
    save_path="model_checkpoints",
    class_weights=[1, 1],
    training_epochs=30,
    cooldown_epochs=10,
    lr_ft=0.00001,
    patience=10,
    wandb=[],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # defining the loss function
    class_weights = torch.FloatTensor(class_weights).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    path_save = os.path.join(save_path, project_name + ".pt")
    early_stopping = EarlyStopping_2metrics(
        patience=patience, min_delta=0.05, path_save=path_save
    )

    num_epochs = training_epochs + cooldown_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        acc_r = 0.0
        model.train()  # Optional when not using Model Specific layer
        for batch in training_generator:
            inputs, targets = batch

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            output_train = model(inputs)
            loss = loss_function(output_train, targets)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            optimizer.zero_grad()

            pred = np.argmax(output_train.cpu().detach().numpy(), axis=1)
            acc = accuracy_score(targets.cpu().detach().numpy(), pred, normalize=True)
            acc_r += acc

        acc_r = acc_r / len(training_generator)
        running_loss = running_loss / len(training_generator)

        if wandb:
            wandb.log({"train_acc": acc_r})
            wandb.log({"loss_ft": running_loss})

        # ---------------- validation-------------------------------------
        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer

        f1_r = 0.0
        for data, labels in val_generator:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            output_val = model(data)
            loss_val = loss_function(output_val, labels)
            valid_loss += loss_val.item()

            pred = np.argmax(output_val.cpu().detach().numpy(), axis=1)

            if len(pred.shape) == 0:
                pred = np.expand_dims(np.array(pred), axis=0)
            f1 = f1_score(labels.cpu().detach().numpy(), pred)
            f1_r += f1
        f1_r = f1_r / len(val_generator)
        valid_loss = valid_loss / len(val_generator)
        if wandb:
            wandb.log({"val_f1": f1_r})
            wandb.log({"val_loss_ft": valid_loss})

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {running_loss} \t\t Validation Loss: {valid_loss}"
        )

        scheduler.step()

        if early_stopping.early_stop(model, optimizer, -f1_r, valid_loss, epoch):
            break

    return path_save


def Training_model(project_name, dict_generators, args, wandb=[], freeze=False, k=1):
    params_model = {"freeze": freeze, "channels": args.inChannels}

    scheme = args.scheme

    if scheme == 1:  # Two class from a pretrained-model
        model = Selec_model_two_classes(args.model_name, **params_model)
        path_save = Training_fine_tunning(
            model,
            dict_generators["training_generator_ft"],
            dict_generators["val_generator_ft"],
            project_name=project_name + "_Fold_" + str(k),
            save_path=args.save,
            class_weights=args.class_weights,
            training_epochs=args.nTraining_epochs_ft,
            cooldown_epochs=args.nCooldown_epochs_ft,
            lr_ft=args.lr * 10,
            wandb=wandb,
        )

        # load checkpoint model
        model = Selec_model_two_classes(args.model_name, **params_model)
        checkpoint = torch.load(path_save)
        model.load_state_dict(checkpoint["model_state_dict"])

    elif scheme == 2:  # SimCLR from pretrained model and finetuning
        # defining the model
        model_cl = Selec_embedding(args.model_name, **params_model)
        path_save = Training_CL(
            model_cl,
            dict_generators["training_generator_cl"],
            dict_generators["val_generator_cl"],
            training_epochs=args.nTraining_epochs_cl,
            cooldown_epochs=args.nCooldown_epochs_cl,
            batch_size=args.batch_size,
            lr_cl=args.lr,
            project_name=project_name + "_Fold_" + str(k),
            save_path=args.save,
            wandb=wandb,
        )

        # load checkpoint model
        model_cl = Selec_embedding(args.model_name, **params_model)
        checkpoint = torch.load(path_save)
        model_cl.load_state_dict(checkpoint["model_state_dict"])

        model = finetuning_model(model_cl.embedding)
        path_save = Training_fine_tunning(
            model,
            dict_generators["training_generator_ft"],
            dict_generators["val_generator_ft"],
            project_name=project_name + "_Fold_" + str(k),
            save_path=args.save,
            class_weights=args.class_weights,
            training_epochs=args.nTraining_epochs_ft,
            cooldown_epochs=args.nCooldown_epochs_ft,
            lr_ft=args.lr * 10,
            wandb=wandb,
        )

        # load checkpoint model
        model = finetuning_model(model_cl.embedding)
        checkpoint = torch.load(path_save)
        model.load_state_dict(checkpoint["model_state_dict"])

    elif (
        scheme == 3
    ):  # Two class from pretrained model and training based on ge2e and crossentropy
        model_cl = Selec_embedding(args.model_name, **params_model)
        model = finetuning_model(model_cl.embedding, freeze=False)
        path_save = Training_fine_tunning_ge2e(
            model,
            dict_generators["training_generator_ft"],
            dict_generators["val_generator_ft"],
            project_name=project_name + "_Fold_" + str(k),
            save_path=args.save,
            class_weights=args.class_weights,
            training_epochs=args.nTraining_epochs_ft,
            cooldown_epochs=args.nCooldown_epochs_ft,
            lr_ft=args.lr,
            wandb=wandb,
        )

        # load checkpoint model
        model = finetuning_model(model_cl.embedding)
        checkpoint = torch.load(path_save)
        model.load_state_dict(checkpoint["model_state_dict"])

    elif (
        scheme == 4
    ):  # SimCLR from pretrained model and finetuning with ge2e and crossentropy
        model_cl = Selec_embedding(args.model_name, **params_model)
        path_save = Training_CL(
            model_cl,
            dict_generators["training_generator_cl"],
            dict_generators["val_generator_cl"],
            training_epochs=args.nTraining_epochs_cl,
            cooldown_epochs=args.nCooldown_epochs_cl,
            batch_size=args.batch_size,
            lr_cl=args.lr,
            project_name=project_name + "_Fold_" + str(k),
            save_path=args.save,
            wandb=wandb,
        )

        # load checkpoint model
        model_cl = Selec_embedding(args.model_name, **params_model)
        checkpoint = torch.load(path_save)
        model_cl.load_state_dict(checkpoint["model_state_dict"])

        model = finetuning_model(model_cl.embedding, freeze=False)
        path_save = Training_fine_tunning_ge2e(
            model,
            dict_generators["training_generator_ft"],
            dict_generators["val_generator_ft"],
            project_name=project_name + "_Fold_" + str(k),
            save_path=args.save,
            class_weights=args.class_weights,
            training_epochs=args.nTraining_epochs_ft,
            cooldown_epochs=args.nCooldown_epochs_ft,
            lr_ft=args.lr,
            wandb=wandb,
        )

        # load checkpoint model
        model = finetuning_model(model_cl.embedding)
        checkpoint = torch.load(path_save)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


def get_beta(epoch):
    cycle_length = 4  # Number of epochs in a single cycler
    current_cycle_epoch = (epoch - 1) % cycle_length  # Current epoch within the cycle
    beta = current_cycle_epoch / (cycle_length - 1)

    # Reescale beta value between 0 and 0.5
    beta = beta / 2

    return beta


def VQVAE_trainer(model, trainloader, validloader, epochs, lr, supervised, wandb_flag):
    loss = torch.nn.MSELoss(reduction="sum")
    opt = torch.optim.Adam(model.parameters(), lr)
    if supervised:
        loss_class = torch.nn.BCELoss(reduction="sum")

    eta = 1

    loss_train = []
    loss_rec_train = []
    loss_valid = []
    loss_rec_valid = []
    loss_vq_train = []
    loss_vq_valid = []
    loss_class_train = []
    loss_class_valid = []

    for e in range(epochs):
        model.train()
        train_rec_loss = 0
        train_vq_loss = 0
        train_loss = 0
        train_class_loss = 0
        codes_usage = np.zeros(model.K)

        with tqdm(total=len(trainloader)) as pbar_train:
            for x, y, z in trainloader:
                opt.zero_grad()
                # Data
                x = x.to(torch.float32).to(model.device)
                # Label
                y = y.to(torch.float32).to(model.device)

                # Forward pass
                x_hat, y_hat, vq_loss, z, z_q, enc_idx = model(x)

                # Reconstruction loss
                rec_loss = loss(x_hat, x)

                # If supervised, add classification loss
                if supervised:
                    class_loss = loss_class(y_hat, y.reshape(-1, 1))
                    train_class_loss += class_loss.item()
                else:
                    class_loss = 0
                # Total loss
                loss_vq = rec_loss + vq_loss + eta * class_loss

                # Backward pass
                loss_vq.backward()
                opt.step()
                # Update losses
                train_rec_loss += rec_loss.item()
                train_vq_loss += vq_loss.item()
                train_loss += loss_vq.item()
                # Update progress bar
                pbar_train.update(1)

                # Update codes usage
                codes_usage += model.usage.detach().cpu().numpy()

            # Update lsits
            loss_rec_train.append(train_rec_loss / len(trainloader))
            loss_train.append(train_loss / len(trainloader))
            loss_vq_train.append(train_vq_loss / len(trainloader))
            loss_class_train.append(train_class_loss / len(trainloader))

            # Codes_usage is a list of K elements, each element is the number of times a code has been used
            # Plot a histogram of the codes usage
            fig1 = plt.figure()
            plt.bar(np.arange(model.K), codes_usage)
            plt.title("Codes usage")
            plt.xlabel("Code")
            plt.ylabel("Usage")
            # Normalise codes usage and plot the probability distribution
            codes_usage_norm = codes_usage / np.sum(codes_usage)
            fig2 = plt.figure()
            plt.bar(np.arange(model.K), codes_usage_norm)
            # Plot a horizontal line at 1/K and at a label that says "Uniform distribution"
            plt.axhline(1 / model.K, color="r", linestyle="dashed")
            plt.text(
                0,
                1 / model.K,
                "Uniform distribution",
                horizontalalignment="left",
                verticalalignment="bottom",
            )
            plt.legend(["Codes usage", "Uniform distribution"])
            plt.title("Codes usage normalized")
            plt.xlabel("Code")
            plt.ylabel("Usage")

            if wandb_flag:
                wandb.log(
                    {
                        "train/Epoch": e,
                        "train/Loss": loss_train[-1],
                        "train/Rec Loss": loss_rec_train[-1],
                        "train/Quant Loss": loss_vq_train[-1],
                        "train/Class Loss": loss_class_train[-1],
                    }
                )
                # Log the image
                wandb.log({"train/Codes usage": fig1})
                wandb.log({"train/Codes usage normalized": fig2})
            plt.close(fig1)
            plt.close(fig2)

            # Reset usage
            model.reset_usage()
            # Plot reconstruction
            check_reconstruction(x, x_hat, wandb_flag, train_flag=True)

            if supervised:
                pbar_train.set_description(
                    "Epoch: {}; Loss: {:.5f}; Rec Loss: {:.5f}; Quant Loss: {:.5f}; Class Loss: {:.5f}".format(
                        e,
                        loss_train[-1],
                        loss_rec_train[-1],
                        loss_vq_train[-1],
                        loss_class_train[-1],
                    )
                )

            else:
                pbar_train.set_description(
                    "Epoch: {}; Loss: {:.5f}; Rec Loss: {:.5f}; Quant Loss: {:.5f}".format(
                        e,
                        loss_train[-1],
                        loss_rec_train[-1],
                        loss_vq_train[-1],
                    )
                )

            # Validation
            model.eval()
            valid_rec_loss = 0
            valid_vq_loss = 0
            valid_loss = 0
            valid_class_loss = 0

            val_codes_usage = np.zeros(model.K)

            with tqdm(total=len(validloader)) as pbar_valid:
                for x, y, z in validloader:
                    # Data
                    x = x.to(torch.float32).to(model.device)
                    # Label
                    y = y.to(torch.float32).to(model.device)

                    # Forward pass
                    x_hat, y_hat, vq_loss, z, z_q, enc_idx = model(x)

                    # Reconstruction loss
                    rec_loss = loss(x_hat, x)
                    # Latent loss
                    quant_loss = vq_loss
                    # If supervised, add classification loss
                    if supervised:
                        class_loss = loss_class(y_hat, y.reshape(-1, 1))
                        valid_class_loss += class_loss.item()
                    else:
                        class_loss = 0
                    # Total loss
                    loss_vq = rec_loss + quant_loss + eta * class_loss

                    # Update losses
                    valid_rec_loss += rec_loss.item()
                    valid_vq_loss += quant_loss.item()
                    valid_loss += loss_vq.item()
                    # Update progress bar
                    pbar_valid.update(1)

                    # Update codes usage
                    val_codes_usage += model.usage.detach().cpu().numpy()

                # Update lsits
                loss_valid.append(valid_loss / len(validloader))
                loss_vq_valid.append(valid_vq_loss / len(validloader))
                loss_class_valid.append(valid_class_loss / len(validloader))
                loss_rec_valid.append(valid_rec_loss / len(validloader))
                val_codes_usage_norm = val_codes_usage / np.sum(val_codes_usage)

                fig1 = plt.figure()
                plt.bar(np.arange(model.K), codes_usage)
                plt.title("Codes usage")
                plt.xlabel("Code")
                plt.ylabel("Usage")
                # Normalise codes usage and plot the probability distribution
                codes_usage_norm = codes_usage / np.sum(codes_usage)
                fig2 = plt.figure()
                plt.bar(np.arange(model.K), codes_usage_norm)
                # Plot a horizontal line at 1/K and at a label that says "Uniform distribution"
                plt.axhline(1 / model.K, color="r", linestyle="dashed")
                plt.text(
                    0,
                    1 / model.K,
                    "Uniform distribution",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                )
                plt.legend(["Codes usage", "Uniform distribution"])
                plt.title("Codes usage normalized")
                plt.xlabel("Code")
                plt.ylabel("Usage")
                if wandb_flag:
                    # Log the image
                    wandb.log({"valid/Codes usage": fig1})
                    wandb.log({"valid/Codes usage normalized": fig2})
                plt.close(fig1)
                plt.close(fig2)

                if supervised:
                    pbar_valid.set_description(
                        "Epoch: {}; Loss: {:.5f}; Rec Loss: {:.5f}; Quant Loss: {:.5f}; Class Loss: {:.5f}".format(
                            e,
                            loss_valid[-1],
                            loss_rec_valid[-1],
                            loss_vq_valid[-1],
                            loss_class_valid[-1],
                        )
                    )
                else:
                    pbar_valid.set_description(
                        "Epoch: {}; Loss: {:.5f}; Rec Loss: {:.5f}; Quant Loss: {:.5f}".format(
                            e,
                            loss_valid[-1],
                            loss_rec_valid[-1],
                            loss_vq_valid[-1],
                        )
                    )

                # Reset usage
                model.reset_usage()
                # Plot reconstruction
                check_reconstruction(x, x_hat, wandb_flag, train_flag=False)

                if wandb_flag:
                    wandb.log(
                        {
                            "valid/Epoch": e,
                            "valid/Loss": loss_valid[-1],
                            "valid/Rec Loss": loss_rec_valid[-1],
                            "valid/Quant Loss": loss_vq_valid[-1],
                            "valid/Class Loss": loss_class_valid[-1],
                        }
                    )

            # If the validation loss is the best, save the model
            if loss_valid[-1] == min(loss_valid):
                print("Saving the best model at epoch {}".format(e))
                name = "local_results/plps/"
                if supervised:
                    name += "vqvae/VAE_best_model_supervised"
                else:
                    name += "vqvae/VAE_best_model_unsupervised"
                name += ".pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                    },
                    name,
                )

            # Early stopping: If in the last 20 epochs the validation loss has not improved, stop the training
            if e > 50:
                if loss_valid[-1] > max(loss_valid[-20:-1]):
                    print("Early stopping")
                    break

    return (
        loss_train,
        loss_valid,
        loss_vq_train,
        loss_vq_valid,
        loss_class_train,
        loss_class_valid,
    )


def VAE_trainer(model, trainloader, validloader, epochs, lr, supervised, wandb_flag):
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr)
    loss = torch.nn.MSELoss(reduction="sum")
    if supervised:
        if model.n_classes == 2:
            loss_class = torch.nn.BCELoss(reduction="sum")
        else:
            loss_class = torch.nn.CrossEntropyLoss(reduction="sum")

    elbo_training = []
    kl_div_training = []
    rec_loss_training = []
    if supervised:  # if supervised we also have BCE loss
        bce_loss_training = []
    elbo_validation = []
    kl_div_validation = []
    rec_loss_validation = []
    if supervised:  # if supervised we also have BCE loss
        bce_loss_validation = []

    model.train()
    print("Training the VAE model")
    for e in range(epochs):
        train_loss = 0
        kl_div = 0
        rec_loss = 0
        if supervised:
            bce_loss = 0
        model.train()

        # beta_sc = get_beta(e)
        # print("Beta: ", beta_sc)
        beta_sc = 0.1
        beta_bce = 50
        beta_rec = 0.5

        with tqdm(trainloader, unit="batch") as tepoch:
            for x, y, z in tepoch:
                tepoch.set_description(f"Epoch {e}")
                # Move data to device
                x = x.to(model.device).to(torch.float32)
                if model.n_classes == 2:
                    y = y.to(model.device).to(torch.float32)
                else:
                    z = z.type(torch.LongTensor).to(model.device)

                # Gradient to zero
                opt.zero_grad()
                # Forward pass
                if supervised:
                    x_hat, y_hat, mu, logvar = model(x)
                else:
                    x_hat, mu, logvar = model(x)
                # Compute variational lower bound
                reconstruction_loss = loss(x_hat, x)
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                if supervised:
                    if model.n_classes == 2:
                        bce = loss_class(y_hat, y.view(-1, 1))
                    else:
                        bce = loss_class(y_hat, z)
                    variational_lower_bound = (
                        beta_rec * reconstruction_loss
                        + beta_sc * kl_divergence
                        + beta_bce * bce
                    )
                else:
                    variational_lower_bound = (
                        beta_rec * reconstruction_loss + beta_sc * kl_divergence
                    )
                # Backward pass
                variational_lower_bound.backward()
                # Update parameters
                opt.step()

                # Update losses storing
                train_loss += variational_lower_bound.item()
                kl_div += kl_divergence.item()
                rec_loss += reconstruction_loss.item()
                if supervised:
                    bce_loss += bce.item()

            # Store losses
            elbo_training.append(train_loss / len(trainloader))
            kl_div_training.append(kl_div / len(trainloader))
            rec_loss_training.append(rec_loss / len(trainloader))
            if supervised:
                bce_loss_training.append(bce_loss / len(trainloader))

            # Print Losses at current epoch
            print(
                f"Epoch {e}: Train ELBO: {elbo_training[-1]:.2f}, Train KL divergence: {kl_div_training[-1]:.2f}, Train reconstruction loss: {rec_loss_training[-1]:.2f}"
            )
            if supervised:
                print(f"Train CrossEntropy loss: {bce_loss_training[-1]:.2f}")

            # Plot reconstruction
            check_reconstruction(x, x_hat, wandb_flag, train_flag=True)

            # Log to wandb
            if wandb_flag:
                wandb.log(
                    {
                        "train/ELBO": -elbo_training[-1],
                        "train/KL_div": kl_div_training[-1],
                        "train/rec_loss": rec_loss_training[-1],
                    }
                )
                if supervised:  # if supervised we also have BCE loss
                    wandb.log(
                        {
                            "train/BCE_loss": bce_loss_training[-1],
                        }
                    )

            # Validate model

            model.eval()
            print("Validating the VAE model")
            with torch.no_grad():
                valid_loss = 0
                kl_div = 0
                rec_loss = 0
                if supervised:
                    bce_loss = 0
                with tqdm(validloader, unit="batch") as tepoch:
                    for x, y, z in tepoch:
                        # Move data to device
                        x = x.to(model.device).to(torch.float32)
                        if model.n_classes == 2:
                            y = y.to(model.device).to(torch.float32)
                        else:
                            z = z.type(torch.LongTensor).to(model.device)

                        # Forward pass
                        if supervised:
                            x_hat, y_hat, mu, logvar = model(x)
                        else:
                            x_hat, mu, logvar = model(x)
                        # Compute variational lower bound
                        reconstruction_loss = loss(x_hat, x)
                        kl_divergence = -0.5 * torch.sum(
                            1 + logvar - mu.pow(2) - logvar.exp()
                        )
                        if supervised:
                            if model.n_classes == 2:
                                bce = loss_class(y_hat, y.view(-1, 1))
                            else:
                                bce = loss_class(y_hat, z)
                            variational_lower_bound = (
                                beta_rec * reconstruction_loss
                                + beta_sc * kl_divergence
                                + beta_bce * bce
                            )
                        else:
                            variational_lower_bound = (
                                beta_rec * reconstruction_loss + beta_sc * kl_divergence
                            )

                        # Update losses storing
                        valid_loss += variational_lower_bound.item()
                        kl_div += kl_divergence.item()
                        rec_loss += reconstruction_loss.item()
                        if supervised:
                            bce_loss += bce.item()

                    # Store losses
                    elbo_validation.append(valid_loss / len(validloader))
                    kl_div_validation.append(kl_div / len(validloader))
                    rec_loss_validation.append(rec_loss / len(validloader))
                    if supervised:
                        bce_loss_validation.append(bce_loss / len(validloader))

                    # Print Losses at current epoch
                    print(
                        f"Epoch {e}: Valid ELBO: {elbo_validation[-1]:.2f}, Valid KL divergence: {kl_div_validation[-1]:.2f}, Valid reconstruction loss: {rec_loss_validation[-1]:.2f}"
                    )
                    if supervised:
                        print(f"Valid CrossEntropy loss: {bce_loss_validation[-1]:.2f}")

                    # Plot reconstruction
                    check_reconstruction(x, x_hat, wandb_flag, train_flag=False)

                    # Log to wandb
                    if wandb_flag:
                        wandb.log(
                            {
                                "valid/ELBO": -elbo_validation[-1],
                                "valid/KL_div": kl_div_validation[-1],
                                "valid/rec_loss": rec_loss_validation[-1],
                            }
                        )
                        if supervised:  # if supervised we also have BCE loss
                            wandb.log(
                                {
                                    "valid/BCE_loss": bce_loss_validation[-1],
                                }
                            )

                    # If the validation loss is the best, save the model
                    if elbo_validation[-1] <= min(elbo_validation):
                        print("Storing the best model at epoch ", e)
                        name = "local_results/plps/"
                        if supervised:
                            name += "vae_supervised/"
                        else:
                            name += "vae_unsupervised/"
                        # check if the folder exists if not create it
                        if not os.path.exists(name):
                            os.makedirs(name)
                        name += "VAE_best_model.pt"
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": opt.state_dict(),
                            },
                            name,
                        )

                    if wandb_flag:
                        wandb.log(
                            {
                                "Epoch": e,
                            }
                        )

                    # Early stopping: If in the last 20 epochs the validation loss has not improved, stop the training
                    if e > 50:
                        if elbo_validation[-1] > max(elbo_validation[-20:-1]):
                            print("Early stopping")
                            break

    return (
        elbo_training,
        kl_div_training,
        rec_loss_training,
        elbo_validation,
        kl_div_validation,
        rec_loss_validation,
    )


def GMVAE_trainer(model, trainloader, validloader, epochs, lr, supervised, wandb_flag):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    valid_loss_store = []

    true_label_list = []
    pred_label_list = []

    for e in range(epochs):
        model.train()
        train_loss = 0
        rec_loss = 0
        gaussian_loss = 0
        cat_loss = 0
        clf_loss = 0
        metric_loss = 0
        usage = np.zeros(model.k)

        # Use tqdm for progress bar
        for batch_idx, (data, labels) in enumerate(tqdm(trainloader)):
            # Make sure dtype is Tensor float
            data = data.to(model.device).float()

            optimizer.zero_grad()

            (
                loss,
                rec_loss_b,
                gaussian_loss_b,
                cat_loss_b,
                clf_loss_b,
                metric_loss_b,
                x,
                x_hat,
                y_pred,
            ) = model.loss(data, labels, e)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            rec_loss += rec_loss_b.item()
            gaussian_loss += gaussian_loss_b.item()
            if supervised:
                clf_loss += clf_loss_b.item()
            cat_loss += cat_loss_b.item()
            metric_loss += metric_loss_b.item()
            usage += torch.sum(y_pred, dim=0).cpu().detach().numpy()

            true_label_list.append(labels.cpu().numpy())
            pred_label_list.append(torch.argmax(y_pred.cpu().detach(), dim=1))

        # Check reconstruction of X
        check_reconstruction(x, x_hat, wandb_flag, train_flag=True)

        # Check unsupervised cluster accuracy and NMI
        true_label = torch.tensor(np.concatenate(true_label_list))
        pred_label = torch.tensor(np.concatenate(pred_label_list))
        # Repeat true labels. Labels are Batch_size,1. Repeat each one N times:
        N = model.x_hat_shape_before_flat[-1]
        true_label = true_label.repeat_interleave(N, dim=0)
        acc = cluster_acc(pred_label, true_label)
        nmi_score = nmi(pred_label, true_label)

        print(
            "Epoch: {} Train Loss: {:.4f} Rec Loss: {:.4f} Gaussian Loss: {:.4f} Cat Loss: {:.4f} Clf Loss: {:.4f} Metric Loss: {:.4f} UAcc: {:.4f} NMI: {:.4f}".format(
                e,
                train_loss / len(trainloader.dataset),
                rec_loss / len(trainloader.dataset),
                gaussian_loss / len(trainloader.dataset),
                cat_loss / len(trainloader.dataset),
                clf_loss / len(trainloader.dataset),
                metric_loss / len(trainloader.dataset),
                acc,
                nmi_score,
            )
        )
        if wandb_flag:
            wandb.log(
                {
                    "train/Epoch": e,
                    "train/Loss": train_loss / len(trainloader.dataset),
                    "train/Rec Loss": rec_loss / len(trainloader.dataset),
                    "train/Gaussian Loss": gaussian_loss / len(trainloader.dataset),
                    "train/Cat Loss": cat_loss / len(trainloader.dataset),
                    "train/Clf Loss": clf_loss / len(trainloader.dataset),
                    "train/Categorical usage": usage / len(trainloader.dataset),
                    "train/Metric Loss": metric_loss / len(trainloader.dataset),
                    "train/UAcc": acc,
                    "train/NMI": nmi_score,
                }
            )

        if validloader is not None:
            model.eval()
            valid_loss = 0
            val_rec_loss = 0
            val_gaussian_loss = 0
            val_clf_loss = 0
            val_cat_loss = 0
            val_metric_loss = 0
            val_usage = 0

            true_label_list = []
            pred_label_list = []

            for batch_idx, (data, labels) in enumerate(tqdm(validloader)):
                # Make sure dtype is Tensor float
                data = data.to(model.device).float()

                (
                    loss,
                    rec_loss_v,
                    gaussian_loss_v,
                    cat_loss_v,
                    clf_loss_v,
                    metric_loss_v,
                    x,
                    x_hat,
                    y_pred,
                ) = model.loss(data, labels)
                valid_loss += loss.item()
                val_rec_loss += rec_loss_v.item()
                val_gaussian_loss += gaussian_loss_v.item()
                if supervised:
                    val_clf_loss += clf_loss_v.item()
                val_cat_loss += cat_loss_v.item()
                val_metric_loss += metric_loss_v.item()
                val_usage += torch.sum(y_pred, dim=0).cpu().detach().numpy()

                true_label_list.append(labels.cpu().numpy())
                pred_label_list.append(torch.argmax(y_pred.cpu().detach(), dim=1))

            # Check reconstruction of X
            check_reconstruction(x, x_hat, wandb_flag, train_flag=True)

            # Check unsupervised cluster accuracy and NMI
            true_label = torch.tensor(np.concatenate(true_label_list))
            pred_label = torch.tensor(np.concatenate(pred_label_list))
            # Repeat true labels. Labels are Batch_size,1. Repeat each one N times:
            N = model.x_hat_shape_before_flat[-1]
            true_label = true_label.repeat_interleave(N, dim=0)
            acc = cluster_acc(pred_label, true_label)
            nmi_score = nmi(pred_label, true_label)

            print(
                "Epoch: {} Valid Loss: {:.4f} Rec Loss: {:.4f} Gaussian Loss: {:.4f} Cat Loss : {:.4f} Clf Loss: {:.4f} Metric Loss: {:.4f} UAcc: {:.4f} NMI: {:.4f}".format(
                    e,
                    valid_loss / len(validloader.dataset),
                    val_rec_loss / len(validloader.dataset),
                    val_gaussian_loss / len(validloader.dataset),
                    val_cat_loss / len(validloader.dataset),
                    val_clf_loss / len(validloader.dataset),
                    val_metric_loss / len(validloader.dataset),
                    acc,
                    nmi_score,
                )
            )
            valid_loss_store.append(valid_loss / len(validloader.dataset))
            if wandb_flag:
                wandb.log(
                    {
                        "valid/Epoch": e,
                        "valid/Loss": valid_loss / len(validloader.dataset),
                        "valid/Rec Loss": val_rec_loss / len(validloader.dataset),
                        "valid/Gaussian Loss": val_gaussian_loss
                        / len(validloader.dataset),
                        "valid/Cat Loss": val_cat_loss / len(validloader.dataset),
                        "valid/Clf Loss": val_clf_loss / len(validloader.dataset),
                        "valid/Metric Loss": val_metric_loss / len(validloader.dataset),
                        "valid/UAcc": acc,
                        "valid/NMI": nmi_score,
                        "valid/Categorical usage": usage / len(trainloader.dataset),
                    }
                )
        # Store best model
        # If the validation loss is the best, save the model
        if valid_loss_store[-1] <= min(valid_loss_store):
            print("Storing the best model at epoch ", e)
            name = "local_results/spectrograms/manner_gmvae/"
            # check if the folder exists if not create it
            if not os.path.exists(name):
                os.makedirs(name)
            name += "GMVAE_cnn_best_model_unsupervised.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                name,
            )

        check_reconstruction(x, x_hat, wandb_flag, train_flag=False)

        # Early stopping: If in the last 20 epochs the validation loss has not improved, stop the training
        if e > 50:
            if valid_loss_store[-1] > max(valid_loss_store[-20:-1]):
                print("Early stopping")
                break


def check_reconstruction(x, x_hat, wandb_flag=False, train_flag=True):
    # Take the first 10 samples, but check if there are at least 10 samples
    if x.shape[0] < 10:
        idx = np.arange(x.shape[0])
    else:
        idx = np.arange(10)
    x_sample = x[idx]
    x_hat_sample = x_hat[idx]
    if len(x_sample.shape) == 4:
        x_sample = x_sample.squeeze(1)[0]
        x_hat_sample = x_hat_sample.squeeze(1)[0]
    # calculate error
    error = torch.abs(x_sample - x_hat_sample)

    # Plot them as heatmaps
    fig, axs = plt.subplots(3, 1, figsize=(20, 20))
    cmap = cm.get_cmap("viridis")
    normalizer = Normalize(
        torch.min(torch.cat((x_sample, x_hat_sample))),
        torch.max(torch.cat((x_sample, x_hat_sample))),
    )
    im = cm.ScalarMappable(norm=normalizer)
    axs[0].set_title("Original")
    axs[0].imshow(x_sample.cpu().detach().numpy(), cmap=cmap, norm=normalizer)
    axs[1].set_title("Reconstruction")
    axs[1].imshow(x_hat_sample.cpu().detach().numpy(), cmap=cmap, norm=normalizer)
    axs[2].set_title("Error")
    axs[2].imshow(error.cpu().detach().numpy(), cmap=cmap, norm=normalizer)
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()
    if wandb_flag:
        if train_flag:
            name = "train/rec_img"
        else:
            name = "valid/rec_img"
        wandb.log({name: fig})
    plt.close(fig)


def VAE_tester(
    model,
    testloader,
    test_data,
    audio_features="plps",
    supervised=False,
    wandb_flag=False,
):
    # Set model in evaluation mode
    model.eval()
    print("Evaluating the VAE model")

    with torch.no_grad():
        # Get batch size from testloader
        batch_size = testloader.batch_size
        y_hat_array = np.zeros((batch_size, 1))
        z_array = np.zeros((batch_size, 1))
        y_array = np.zeros((batch_size, 1))
        # Create x_array of shape Batch x Output shape
        x_array = np.zeros(
            (batch_size, test_data[audio_features].iloc[0].shape[0])
        )  # 32 is the batch size
        x_hat_array = np.zeros(x_array.shape)
        with tqdm(testloader, unit="batch") as tepoch:
            for x, y in tepoch:
                # Move data to device
                x = x.to(model.device).to(torch.float32)
                if model.n_classes == 2:
                    y = y.to(model.device).to(torch.float32)
                else:
                    z = z.type(torch.LongTensor).to(model.device)

                # Forward pass
                if supervised:
                    x_hat, y_hat, mu, logvar = model(x)
                    # Concatenate true values
                    y_array = np.concatenate(
                        (y_array, y.cpu().detach().numpy().reshape(-1, 1))
                    )
                    # Concatenate true values (vowels)
                    z_array = np.concatenate(
                        (z_array, z.cpu().detach().numpy().reshape(-1, 1))
                    )

                    # Concatenate predictions (labels)
                    if model.n_classes == 2:
                        y_hat_array = np.concatenate(
                            (y_hat_array, y_hat.cpu().detach().numpy())
                        )
                    # Concatenate predictions (vowels)
                    else:
                        y_hat_array = np.concatenate(
                            (
                                y_hat_array,
                                np.argmax(y_hat.cpu().detach().numpy(), axis=1).reshape(
                                    -1, 1
                                ),
                            )
                        )
                else:
                    x_hat, mu, logvar = model(x)

                # Concatenate predictions
                x_hat_array = np.concatenate(
                    (x_hat_array, x_hat.cpu().detach().numpy()), axis=0
                )
                x_array = np.concatenate((x_array, x.cpu().detach().numpy()), axis=0)

        # Remove the first batch_size elements
        x_array = x_array[batch_size:]
        x_hat_array = x_hat_array[batch_size:]
        if supervised:
            y_array = y_array[batch_size:]
            y_hat_array = y_hat_array[batch_size:]

        # Calculate mse between x and x_hat
        mse = ((x_array - x_hat_array) ** 2).mean(axis=None)
        # Results for all frames
        print(f"Reconstruction loss: {mse}")

        # Results per patient
        rec_loss_per_patient = []
        for i in test_data["id_patient"].unique():
            idx = test_data["id_patient"] == i
            rec_loss_per_patient.append(((x_array[idx] - x_hat_array[idx]) ** 2).mean())

        print("Results per patient in mean and std:")
        print(
            f"Reconstruction loss: {np.mean(rec_loss_per_patient)} +- {np.std(rec_loss_per_patient)}"
        )
        # Calculate results in total
        if supervised:
            print("Results for all frames:")
            y_bin = np.round(y_hat_array)
            accuracy = accuracy_score(y_array, y_bin)
            balanced_accuracy = balanced_accuracy_score(y_array, y_bin)
            auc = roc_auc_score(y_array, y_hat_array)
            print("Results for all frames:")
            print(
                f"Accuracy: {accuracy:.2f}, Balanced accuracy: {balanced_accuracy:.2f}, AUC: {auc:.2f}"
            )

            # Calculate results per patient
            accuracy_per_patient = []
            balanced_accuracy_per_patient = []
            for i in test_data["id_patient"].unique():
                # Get the predictions for the patient
                y_patient = y_array[test_data.id_patient == i]
                y_hat_patient = y_hat_array[test_data.id_patient == i]

                # Calculate the metrics
                accuracy_patient = accuracy_score(y_patient, np.round(y_hat_patient))
                balanced_accuracy_patient = balanced_accuracy_score(
                    y_patient, np.round(y_hat_patient)
                )

                # Store the results
                accuracy_per_patient.append(accuracy_patient)
                balanced_accuracy_per_patient.append(balanced_accuracy_patient)

            # Print the results
            print("Results per patient in mean and std:")
            print(
                f"Accuracy: {np.mean(accuracy_per_patient):.2f} +- {np.std(accuracy_per_patient):.2f}, Balanced accuracy: {np.mean(balanced_accuracy_per_patient):.2f} +- {np.std(balanced_accuracy_per_patient):.2f}"
            )
            if wandb_flag:
                wandb.log(
                    {
                        "test/accuracy": accuracy,
                        "test/balanced_accuracy": balanced_accuracy,
                        "test/auc": auc,
                        "test/accuracy_per_patient": np.mean(accuracy_per_patient),
                        "test/balanced_accuracy_per_patient": np.mean(
                            balanced_accuracy_per_patient
                        ),
                    }
                )


def VQVAE_tester(
    model,
    testloader,
    test_data,
    audio_features="plps",
    supervised=False,
    wandb_flag=False,
):
    # Set model in evaluation mode
    model.eval()
    print("Evaluating the VQ-VAE model")
    with torch.no_grad():
        # Get batch size from testloader
        batch_size = testloader.batch_size
        y_hat_array = np.zeros((batch_size, 1))
        y_array = np.zeros((batch_size, 1))
        # Create x_array of shape Batch x Output shape
        x_array = np.zeros(
            (batch_size, test_data[audio_features].iloc[0].shape[0])
        )  # 32 is the batch size
        x_hat_array = np.zeros(x_array.shape)
        z_array = np.zeros((batch_size, model.latent_dim))
        z_q_array = np.zeros((batch_size, model.latent_dim))
        enc_idx_array = np.zeros((batch_size, 1))
        vowel_array = np.zeros((batch_size, 1))

        with tqdm(testloader, unit="batch") as tepoch:
            for x, y, v in tepoch:
                # Move data to device
                x = x.to(model.device).to(torch.float32)
                y = y.to(model.device).to(torch.float32)
                v = v.to(model.device).to(torch.float32)

                # Forward pass
                if supervised:
                    x_hat, y_hat, vq_loss, z, z_q, enc_idx = model(x)
                    # Concatenate predictions
                    y_array = np.concatenate(
                        (y_array, y.cpu().detach().numpy().reshape(-1, 1))
                    )

                    y_hat_array = np.concatenate(
                        (y_hat_array, y_hat.cpu().detach().numpy())
                    )
                else:
                    x_hat, y_hat, vq_loss, z, z_q, enc_idx = model(x)

                # Concatenate predictions
                x_hat_array = np.concatenate(
                    (x_hat_array, x_hat.cpu().detach().numpy()), axis=0
                )
                x_array = np.concatenate((x_array, x.cpu().detach().numpy()), axis=0)
                # Concatenate latent variables
                z_array = np.concatenate((z_array, z.cpu().detach().numpy()), axis=0)
                z_q_array = np.concatenate(
                    (z_q_array, z_q.cpu().detach().numpy()), axis=0
                )

                enc_idx_array = np.concatenate(
                    (enc_idx_array, enc_idx.cpu().detach().numpy().reshape(-1, 1)),
                    axis=0,
                )
                vowel_array = np.concatenate(
                    (vowel_array, v.cpu().detach().numpy().reshape(-1, 1)), axis=0
                )

        # Remove the first batch_size elements
        x_array = x_array[batch_size:]
        x_hat_array = x_hat_array[batch_size:]
        if supervised:
            y_array = y_array[batch_size:]
            y_hat_array = y_hat_array[batch_size:]

        # Calculate mse between x and x_hat
        mse = ((x_array - x_hat_array) ** 2).mean(axis=None)
        # Results for all frames
        print(f"Reconstruction loss: {mse}")

        # Results per patient
        rec_loss_per_patient = []
        for i in test_data["id_patient"].unique():
            idx = test_data["id_patient"] == i
            rec_loss_per_patient.append(((x_array[idx] - x_hat_array[idx]) ** 2).mean())

        print("Results per patient in mean and std:")
        print(
            f"Reconstruction loss: {np.mean(rec_loss_per_patient)} +- {np.std(rec_loss_per_patient)}"
        )
        # Calculate results in total
        if supervised:
            y_bin = np.round(y_hat_array)
            accuracy = accuracy_score(y_array, y_bin)
            balanced_accuracy = balanced_accuracy_score(y_array, y_bin)
            auc = roc_auc_score(y_array, y_hat_array)
            print("Results for all frames:")
            print(
                f"Accuracy: {accuracy:.2f}, Balanced accuracy: {balanced_accuracy:.2f}, AUC: {auc:.2f}"
            )

            # Calculate results per patient
            accuracy_per_patient = []
            balanced_accuracy_per_patient = []
            for i in test_data["id_patient"].unique():
                # Get the predictions for the patient
                y_patient = y_array[test_data.id_patient == i]
                y_hat_patient = y_hat_array[test_data.id_patient == i]

                # Calculate the metrics
                accuracy_patient = accuracy_score(y_patient, np.round(y_hat_patient))
                balanced_accuracy_patient = balanced_accuracy_score(
                    y_patient, np.round(y_hat_patient)
                )

                # Store the results
                accuracy_per_patient.append(accuracy_patient)
                balanced_accuracy_per_patient.append(balanced_accuracy_patient)

            # Print the results
            print("Results per patient in mean and std:")
            print(
                f"Accuracy: {np.mean(accuracy_per_patient):.2f} +- {np.std(accuracy_per_patient):.2f}, Balanced accuracy: {np.mean(balanced_accuracy_per_patient):.2f} +- {np.std(balanced_accuracy_per_patient):.2f}"
            )

            if wandb_flag:
                wandb.log(
                    {
                        "test/mse": mse,
                        "test/accuracy_per_patient": np.mean(accuracy_per_patient),
                        "test/balanced_accuracy_per_patient": np.mean(
                            balanced_accuracy_per_patient
                        ),
                        "test/auc": auc,
                        "test/accuracy": accuracy,
                        "test/balanced_accuracy": balanced_accuracy,
                    }
                )

    return x_array, x_hat_array, z_array, z_q_array, enc_idx_array, vowel_array


def GMVAE_tester(
    model,
    testloader,
    test_data,
    audio_features="plps",
    supervised=False,
    wandb_flag=False,
):
    # Warning for the precision of the matrix multiplication
    torch.set_float32_matmul_precision("high")
    # Set model in evaluation mode
    model.eval()
    print("Evaluating the VAE model")

    with torch.no_grad():
        # Get batch size from testloader
        batch_size = testloader.batch_size
        y_hat_array = np.zeros((batch_size, 1))
        z_array = np.zeros((batch_size, 1))
        y_array = np.zeros((batch_size, 1))

        # Create x_array of shape Batch x Output shape
        if audio_features == "spectrogram":
            x_array = np.zeros((batch_size, 1, 65, 41))
        else:
            x_array = np.zeros((batch_size, test_data[audio_features].iloc[0].shape[0]))

        x_hat_array = np.zeros(x_array.shape)
        with tqdm(testloader, unit="batch") as tepoch:
            for x, y in tepoch:
                # Move data to device
                x = x.to(model.device).to(torch.float32)
                if model.k == 2:
                    y = y.to(model.device).to(torch.float32)
                elif model.k == 5:
                    z = z.type(torch.LongTensor).to(model.device)
                elif model.k == 10:
                    c = c.type(torch.LongTensor).to(model.device)

                # Forward pass
                if supervised:
                    x_hat, y_hat, mu, logvar = model(x)
                    # Concatenate true values
                    y_array = np.concatenate(
                        (y_array, y.cpu().detach().numpy().reshape(-1, 1))
                    )
                    # Concatenate true values (vowels)
                    z_array = np.concatenate(
                        (z_array, z.cpu().detach().numpy().reshape(-1, 1))
                    )

                    # Concatenate predictions (labels)
                    if model.n_classes == 2:
                        y_hat_array = np.concatenate(
                            (y_hat_array, y_hat.cpu().detach().numpy())
                        )
                    # Concatenate predictions (vowels)
                    else:
                        y_hat_array = np.concatenate(
                            (
                                y_hat_array,
                                np.argmax(y_hat.cpu().detach().numpy(), axis=1).reshape(
                                    -1, 1
                                ),
                            )
                        )
                else:
                    x_hat, _, _, _, _, _, _, _, _ = model.forward(x)

                # Concatenate predictions
                x_hat_array = np.concatenate(
                    (x_hat_array, x_hat.cpu().detach().numpy()), axis=0
                )
                x_array = np.concatenate((x_array, x.cpu().detach().numpy()), axis=0)

        # Remove the first batch_size elements
        x_array = x_array[batch_size:]
        x_hat_array = x_hat_array[batch_size:]
        if supervised:
            y_array = y_array[batch_size:]
            y_hat_array = y_hat_array[batch_size:]

        # Calculate mse between x and x_hat
        mse = ((x_array - x_hat_array) ** 2).mean(axis=None)
        # Results for all frames
        print(f"Reconstruction loss: {mse}")

        # Results per patient
        rec_loss_per_patient = []
        for i in test_data["id_patient"].unique():
            idx = test_data["id_patient"] == i
            rec_loss_per_patient.append(((x_array[idx] - x_hat_array[idx]) ** 2).mean())

        print("Results per patient in mean and std:")
        print(
            f"Reconstruction loss: {np.mean(rec_loss_per_patient)} +- {np.std(rec_loss_per_patient)}"
        )
        # Calculate results in total
        if supervised:
            print("Results for all frames:")
            y_bin = np.round(y_hat_array)
            accuracy = accuracy_score(y_array, y_bin)
            balanced_accuracy = balanced_accuracy_score(y_array, y_bin)
            auc = roc_auc_score(y_array, y_hat_array)
            print("Results for all frames:")
            print(
                f"Accuracy: {accuracy:.2f}, Balanced accuracy: {balanced_accuracy:.2f}, AUC: {auc:.2f}"
            )

            # Calculate results per patient
            accuracy_per_patient = []
            balanced_accuracy_per_patient = []
            for i in test_data["id_patient"].unique():
                # Get the predictions for the patient
                y_patient = y_array[test_data.id_patient == i]
                y_hat_patient = y_hat_array[test_data.id_patient == i]

                # Calculate the metrics
                accuracy_patient = accuracy_score(y_patient, np.round(y_hat_patient))
                balanced_accuracy_patient = balanced_accuracy_score(
                    y_patient, np.round(y_hat_patient)
                )

                # Store the results
                accuracy_per_patient.append(accuracy_patient)
                balanced_accuracy_per_patient.append(balanced_accuracy_patient)

            # Print the results
            print("Results per patient in mean and std:")
            print(
                f"Accuracy: {np.mean(accuracy_per_patient):.2f} +- {np.std(accuracy_per_patient):.2f}, Balanced accuracy: {np.mean(balanced_accuracy_per_patient):.2f} +- {np.std(balanced_accuracy_per_patient):.2f}"
            )
            if wandb_flag:
                wandb.log(
                    {
                        "test/accuracy": accuracy,
                        "test/balanced_accuracy": balanced_accuracy,
                        "test/auc": auc,
                        "test/accuracy_per_patient": np.mean(accuracy_per_patient),
                        "test/balanced_accuracy_per_patient": np.mean(
                            balanced_accuracy_per_patient
                        ),
                    }
                )
