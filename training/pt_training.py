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


def VAE_trainer(model, trainloader, validloader, epochs, lr, supervised, wandb_flag):
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr)
    loss = torch.nn.MSELoss(reduction="sum")
    if supervised:
        loss_class = torch.nn.BCELoss(reduction="sum")

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

        beta_sc = get_beta(e)
        print("Beta: ", beta_sc)

        with tqdm(trainloader, unit="batch") as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {e}")
                # Move data to device
                x = x.to(model.device).to(torch.float32)
                y = y.to(model.device).to(torch.float32)

                # Gradient to zero
                opt.zero_grad()
                # Forward pass
                if supervised:
                    x_hat, y_hat, mu, logvar = model(x)
                else:
                    x_hat, mu, logvar = model(x)
                # Compute variational lower bound
                # reconstruction_loss = loss(x_hat, x, var=0.01 * torch.ones_like(x_hat))
                reconstruction_loss = loss(x_hat, x)
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                if supervised:
                    bce = loss_class(y_hat, y.view(-1, 1))
                    variational_lower_bound = (
                        reconstruction_loss + beta_sc * kl_divergence + bce
                    )
                else:
                    variational_lower_bound = (
                        reconstruction_loss + beta_sc * kl_divergence
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
                print(f"Train BCE loss: {bce_loss_training[-1]:.2f}")

            # Get 10 fisrt samples and check their value vs their reconstruction
            idx = np.arange(10)
            x_sample = x[idx]
            x_hat_sample = x_hat[idx]
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
            axs[1].imshow(
                x_hat_sample.cpu().detach().numpy(), cmap=cmap, norm=normalizer
            )
            axs[2].set_title("Error")
            axs[2].imshow(error.cpu().detach().numpy(), cmap=cmap, norm=normalizer)
            fig.colorbar(im, ax=axs.ravel().tolist())
            plt.show()
            # Log to wandb
            if wandb_flag:
                wandb.log(
                    {
                        "train/ELBO": elbo_training[-1],
                        "train/KL_div": kl_div_training[-1],
                        "train/rec_loss": rec_loss_training[-1],
                        "train/rec_img": fig,
                        "train/epoch": e,
                    }
                )
                if supervised:  # if supervised we also have BCE loss
                    wandb.log(
                        {
                            "train/BCE_loss": bce_loss_training[-1],
                        }
                    )
            # Close the img
            plt.close(fig)

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
                    for x, y in tepoch:
                        # Move data to device
                        x = x.to(model.device).to(torch.float32)
                        y = y.to(model.device).to(torch.float32)

                        # Forward pass
                        if supervised:
                            x_hat, y_hat, mu, logvar = model(x)
                        else:
                            x_hat, mu, logvar = model(x)
                        # Compute variational lower bound
                        # reconstruction_loss = loss(
                        #     x_hat, x, var=0.01 * torch.ones_like(x_hat)
                        # )
                        reconstruction_loss = loss(x_hat, x)
                        kl_divergence = -0.5 * torch.sum(
                            1 + logvar - mu.pow(2) - logvar.exp()
                        )
                        if supervised:
                            bce = loss_class(y_hat, y.view(-1, 1))
                            variational_lower_bound = (
                                reconstruction_loss + beta_sc * kl_divergence + bce
                            )
                        else:
                            variational_lower_bound = (
                                reconstruction_loss + beta_sc * kl_divergence
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
                        print(f"Valid BCE loss: {bce_loss_validation[-1]:.2f}")
                    # Get 10 fisrt samples and check their value vs their reconstruction
                    # Plot them as heatmaps
                    fig, axs = plt.subplots(3, 1, figsize=(20, 20))
                    cmap = cm.get_cmap("viridis")
                    normalizer = Normalize(
                        torch.min(torch.cat((x_sample, x_hat_sample))),
                        torch.max(torch.cat((x_sample, x_hat_sample))),
                    )
                    im = cm.ScalarMappable(norm=normalizer)
                    axs[0].set_title("Original")
                    axs[0].imshow(
                        x_sample.cpu().detach().numpy(), cmap=cmap, norm=normalizer
                    )
                    axs[1].set_title("Reconstruction")
                    axs[1].imshow(
                        x_hat_sample.cpu().detach().numpy(), cmap=cmap, norm=normalizer
                    )
                    axs[2].set_title("Error")
                    axs[2].imshow(
                        error.cpu().detach().numpy(), cmap=cmap, norm=normalizer
                    )

                    fig.colorbar(im, ax=axs.ravel().tolist())
                    plt.show()

                    # Log to wandb
                    if wandb_flag:
                        wandb.log(
                            {
                                "valid/ELBO": elbo_validation[-1],
                                "valid/KL_div": kl_div_validation[-1],
                                "valid/rec_loss": rec_loss_validation[-1],
                                "valid/rec_img": fig,
                                "valid/epoch": e,
                            }
                        )
                        if supervised:  # if supervised we also have BCE loss
                            wandb.log(
                                {
                                    "valid/BCE_loss": bce_loss_validation[-1],
                                }
                            )
                    plt.close(fig)

                    # If the validation loss is the best, save the model
                    if elbo_validation[-1] == min(elbo_validation):
                        name = "local_results/VAE_best_model"
                        if supervised:
                            name += "_supervised"
                        else:
                            name += "_unsupervised"
                        name += ".pt"
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": opt.state_dict(),
                            },
                            name,
                        )

    return (
        elbo_training,
        kl_div_training,
        rec_loss_training,
        elbo_validation,
        kl_div_validation,
        rec_loss_validation,
    )


def VAE_tester(model, testloader, supervised=False, wandb_flag=False):
    loss_nll = torch.nn.GaussianNLLLoss(reduction="sum")
    loss_mse = torch.nn.MSELoss(reduction="sum")
    loss_class = torch.nn.BCELoss(reduction="sum")

    model.eval()
    print("Evaluating the VAE model")
    with torch.no_grad():
        test_loss = 0
        test2_loss = 0
        if supervised:
            bce_loss = 0
        with tqdm(testloader, unit="batch") as tepoch:
            for x, y in tepoch:
                # Move data to device
                x = x.to(model.device).to(torch.float32)
                y = y.to(model.device).to(torch.float32)

                # Forward pass
                if supervised:
                    x_hat, y_hat, mu, logvar = model(x)
                else:
                    x_hat, mu, logvar = model(x)
                # Compute variational lower bound
                nll_loss = loss_nll(x_hat, x, var=0.01 * torch.ones_like(x_hat))
                mse_loss = loss_mse(x_hat, x)
                if supervised:
                    bce = loss_class(y_hat, y.view(-1, 1))

                # Update losses storing
                test_loss += mse_loss.item()
                test2_loss += nll_loss.item()
                if supervised:
                    bce_loss += bce.item()

        # Store losses
        reconstruction_error_nll = test2_loss / len(testloader.dataset)
        reconstruction_error_mse = test_loss / len(testloader.dataset)
        if supervised:
            bce_loss = bce_loss / len(testloader.dataset)

        # Print Losses at current epoch
        print(
            f"Test MSE: {reconstruction_error_mse:.2f}, Test NLL: {reconstruction_error_nll:.2f}"
        )
        if supervised:
            print(f"Test BCE: {bce_loss:.2f}")

        # Calculate accuracy and balanced accuracy and AUC if supervised
        if supervised:
            y_bin = torch.round(y_hat)
            accuracy = accuracy_score(
                y.cpu().detach().numpy(), y_bin.cpu().detach().numpy()
            )
            balanced_accuracy = balanced_accuracy_score(
                y.cpu().detach().numpy(), y_bin.cpu().detach().numpy()
            )
            auc = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
            print(
                f"Accuracy: {accuracy:.2f}, Balanced accuracy: {balanced_accuracy:.2f}, AUC: {auc:.2f}"
            )
            if wandb_flag:
                wandb.log(
                    {
                        "test/accuracy": accuracy,
                        "test/balanced_accuracy": balanced_accuracy,
                        "test/AUC": auc,
                        "test/BCE_loss": bce_loss,
                    }
                )

        if wandb_flag:
            wandb.log(
                {
                    "test/MSE": reconstruction_error_mse,
                    "test/NLL": reconstruction_error_nll,
                }
            )

    return reconstruction_error_mse, reconstruction_error_nll
