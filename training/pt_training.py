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
from sklearn.manifold import TSNE
import seaborn as sns


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


def MARTA_trainer(
    model,
    trainloader,
    validloader,
    epochs,
    lr,
    wandb_flag,
    path_to_save,
    supervised,
    classifier,
    domain_adversarial=False,
):
    # ============================= LR scheduler =============================
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T_max = 50  # Maximum number of iterations or epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=0.0001
    )

    # ============================= Storage variables =============================
    valid_loss_store = []

    for e in range(epochs):
        model.train()
        usage = np.zeros(model.k)

        true_manner_list = []
        gaussian_component = []
        y_pred_list = []
        label_list = []
        domain_list = []
        domain_pred_list = []

        (
            tr_running_loss,
            tr_rec_loss,
            tr_gauss_loss,
            tr_cat_loss,
            tr_metric_loss,
            tr_domain_loss,
        ) = (0, 0, 0, 0, 0, 0)

        for batch_idx, (data, labels, manner, dataset) in enumerate(tqdm(trainloader)):
            # Assert no nan in data, labels, manner or dataset
            assert not torch.isnan(data).any()
            assert not torch.isnan(labels).any()
            assert not torch.isnan(manner).any()

            # Make sure dtype is Tensor float
            data = data.to(model.device).float()
            # Dataset is "albayzin", "neurovoz" or "gita". Map them to 0, 1 and 2 respectively
            mapping = {"albayzin": 0, "neurovoz": 1, "gita": 2, "italian": 3}
            dataset = torch.tensor([mapping[dataset[i]] for i in range(len(dataset))])
            # Repeat each dataset window_size times
            dataset = (
                dataset.repeat_interleave(manner.shape[1]).to(model.device).float()
            )
            assert not torch.isnan(dataset).any()

            # ==== Forward pass ====

            if classifier:
                manner[manner > 7] = manner[manner > 7] - 8
                manner = manner.to(model.device).int()
                labels = labels.to(model.device).float()
                y_pred = model.classifier_forward(data, manner)
                y_pred = torch.sigmoid(y_pred)
                (
                    complete_loss,
                    recon_loss,
                    gaussian_loss,
                    categorical_loss,
                    metric_loss,
                ) = model.classifier_loss(y_pred, labels)
                y_pred_list = np.concatenate(
                    (y_pred_list, y_pred.cpu().detach().numpy().squeeze())
                )

                label_list = np.concatenate((label_list, labels.cpu().detach().numpy()))
            else:
                # Assert that any manner is no bigger than 7
                if not supervised:
                    assert torch.max(manner) <= 7
                (
                    x,
                    x_hat,
                    pz_mu,
                    pz_var,
                    y_logits,
                    y,
                    qz_mu,
                    qz_var,
                    z_sample,
                    e_s,
                    e_hat_s,
                    domain_pred,
                ) = model(data)

                # ==== Loss ====
                (
                    complete_loss,
                    recon_loss,
                    gaussian_loss,
                    categorical_loss,
                    metric_loss,
                    domain_loss,
                ) = model.loss(
                    data,
                    manner,
                    x_hat,
                    z_sample,
                    qz_mu,
                    qz_var,
                    pz_mu,
                    pz_var,
                    y,
                    y_logits,
                    domain_pred,
                    dataset,
                )
            # ==== Backward pass ====
            optimizer.zero_grad()
            complete_loss.backward()
            optimizer.step()

            # ==== Update metrics ====
            tr_running_loss += complete_loss.item()
            if not classifier:
                tr_rec_loss += recon_loss.item()
                tr_gauss_loss += gaussian_loss.item()
                tr_cat_loss += categorical_loss.item()
                tr_metric_loss += metric_loss.item()
                tr_domain_loss += domain_loss.item()
                usage += torch.sum(y, dim=0).cpu().detach().numpy()
                gaussian_component.append(y.cpu().detach().numpy())
                true_manner_list.append(manner)
                domain_list.append(dataset.cpu().detach().numpy())
                domain_pred = torch.argmax(domain_pred, dim=1)
                domain_pred_list.append(domain_pred.cpu().detach().numpy())

        # Scheduler step
        scheduler.step()

        # Check reconstruction of X
        if not classifier:
            check_reconstruction(x, x_hat, wandb_flag, train_flag=True)
            # Check unsupervised cluster accuracy and NMI
            true_manner = torch.tensor(np.concatenate(true_manner_list))
            gaussian_component = torch.tensor(np.concatenate(gaussian_component))
            acc = cluster_acc(gaussian_component, true_manner)
            nmi_score = nmi(gaussian_component, true_manner)
            domain_acc = accuracy_score(
                np.concatenate(domain_list), np.concatenate(domain_pred_list)
            )
            print(
                "Epoch: {} Train Loss: {:.4f} Rec Loss: {:.4f} Gaussian Loss: {:.4f} Cat Loss: {:.4f} Metric Loss: {:.4f} Domain Loss: {:.4f} UAcc: {:.4f} NMI: {:.4f} Domain acc {:.4f} ".format(
                    e,
                    tr_running_loss / len(trainloader.dataset),
                    tr_rec_loss / len(trainloader.dataset),
                    tr_gauss_loss / len(trainloader.dataset),
                    tr_cat_loss / len(trainloader.dataset),
                    tr_metric_loss / len(trainloader.dataset),
                    tr_domain_loss / len(trainloader.dataset),
                    acc,
                    nmi_score,
                    domain_acc,
                )
            )
        else:
            best_th_youden, best_th_eer, auc = threshold_selection(
                label_list, y_pred_list, verbose=0
            )
            y_pred = np.round(y_pred_list)
            acc_super = accuracy_score(label_list, y_pred)
            balanced_acc = balanced_accuracy_score(label_list, y_pred)
            print(
                "Epoch: {} Train Loss: {:.4f} Acc: {:.4f} Bacc: {:.4f} AUC: {:.4f}".format(
                    e, tr_running_loss, acc_super, balanced_acc, auc
                )
            )

        if wandb_flag:
            if classifier:
                wandb.log(
                    {
                        "train/Epoch": e,
                        "train/Loss": tr_running_loss,
                        "train/Acc": acc_super,
                    }
                )
            else:
                wandb.log(
                    {
                        "train/Epoch": e,
                        "train/Loss": tr_running_loss / len(trainloader.dataset),
                        "train/Rec Loss": tr_rec_loss / len(trainloader.dataset),
                        "train/Gaussian Loss": tr_gauss_loss / len(trainloader.dataset),
                        "train/Categorical usage": usage,
                        "train/Metric Loss": tr_metric_loss / len(trainloader.dataset),
                        "train/Acc": acc,
                        "train/NMI": nmi_score,
                        "train/usage": usage / len(trainloader.dataset),
                    }
                )

        if validloader is not None:
            model.eval()
            with torch.no_grad():
                usage = np.zeros(model.k)
                (
                    v_running_loss,
                    v_rec_loss,
                    v_gauss_loss,
                    v_cat_loss,
                    v_metric_loss,
                    v_domain_loss,
                ) = (0, 0, 0, 0, 0, 0)
                true_manner_list = []
                gaussian_component = []
                label_list = []
                y_pred_list = []
                domain_list = []
                domain_pred_list = []

                for batch_idx, (data, labels, manner, dataset, id) in enumerate(
                    tqdm(validloader)
                ):
                    # Make sure dtype is Tensor float
                    data = data.to(model.device).float()
                    # Dataset is "albayzin", "neurovoz" or "gita". Map them to 0, 1 and 2 respectively
                    mapping = {"albayzin": 0, "neurovoz": 1, "gita": 2, "italian": 3}
                    dataset = torch.tensor(
                        [mapping[dataset[i]] for i in range(len(dataset))]
                    )
                    # Repeat each dataset window_size times
                    dataset = (
                        dataset.repeat_interleave(manner.shape[1])
                        .to(model.device)
                        .float()
                    )

                    if classifier:
                        # Remove Albayzin to validate
                        # data = data[dataset != "albayzin"].squeeze(0)
                        # manner = manner[dataset != "albayzin"].squeeze(0)
                        # labels = labels[dataset != "albayzin"].squeeze(0)
                        manner[manner > 7] = manner[manner > 7] - 8
                        manner = manner.to(model.device).int()
                        labels = labels.to(model.device).float()
                        y_pred = model.classifier_forward(data, manner)
                        y_pred = torch.sigmoid(y_pred)
                        (
                            complete_loss,
                            recon_loss,
                            gaussian_loss,
                            categorical_loss,
                            metric_loss,
                        ) = model.classifier_loss(y_pred, labels)
                        y_pred_list = np.concatenate(
                            (
                                y_pred_list,
                                y_pred.cpu().detach().numpy().squeeze().reshape(-1),
                            )
                        )
                        label_list = np.concatenate(
                            (label_list, labels.cpu().detach().numpy())
                        )
                    else:
                        if not supervised:
                            # Assert that any manner is no bigger than 7
                            assert torch.max(manner) <= 7
                        # ==== Forward pass ====
                        (
                            x,
                            x_hat,
                            pz_mu,
                            pz_var,
                            y_logits,
                            y,
                            qz_mu,
                            qz_var,
                            z_sample,
                            e_s,
                            e_hat_s,
                            domain_pred,
                        ) = model(data)

                        # ==== Loss ====
                        (
                            complete_loss,
                            recon_loss,
                            gaussian_loss,
                            categorical_loss,
                            metric_loss,
                            domain_loss,
                        ) = model.loss(
                            data,
                            manner,
                            x_hat,
                            z_sample,
                            qz_mu,
                            qz_var,
                            pz_mu,
                            pz_var,
                            y,
                            y_logits,
                            domain_pred,
                            dataset,
                        )

                    # ==== Update metrics ====
                    v_running_loss += complete_loss.item()

                    if not classifier:
                        v_rec_loss += recon_loss.item()
                        v_gauss_loss += gaussian_loss.item()
                        v_cat_loss += categorical_loss.item()
                        v_metric_loss += metric_loss.item()
                        v_domain_loss += domain_loss.item()

                        usage += torch.sum(y, dim=0).cpu().detach().numpy()

                        true_manner_list.append(manner)
                        gaussian_component.append(y.cpu().detach().numpy())
                        domain_list.append(dataset.cpu().detach().numpy())
                        domain_pred = torch.argmax(domain_pred, dim=1)
                        domain_pred_list.append(domain_pred.cpu().detach().numpy())

                # Check reconstruction of X
                if not classifier:
                    check_reconstruction(x, x_hat, wandb_flag, train_flag=True)
                    # Check unsupervised cluster accuracy and NMI
                    true_manner = torch.tensor(np.concatenate(true_manner_list))
                    gaussian_component = torch.tensor(
                        np.concatenate(gaussian_component)
                    )
                    acc = cluster_acc(gaussian_component, true_manner)
                    nmi_score = nmi(gaussian_component, true_manner)
                    domain_acc = accuracy_score(
                        np.concatenate(domain_list), np.concatenate(domain_pred_list)
                    )
                else:
                    best_th_youden, best_th_eer, auc = threshold_selection(
                        label_list, y_pred_list, verbose=1
                    )
                    y_pred = np.round(y_pred_list)
                    acc_super = accuracy_score(label_list, y_pred)
                    bacc = balanced_accuracy_score(label_list, y_pred)
                    print(
                        "Epoch: {} Valid Loss: {:.4f} Acc: {:.4f} Bacc: {:.4f} AUC: {:4f}".format(
                            e, v_running_loss, acc_super, bacc, auc
                        )
                    )

            if not classifier:
                print(
                    "Epoch: {} Valid Loss: {:.4f} Rec Loss: {:.4f} Gaussian Loss: {:.4f} Cat Loss : {:.4f} Metric Loss: {:.4f} Domain Loss: {:.4f} UAcc: {:.4f} NMI: {:.4f} Domain Acc {:.4f}".format(
                        e,
                        v_running_loss / len(validloader.dataset),
                        v_rec_loss / len(validloader.dataset),
                        v_gauss_loss / len(validloader.dataset),
                        v_cat_loss / len(validloader.dataset),
                        v_metric_loss / len(validloader.dataset),
                        v_domain_loss / len(validloader.dataset),
                        acc,
                        nmi_score,
                        domain_acc,
                    )
                )
                print(len(validloader.dataset))
                valid_loss_store.append(v_running_loss)
            else:
                # If supervised, use the balanced accuracy to store the best model. The greater the better
                valid_loss_store.append(v_running_loss)

            if wandb_flag:
                if classifier:
                    wandb.log(
                        {
                            "valid/Epoch": e,
                            "valid/Loss": v_running_loss,
                            "valid/Acc": acc_super,
                        }
                    )
                else:
                    wandb.log(
                        {
                            "valid/Epoch": e,
                            "valid/Loss": v_running_loss / len(validloader.dataset),
                            "valid/Rec Loss": v_rec_loss / len(validloader.dataset),
                            "valid/Gaussian Loss": v_gauss_loss,
                            "valid/Cat Loss": v_cat_loss / len(validloader.dataset),
                            "valid/Metric Loss": v_metric_loss
                            / len(validloader.dataset),
                            "valid/Acc": acc,
                            "valid/NMI": nmi_score,
                            "valid/Categorical usage": usage / len(trainloader.dataset),
                        }
                    )
        # Store best model
        # If the validation loss is the best, save the model
        if valid_loss_store[-1] <= min(valid_loss_store):
            print("Storing the best model at epoch ", e)
            name = path_to_save
            # check if the folder exists if not create it
            if not os.path.exists(name):
                os.makedirs(name)
            name += "/GMVAE_cnn_best_model_2d.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                name,
            )
            # Store best youden th in a txt
            if classifier:
                print("Best threshold Youden: ", best_th_youden)
                print("Best threshold EER: ", best_th_eer)
                name = path_to_save
                namefile = name + "/best_threshold.txt"
                with open(namefile, "w") as f:
                    f.write(str(best_th_youden))

        if not classifier:
            check_reconstruction(x, x_hat, wandb_flag, train_flag=False)

        # Early stopping: If in the last 10 epochs the validation loss has not improved, stop the training
        if e > 40:
            if not classifier:
                if valid_loss_store[-1] > max(valid_loss_store[-10:-1]):
                    print("Early stopping")
                    break
            if classifier:
                if valid_loss_store[-1] > max(valid_loss_store[-10:-1]):
                    print("Early stopping")
                    print("Reloading best model")
                    # Restore best model:
                    name = path_to_save
                    name += "/GMVAE_cnn_best_model_2d.pt"
                    checkpoint = torch.load(name)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
    if wandb_flag:
        if train_flag:
            name = "train/rec_img"
        else:
            name = "valid/rec_img"
        wandb.log({name: fig})
    plt.close(fig)


def MARTA_tester(
    model,
    testloader,
    test_data,
    supervised=False,
    wandb_flag=False,
    path_to_plot=None,
    best_threshold=0.5,
    masked=8,
    train=False,
):
    # Set model in evaluation mode
    model.eval()
    print("Evaluating the VAE model")

    with torch.no_grad():
        # Get batch size from testloader
        batch_size = testloader.batch_size
        y_hat_array = []
        y_array = []

        # Create x_array of shape Batch x Output shape
        x_array = np.zeros((batch_size, 1, 65, 25))
        manner_array = np.zeros((batch_size, 25))
        dataset_array = np.zeros((batch_size))

        x_hat_array = np.zeros(x_array.shape)
        for batch_idx, (x, labels, manner, dataset) in enumerate(tqdm(testloader)):
            # Move data to device
            x = x.to(model.device).float()

            if supervised:
                # ==== Forward pass ====
                manner[manner > 7] = manner[manner > 7] - 8
                if masked < 8:
                    # Mask the data where the manner are equal to mask value
                    mask = torch.ones_like(x)
                    idx_to_mask = torch.where(manner == masked)
                    # idx_to_mask is a tuple of (128, 25) and mask is (128, 1, 65, 25)
                    mask[idx_to_mask[0], :, :, idx_to_mask[1]] = 0
                    x = x * mask
                manner_array = np.concatenate(
                    (manner_array, manner.cpu().detach().numpy())
                )
                dataset_array = np.concatenate((dataset_array, np.array(dataset)))

                manner = manner.to(model.device).int()
                labels = labels.to(model.device).float()

                y_logit = model.classifier_forward(x, manner)
                y_pred = torch.sigmoid(y_logit)
                y_hat_array = np.concatenate(
                    (
                        y_hat_array,
                        y_pred.cpu().detach().numpy().squeeze().reshape(-1),
                    )
                )

                y_array = np.concatenate((y_array, labels.cpu().detach().numpy()))
                (
                    x,
                    x_hat,
                    _,  # pz_mu,
                    _,  # pz_var,
                    _,  # y_logits,
                    _,  # y
                    _,  # qz_mu,
                    _,  # qz_var,
                    _,  # z_sample,
                    _,  # e_s,
                    _,  # e_hat_s,
                    _,  # domain_pred
                ) = model(x)

            else:
                # ==== Forward pass ====
                (
                    x,
                    x_hat,
                    _,  # pz_mu,
                    _,  # pz_var,
                    _,  # y_logits,
                    _,
                    _,  # qz_mu,
                    _,  # qz_var,
                    _,  # z_sample,
                    _,  # e_s,
                    _,  # e_hat_s,
                    _,  # domain_pred
                ) = model(x)
                dataset_array = np.concatenate((dataset_array, np.array(dataset)))

            # Concatenate predictions
            x_hat_array = np.concatenate(
                (x_hat_array, x_hat.cpu().detach().numpy()), axis=0
            )
            x_array = np.concatenate((x_array, x.cpu().detach().numpy()), axis=0)

        print("Removing unused elements")
        # Remove all from GPU to release memory
        del (
            x,
            manner,
            labels,
            x_hat,
            _,
        )

        # Remove the first batch_size elements
        x_array = x_array[batch_size:]
        x_hat_array = x_hat_array[batch_size:]
        dataset_array = dataset_array[batch_size:]

        tasks = ["running_speech", "texts", "all"]

        print("Calculating MSE")
        for dataset_i in np.unique(dataset_array):
            if dataset_i == "albayzin":
                continue
            print("==============================================================")
            print("Calculating results for dataset ", dataset_i)
            idx = dataset_array == dataset_i
            # if in idx is all false, continue
            if not np.any(idx):
                continue
            x_array_dataset = x_array[idx]
            x_hat_array_dataset = x_hat_array[idx]
            test_data_dataset = test_data[test_data["dataset"] == dataset_i]
            if supervised:
                y_hat_array_dataset = y_hat_array[idx]
                y_array_dataset = y_array[idx]

            for task in tasks:
                print("-----------------------------------------")
                print("Calculating results for task ", task)
                if task == "running_speech":
                    idx = (test_data_dataset["text"] == "Monologo") | (
                        test_data_dataset["text"] == "ESPONTANEA"
                    ).astype(bool)
                elif task == "texts":
                    idx = (test_data_dataset["text"] != "Monologo") & (
                        test_data_dataset["text"] != "ESPONTANEA"
                    ).astype(bool)
                else:
                    idx = np.ones(len(test_data_dataset)).astype(bool)

                if supervised:
                    calculate_classification_metrics(
                        x_array_dataset[idx],
                        x_hat_array_dataset[idx],
                        test_data_dataset[idx],
                        y_hat_array_dataset[idx],
                        y_array_dataset[idx],
                        supervised,
                        path_to_plot,
                        dataset_i,
                        best_threshold,
                        train,
                    )
                else:
                    calculate_classification_metrics(
                        x_array_dataset[idx],
                        x_hat_array_dataset[idx],
                        test_data_dataset[idx],
                        None,
                        None,
                        supervised,
                        path_to_plot,
                        dataset_i,
                        best_threshold,
                        train,
                    )


def check_latent_space(model, train_data, test_data, path_to_plot, latent_dim=3):
    import random

    # Set model in evaluation mode
    model.eval()

    # Separate the train and test data based on labels
    train_data_0 = [item for item in train_data if item[1] == 0]
    train_data_1 = [item for item in train_data if item[1] == 1]
    test_data_0 = [item for item in test_data if item[1] == 0]
    test_data_1 = [item for item in test_data if item[1] == 1]

    # Perform stratified random sampling
    train_samples_0 = random.sample(train_data_0, 500)
    train_samples_1 = random.sample(train_data_1, 500)
    test_samples_0 = random.sample(test_data_0, 500)
    test_samples_1 = random.sample(test_data_1, 500)

    # Combine the samples to create the final train and test sets
    train_data = train_samples_0 + train_samples_1
    test_data = test_samples_0 + test_samples_1

    # Construct the dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1024,
        shuffle=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1024,
        shuffle=False,
    )

    store_z_train = []
    store_x_train = []

    for batch_idx, (x, labels, manner, dataset) in enumerate(tqdm(train_dataloader)):
        # Move data to device
        x = x.to(model.device).float()
        # ==== Forward pass ====
        (
            x,
            x_hat,
            pz_mu,
            pz_var,
            y_logits,
            y,
            qz_mu,
            qz_var,
            z_sample,
            e_s,
            e_hat_s,
            domain_pred,
        ) = model(x)

        qz_mu = qz_mu.cpu().detach().numpy().reshape(-1, 25, latent_dim)

        store_z_train.append(qz_mu)
        store_x_train.append(x.cpu().detach().numpy().reshape(-1, 65, 25))

    store_z_test = []
    store_x_test = []
    for batch_idx, (x, labels, manner, dataset) in enumerate(tqdm(test_dataloader)):
        # Move data to device
        x = x.to(model.device).float()
        # ==== Forward pass ====
        (
            x,
            x_hat,
            pz_mu,
            pz_var,
            y_logits,
            y,
            qz_mu,
            qz_var,
            z_sample,
            e_s,
            e_hat_s,
            domain_pred,
        ) = model(x)

        qz_mu = qz_mu.cpu().detach().numpy().reshape(-1, 25, latent_dim)
        store_x_test.append(x.cpu().detach().numpy().reshape(-1, 65, 25))

        store_z_test.append(qz_mu)

    # Calculate tsne 2d for all samples, plot them in different colours
    z_train = np.concatenate(store_z_train)
    x_train = np.concatenate(store_x_train)
    z_test = np.concatenate(store_z_test)
    x_test = np.concatenate(store_x_test)
    all_x = np.concatenate((x_train, x_test), axis=0)
    all_z = np.concatenate((store_z_train, store_z_test), axis=1).squeeze(0)
    # Create labels for train and test datasets
    train_labels = np.concatenate(
        (
            np.zeros(z_train.shape[0] // 2),  # Healthy
            np.ones(z_train.shape[0] // 2),  # Parkinsonian
        )
    )
    test_labels = np.concatenate(
        (
            np.zeros(z_test.shape[0] // 2),  # Healthy
            np.ones(z_test.shape[0] // 2),  # Parkinsonian
        )
    )

    # Combine train and test labels
    labels = np.concatenate((train_labels, test_labels))
    dataset_labels = np.concatenate(
        (np.array(["train"] * z_train.shape[0]), np.array(["test"] * z_test.shape[0]))
    )

    # Combine labels into a single array
    combined_labels = np.array(
        [
            f"{dset}_{'healthy' if lbl == 0 else 'parkinsonian'}"
            for dset, lbl in zip(dataset_labels, labels)
        ]
    )

    # Perform TSNE
    tsne = TSNE(n_components=2, random_state=0)
    z_tsne = tsne.fit_transform(all_z.reshape(all_z.shape[0], -1))

    x_tsne = tsne.fit_transform(all_x.reshape(all_x.shape[0], -1))

    # Plotting
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=z_tsne[:, 0],
        y=z_tsne[:, 1],
        hue=combined_labels,
        style=dataset_labels,
        palette="viridis",
        ax=ax,
        alpha=0.6,
        markers={"train": "s", "test": "o"},
    )
    ax.set_title("TSNE 2D of the latent space")
    plt.legend(title="Labels", loc="best")
    plt.savefig(path_to_plot + "/tsne_2d.png")
    plt.show()

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=x_tsne[:, 0],
        y=x_tsne[:, 1],
        hue=combined_labels,
        style=dataset_labels,
        palette="viridis",
        ax=ax,
        alpha=0.6,
        markers={"train": "s", "test": "o"},
    )
    ax.set_title("TSNE 2D of the latent space")
    plt.legend(title="Labels", loc="best")
    plt.savefig(path_to_plot + "/tsne_2d_spectrograms.png")
    plt.show()


def calculate_classification_metrics(
    x_array_dataset,
    x_hat_array_dataset,
    test_data_dataset,
    y_hat_array_dataset,
    y_array_dataset,
    supervised,
    path_to_plot,
    dataset_i,
    best_threshold,
    train,
):
    from sklearn.metrics import roc_curve, roc_auc_score

    # Calculate mse between x and x_hat
    mse = ((x_array_dataset - x_hat_array_dataset) ** 2).mean(axis=None)
    # Results for all frames
    print(f"Reconstruction loss: {mse}")

    # Plot randomly a test sample and its reconstruction
    idx = np.random.randint(0, x_array_dataset.shape[0])
    x_sample = x_array_dataset[idx].squeeze()
    x_hat_sample = x_hat_array_dataset[idx].squeeze()
    # plot using imshow
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    cmap = cm.get_cmap("viridis")
    axs[0].set_title("Original")
    axs[0].imshow(x_sample, cmap=cmap)
    axs[1].set_title("Reconstruction")
    axs[1].imshow(x_hat_sample, cmap=cmap)
    plt.savefig(path_to_plot + "/rec_img_" + dataset_i + ".png")

    if supervised:
        # Convert predictions and labels to PyTorch tensors
        y_hat_tensor = torch.tensor(y_hat_array_dataset, dtype=torch.float)
        y_tensor = torch.tensor(y_array_dataset, dtype=torch.long)

        print("====== Using 0.5 as threshold ======")
        print("Results for all frames:")
        y_pred = np.round(y_hat_array_dataset)
        print(
            "Value counts of y_predicted: ",
            np.unique(y_pred, return_counts=True),
        )
        print(
            "Value counts of y: ",
            np.unique(y_array_dataset, return_counts=True),
        )
        accuracy = accuracy_score(y_array_dataset, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_array_dataset, y_pred)
        if dataset_i == "neurovoz":
            print("a")
        auc = roc_auc_score(y_array_dataset, y_hat_array_dataset)
        print(
            f"Accuracy: {accuracy:.2f}, Balanced accuracy: {balanced_accuracy:.2f}, AUC: {auc:.2f}"
        )
        # Consensus methods with logits
        (
            mean_log_odds,
            consensus_true,
            consensus_pred,
        ) = soft_output_by_subject_logits(
            y_hat_tensor, y_tensor, test_data_dataset["id_patient"].to_numpy()
        )

        # Calculate metrics for consensus predictions
        accuracy_consensus_logits = accuracy_score(
            consensus_true.numpy(), consensus_pred.numpy()
        )
        balanced_accuracy_consensus_logits = balanced_accuracy_score(
            consensus_true.numpy(), consensus_pred.numpy()
        )
        auc_consensus_logits = roc_auc_score(
            consensus_true.numpy(), torch.sigmoid(mean_log_odds).numpy()
        )

        # Plot the distribution of the sigmoid of the mean log odds for the labels 0 and albels 1 on the same plot
        fig, ax = plt.subplots()
        sns.histplot(
            y_hat_tensor[y_tensor == 0],
            color="blue",
            label="Label 0",
            ax=ax,
            alpha=0.5,
        )
        sns.histplot(
            y_hat_tensor[y_tensor == 1],
            color="red",
            label="Label 1",
            ax=ax,
            alpha=0.5,
        )
        ax.set_title("Distribution of the sigmoid of the mean log odds")
        ax.legend()
        path_to_plot_dist = (
            path_to_plot
            + "/consensus_distribution_"
            + dataset_i
            + "_train_"
            + str(train)
            + ".png"
        )
        plt.savefig(path_to_plot_dist)

        # Print the consensus results
        print("Consensus results with logits:")
        print(
            f"Consensus Accuracy: {accuracy_consensus_logits:.2f}, Consensus Balanced Accuracy: {balanced_accuracy_consensus_logits:.2f}, Consensus AUC: {auc_consensus_logits:.2f}"
        )

        # Implementing consensus method
        mean_soft_odds, consensus_true, consensus_pred = soft_output_by_subject(
            y_hat_tensor, y_tensor, test_data_dataset["id_patient"].to_numpy()
        )

        # Calculate metrics for consensus predictions
        accuracy_consensus = accuracy_score(
            consensus_true.numpy(), consensus_pred.numpy()
        )
        balanced_accuracy_consensus = balanced_accuracy_score(
            consensus_true.numpy(), consensus_pred.numpy()
        )

        auc_consensus = roc_auc_score(consensus_true.numpy(), mean_soft_odds.numpy())

        # Print the consensus results
        print("Consensus results:")
        print(
            f"Consensus Accuracy: {accuracy_consensus:.2f}, Consensus Balanced Accuracy: {balanced_accuracy_consensus:.2f}, Consensus AUC: {auc_consensus:.2f}"
        )

        print("====== Using best threshold (selected in validation) ======")
        print("Best threshold: ", best_threshold)
        y_pred_best_th = np.zeros_like(mean_soft_odds)
        y_pred_best_th[mean_soft_odds >= best_threshold] = 1
        accuracy_consensus_best_th = accuracy_score(
            consensus_true.numpy(), y_pred_best_th
        )
        balanced_accuracy_consensus_best_th = balanced_accuracy_score(
            consensus_true.numpy(), y_pred_best_th
        )
        print(
            f"Consensus Accuracy: {accuracy_consensus_best_th:.2f}, Consensus Balanced Accuracy: {balanced_accuracy_consensus_best_th:.2f}, Consensus AUC: {auc_consensus:.2f}"
        )

        print("====== Using best threshold (selected in test) ======")
        # Get best threshold via Youden's J statistic
        fpr, tpr, thresholds = roc_curve(consensus_true.numpy(), mean_soft_odds.numpy())
        j_scores = tpr - fpr
        best_threshold = thresholds[np.argmax(j_scores)]

        auc_consensus = roc_auc_score(consensus_true.numpy(), mean_soft_odds.numpy())
        print("Best threshold: ", best_threshold)
        y_pred_best_th = np.zeros_like(mean_soft_odds)
        y_pred_best_th[mean_soft_odds >= best_threshold] = 1
        accuracy_consensus_best_th = accuracy_score(
            consensus_true.numpy(), y_pred_best_th
        )
        balanced_accuracy_consensus_best_th = balanced_accuracy_score(
            consensus_true.numpy(), y_pred_best_th
        )
        print(
            f"Consensus Accuracy: {accuracy_consensus_best_th:.2f}, Consensus Balanced Accuracy: {balanced_accuracy_consensus_best_th:.2f}, Consensus AUC: {auc_consensus:.2f}"
        )

        # Calculate results per patient
        accuracy_per_patient = []
        balanced_accuracy_per_patient = []
        for i in test_data_dataset["id_patient"].unique():
            # Get the predictions for the patient
            y_patient = y_array_dataset[test_data_dataset.id_patient == i]
            y_hat_patient = y_hat_array_dataset[test_data_dataset.id_patient == i]

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


def threshold_selection(y_true, y_pred_soft, verbose=0):
    from sklearn.metrics import roc_curve

    # Select best threshold by youden index
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_soft)
    j_scores = tpr - fpr
    youden_th = thresholds[np.argmax(j_scores)]

    # Select best threshold by EER
    fnr = 1 - tpr
    eer_threshold = thresholds[np.argmin(np.absolute((fnr - fpr)))]

    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_soft)

    return youden_th, eer_threshold, auc


def soft_output_by_subject(output_test, Y_test, subject_group_test):
    unique_subjects = np.unique(subject_group_test)
    Y_test_bySubject = []
    mean_probabilities = torch.zeros(len(unique_subjects))

    for i, subject in enumerate(unique_subjects):
        subject_indices = np.where(subject_group_test == subject)
        subject_outputs = output_test[subject_indices]

        # Calculate mean probability for the subject
        mean_probabilities[i] = torch.mean(subject_outputs)

        # Store the first label found for the subject
        Y_test_bySubject.append(Y_test[subject_indices][0])

    # Estimate labels based on mean probability
    estimated_labels = torch.zeros_like(mean_probabilities)
    estimated_labels[mean_probabilities >= 0.5] = 1

    Y_test_tensor_bySubject = torch.tensor(Y_test_bySubject, dtype=torch.long)

    return mean_probabilities, Y_test_tensor_bySubject, estimated_labels


def soft_output_by_subject_logits(output_test, Y_test, subject_group_test):
    unique_subjects = np.unique(subject_group_test)
    Y_test_bySubject = []
    mean_log_odds = torch.zeros(len(unique_subjects))

    for i, subject in enumerate(unique_subjects):
        subject_indices = np.where(subject_group_test == subject)
        subject_outputs = output_test[subject_indices]

        # Calculate mean log odds for the subject
        log_odds = torch.log(subject_outputs + 1e-6) - torch.log(
            1 - subject_outputs + 1e-6
        )
        mean_log_odds[i] = torch.mean(log_odds)

        # Store the first label found for the subject
        Y_test_bySubject.append(Y_test[subject_indices][0])

    # Estimate labels based on mean log odds
    estimated_labels = torch.zeros_like(mean_log_odds)
    estimated_labels[mean_log_odds >= 0] = 1

    Y_test_tensor_bySubject = torch.tensor(Y_test_bySubject, dtype=torch.long)

    return mean_log_odds, Y_test_tensor_bySubject, estimated_labels
