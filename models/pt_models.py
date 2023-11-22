import torch
import timm
import numpy as np
import copy
import time
from utils import log_normal
from pytorch_metric_learning import miners, losses, reducers, samplers


width = 64


def Selec_embedding(model_name, **params):
    if model_name == "Resnet_18":
        model = Resnet_Encoder(**params)
    elif model_name == "Vgg_11":
        model = VGG_Encoder(**params)
    elif model_name == "ViT":
        model = ViT_Encoder(**params)
    else:
        raise Exception("No encoder selected")
    return model


def Selec_model_two_classes(model_name, **params):
    if model_name == "Resnet_18":
        model = ResNet_TwoClass(**params)
    elif model_name == "Vgg_11":
        model = VGG_TwoClass(**params)
    elif model_name == "ViT":
        model = ViT_TwoClass(**params)
    return model


class ViT_TwoClass(torch.nn.Module):
    def __init__(self, channels=3, freeze=True):
        super(ViT_TwoClass, self).__init__()

        self.ViT = timm.create_model(
            "vit_base_patch16_224", pretrained=True, in_chans=channels, num_classes=0
        )

        for param in self.ViT.parameters():
            param.requires_grad = not freeze

        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.ViT.num_features, 2 * width)
        self.output = torch.nn.Linear(2 * width, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.ViT(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output(x)
        return x


class ResNet_TwoClass(torch.nn.Module):
    def __init__(self, num_layers=18, channels=3, freeze=True):
        super(ResNet_TwoClass, self).__init__()

        if num_layers == 18:
            Rnet = timm.create_model(
                "resnet18",
                pretrained=True,
                in_chans=channels,
                num_classes=0,
                global_pool="avg",
            )
        elif num_layers == 50:
            Rnet = timm.create_model(
                "resnet50",
                pretrained=True,
                in_chans=channels,
                num_classes=0,
                global_pool="avg",
            )

        for param in Rnet.parameters():
            param.requires_grad = not freeze

        self.Rnet = Rnet
        self.Rnet.fc = torch.nn.Sequential(
            torch.nn.Linear(self.Rnet.num_features, 2 * width),
            torch.nn.Dropout(p=0.6),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * width, 2),
        )

    def forward(self, x):
        x = self.Rnet(x)
        return x


class VGG_TwoClass(torch.nn.Module):
    def __init__(self, num_layers=11, channels=3, freeze=True):
        super(VGG_TwoClass, self).__init__()

        if num_layers == 11:
            Rnet = timm.create_model(
                "vgg11_bn",
                pretrained=True,
                in_chans=channels,
                num_classes=0,
                global_pool="avg",
            )
        elif num_layers == 13:
            Rnet = timm.create_model(
                "vgg13_bn",
                pretrained=True,
                in_chans=channels,
                num_classes=0,
                global_pool="avg",
            )
        elif num_layers == 16:
            Rnet = timm.create_model(
                "vgg16_bn",
                pretrained=True,
                in_chans=channels,
                num_classes=0,
                global_pool="avg",
            )
        elif num_layers == 19:
            Rnet = timm.create_model(
                "vgg19_bn",
                pretrained=True,
                in_chans=channels,
                num_classes=0,
                global_pool="avg",
            )

        for param in Rnet.parameters():
            param.requires_grad = not freeze

        self.Rnet = Rnet
        self.Rnet.head.fc = torch.nn.Sequential(
            torch.nn.Linear(self.Rnet.num_features, 2 * width),
            torch.nn.Dropout(p=0.6),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * width, 2),
        )

    def forward(self, x):
        x = self.Rnet(x)
        return x


class ViT_Encoder(torch.nn.Module):
    def __init__(self, channels=3, freeze=True):
        super(ViT_Encoder, self).__init__()

        self.ViT = timm.create_model(
            "vit_base_patch16_224", pretrained=True, in_chans=channels, num_classes=0
        )
        self.linear1 = torch.nn.Linear(self.ViT.num_features, 2 * width)
        self.relu = torch.nn.ReLU()

        for param in self.ViT.parameters():
            param.requires_grad = not freeze

        self.embedding = torch.nn.Sequential(self.ViT, self.linear1, self.relu)

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * width, out_features=width),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=width, out_features=width),
        )

        self.linear_probe = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * width, out_features=2)
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def calculate_linear_probe(self, x):
        x = self.embedding(x)
        return self.linear_probe(x)

    def forward(self, x):
        embedding = self.embedding(x)

        projection = self.projection_head(embedding)

        return projection


class Resnet_Encoder(torch.nn.Module):
    def __init__(self, num_layers=18, channels=3, freeze=True):
        super(Resnet_Encoder, self).__init__()

        if num_layers == 18:
            self.Rnet = timm.create_model(
                "resnet18", pretrained=True, in_chans=channels, num_classes=0
            )
        elif num_layers == 50:
            self.Rnet = timm.create_model(
                "resnet50", pretrained=True, in_chans=channels, num_classes=0
            )
        self.linear1 = torch.nn.Linear(self.Rnet.num_features, 2 * width)
        self.relu = torch.nn.ReLU()

        for param in self.Rnet.parameters():
            param.requires_grad = not freeze

        self.embedding = torch.nn.Sequential(self.Rnet, self.linear1, self.relu)

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * width, out_features=width),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=width, out_features=width),
        )

        self.linear_probe = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * width, out_features=2)
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def calculate_linear_probe(self, x):
        x = self.embedding(x)
        return self.linear_probe(x)

    def forward(self, x):
        embedding = self.embedding(x)

        projection = self.projection_head(embedding)

        return projection


class VGG_Encoder(torch.nn.Module):
    def __init__(self, num_layers=11, channels=3, freeze=True):
        super(VGG_Encoder, self).__init__()

        if num_layers == 11:
            VGG = timm.create_model(
                "vgg11_bn", pretrained=True, in_chans=channels, num_classes=0
            )
        elif num_layers == 13:
            VGG = timm.create_model(
                "vgg13_bn", pretrained=True, in_chans=channels, num_classes=0
            )
        elif num_layers == 16:
            VGG = timm.create_model(
                "vgg16_bn", pretrained=True, in_chans=channels, num_classes=0
            )
        elif num_layers == 19:
            VGG = timm.create_model(
                "vgg19_bn", pretrained=True, in_chans=channels, num_classes=0
            )

        for param in VGG.parameters():
            param.requires_grad = not freeze

        VGG.head.fc = torch.nn.Sequential(
            torch.nn.Linear(VGG.num_features, 2 * width),
            torch.nn.ReLU(),
        )

        self.embedding = VGG

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * width, out_features=width),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=width, out_features=width),
        )

        self.linear_probe = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * width, out_features=2)
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def calculate_linear_probe(self, x):
        x = self.embedding(x)
        return self.linear_probe(x)

    def forward(self, x):
        embedding = self.embedding(x)

        projection = self.projection_head(embedding)

        return projection


def finetuning_model(embedding_input, freeze=True):
    class final_model(torch.nn.Module):
        def __init__(self, embedding, freeze=True):
            super(final_model, self).__init__()

            for param in embedding.parameters():
                param.requires_grad = not freeze

            self.embedding = embedding
            self.linear = torch.nn.Linear(2 * width, width)
            self.output = torch.nn.Linear(width, 2)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(p=0.4)

        def forward_emb(self, x):
            return self.embedding(x)

        def forward(self, x):
            x = self.embedding(x)
            x = self.linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.output(x)
            return x

    return final_model(embedding_input, freeze=freeze)


class Spectrogram_networks_vowels(torch.nn.Module):
    def __init__(self, x_dim, hidden_dims, cnn=False):
        super().__init__()

        self.x_dim = x_dim
        self.hidden_dims = hidden_dims
        self.cnn = cnn

        # ===== Inference =====
        # Deterministic x_hat = g(x)
        if cnn:
            self.inference_gx = torch.nn.Sequential(
                torch.nn.Conv2d(self.x_dim, self.hidden_dims[0], 3),
                torch.nn.BatchNorm2d(self.hidden_dims[0]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], 3),
                torch.nn.BatchNorm2d(self.hidden_dims[1]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(self.hidden_dims[1], self.hidden_dims[2], 3),
                torch.nn.BatchNorm2d(self.hidden_dims[2]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
            )
            self.x_hat_shape = self.inference_gx(torch.zeros(1, 1, 65, 33)).shape
        else:
            raise NotImplementedError

        # ===== Generative =====
        # Deterministic x = f(x_hat)
        if cnn:
            self.generative_fxhat = torch.nn.Sequential(
                # ConvTranspose
                torch.nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[1], 5),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2),
                # ConvTranspose
                torch.nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[2], 5),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2),
                # ConvTranspose
                torch.nn.ConvTranspose2d(
                    self.hidden_dims[2],
                    self.x_dim,
                    stride=[2, 1],
                    kernel_size=[3, 6],
                    padding=[4, 0],
                ),
            )
        else:
            raise NotImplementedError


# ======================================= Speech Therapist =======================================


class SpeechTherapist(torch.nn.Module):
    """SpeechTherapist class. The purpose is to train with healthy patients to learn the latent space representation of each phoneme clusterized by manner class. Then, when we test with parkinsonian patients, we will use the latent space representation
    to measure the distance between the latent space representation of each phoneme and the clusters learned by the SpeechTherapist. If the distance is too high, then the patient is parkinsonian. If the distance is low, then the patient is healthy. We are learning how the pronunciation of each manner class
    differ between healthy and parkinsonian patients.
    This class will be wrapper composed by three main components:
    1. SpecEncoder: this network will encode the spectrograms of N, 1, 65, 33 into a feature representation of N*33, Channels*Height.
    2- GMVAE: a Gaussian Mixture VAE. Its input will be a flatten version of the output of the SpecEncoder, i.e., N*33, (to determine). So, it will provide a latent space representation of each window of the spectrogram.
        2.1 This GMVAE will use metric learning to learn the latent space representation of each window of the spectrogram to match clusters with the manner class of each phoneme. Each window represents a phoneme, therefore, each window will be assigned to a manner class and the GMVAE
        will be trained to match the clusters of the latent space with the manner class of each phoneme.
        2.2 The GMVAE will reconstruct the N*33, (to determine) input, i.e., the flatten version of the output of the SpecEncoder.
    3. AudioDecoder: this network will decode the output of the GMVAE, i.e., N*33, (to determine), into a spectrogram of N, 1, 65, 33.

    The loss terms will be:
    1. Reconstruction term of the GMVAE: MSE between both spectrograms, input and output of SpecEncoder and AudioDecoder, respectively.
    2. GaussianLoss of the GMVAE: KL divergence between the latent space representation of each window of the spectrogram andthe Gaussian Mixture.
    3. CategoricalLoss of the GMVAE: KL divergence between the Gaussian components of the latent space representation of each window of the spectrogram and a Uniform of K components determining the number of Gaussian of the Gaussian Mixture.
    3. Metric learning loss of the GMVAE: the metric learning loss will be the triplet loss. The anchor will be the latent space representation of each window of the spectrogram, the positive will be the latent space representation of the same manner class of the anchor and the negative will be the latent space representation of the other manner classes.
    """

    def __init__(self, x_dim, hidden_dims_spectrogram, hidden_dims_gmvae):
        self.x_dim = x_dim
        self.hidden_dims_spectrogram = hidden_dims_spectrogram
        self.hidden_dims_gmvae = hidden_dims_gmvae

        # ============ Instantiate the networks ============
        # From Spectrogram (x) to encoded spectrogram (e_s)
        self.SpecEncoder()
        # From encoded spectrogram (e_s) to latent space representation (z)
        self.inference()
        # From latent space representation (z) to encoded spectrogram (e_s)
        self.generative()
        # From encoded spectrogram (e_s) to spectrogram (x)
        self.SpecDecoder()

        sumreducer = reducers.SumReducer()
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self.metric_loss_func = losses.GeneralizedLiftedStructureLoss(
            reducer=sumreducer
        ).to(self.device)

    def SpecEncoder(self):
        """This network will encode the spectrograms of N, 1, 65, 33 into a feature representation of N*33, Channels*Height."""
        self.spec_enc = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.x_dim,
                self.hidden_dims_spectrogram[0],
                kernel_size=[3, 3],
                padding=[0, 1],
                stride=[2, 1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.hidden_dims[0]),
            torch.nn.Conv2d(
                self.hidden_dims_spectrogram[0],
                self.hidden_dims_spectrogram[1],
                kernel_size=[3, 3],
                padding=[0, 1],
                stride=[2, 1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.hidden_dims_spectrogram[1]),
            torch.nn.Conv2d(
                self.hidden_dims_spectrogram[1],
                self.hidden_dims_spectrogram[2],
                kernel_size=[3, 3],
                padding=[0, 1],
                stride=[2, 1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.hidden_dims_spectrogram[2]),
            Flatten(),
        )

        self.x_hat_shape_before_flatten = self.inference_gx(
            torch.zeros(1, 1, 65, 33)
        ).shape
        self.x_hat_shape_after_flatten = self.flatten(
            self.inference_gx(torch.zeros(1, 1, 65, 33))
        ).shape

        return self.spec_enc

    def SpecDecoder(self):
        """This network will decode the output of the GMVAE, i.e., N*33, Channels*Height, into a spectrogram of N, 1, 65, 33."""
        self.spec_dec = torch.nn.Sequential(
            UnFlatten(),
            # ConvTranspose
            torch.nn.ConvTranspose2d(
                self.hidden_dims_spectrogram[2],
                self.hidden_dims_spectrogram[1],
                kernel_size=[5, 3],
                padding=[0, 1],
                stride=[2, 1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.hidden_dims_spectrogram[1]),
            # ConvTranspose
            torch.nn.ConvTranspose2d(
                self.hidden_dims_spectrogram[1],
                self.hidden_dims_spectrogram[0],
                kernel_size=[5, 3],
                padding=[0, 1],
                stride=[2, 1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.hidden_dims_spectrogram[0]),
            # ConvTranspose
            torch.nn.ConvTranspose2d(
                self.hidden_dims_spectrogram[0],
                self.x_dim,
                stride=[2, 1],
                kernel_size=[3, 3],
                padding=[5, 1],
            ),
        )
        return self.spec_dec

    def inference(self):
        # p(y | e_s): the probability of each gaussian component givben the encoded spectrogram (e_s)
        # This networks get the e_s (N*33, Channels*Height), reduces dimensionality to N*33, h_dim, and then applies GumbelSoftmax to N*33, K components.
        self.py_es = torch.nn.Sequential(
            torch.nn.Linear(self.x_hat_shape_after_flat[1], self.hidden_dims_gmvae[1]),
            torch.nn.ReLU(),  # Check if makes sense to have a ReLU here
            GumbelSoftmax(self.hidden_dims[1], self.k, device=self.device),
        ).to(self.device)

        # y_hat = h(y): the output of the GumbelSoftmax is the probability of each gaussian component given the encoded spectrogram (e_s).
        # This network will upsample the K components to match the dimensionality of the encoded spectrogram (e_s), i.e., N*33, Channels*Height.
        self.hy = torch.nn.Linear(self.k, self.x_hat_shape_after_flat[1]).to(
            self.device
        )

        # q(z | e_s, y): the GaussianMixture network will get the encoded spectrogram (e_s) and the GumbelSoftmax output (y) and will output the parameters of the GaussianMixture, i.e., mu and logvar.
        self.qz_es_y = torch.nn.Sequential(
            torch.nn.Linear(
                self.x_hat_shape_after_flat[1], self.z_dim * 2
            ),  # Check if its enough only with one layer
        ).to(self.device)

        # p(z | y): the GaussianMixture prior will get the GumbelSoftmax output (y) and will output the parameters of the GaussianMixture, i.e., mu and logvar.
        self.pz_y = torch.nn.Sequential(
            torch.nn.Linear(
                self.k, self.z_dim * 2
            ),  # Check if its enough only with one layer
        ).to(self.device)

    def generative(self):
        # This network will get the latent space representation of each window of the spectrogram (z) and will output the feature representation of
        # p(e_s | z): the output of the GMVAE will be the reconstruction of the input, i.e., the encoded spectrogram (N*33, Channels*Height).
        self.pes_z = torch.nn.Sequential(
            torch.nn.Linear(
                self.z_dim, self.x_hat_shape_after_flat[1]
            ),  # Maybe we can use more layers here
        ).to(self.device)

    def spec_encoder_forward(self, x):
        """Forward function of the spectrogram encoder network. It receives the spectrogram (x) and outputs the encoded spectrogram (e_s)."""
        e_s = self.spec_enc(x)
        return e_s

    def infere_forward(self, e_s):
        """Forward function of the inference network. It receives the encoded spectrogram (e_s) and outputs the parameters of the GaussianMixture, i.e., mu and logvar."""
        # p(y | e_s): the probability of each gaussian component givben the encoded spectrogram (e_s). Returns the logits, the softmax probability and the gumbel softmax output, respectively.
        y_logits, prob, y = self.py_es(e_s)

        # y_hat = h(y): now we upsample the K components to match the dimensionality of the encoded spectrogram (e_s), i.e., N*33, Channels*Height.
        y_hat = self.hy(y)

        # We combine both the e_s and y_hat to create the input to the q(z | e_s, y) network.
        es_y = e_s + y_hat

        # q(z | e_s, y): the GaussianMixture network will get the encoded spectrogram (e_s) and the GumbelSoftmax output upsampled (y_hat) and will output the parameters of the GaussianMixture, i.e., mu and logvar.
        qz_mu, qz_logvar = self.qz_es_y(es_y).chunk(2, dim=1)

        # p(z | y): the GaussianMixture prior will get the GumbelSoftmax output (y) and will output the parameters of the GaussianMixture, i.e., mu and logvar.
        pz_mu, pz_logvar = self.pz_y(y).chunk(2, dim=1)
        pz_var = torch.exp(pz_logvar)

        # reparemeterization trick
        qz_var = torch.exp(qz_logvar)
        std = torch.exp(0.5 * qz_logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(qz_mu)

        return (z, y_logits, y, qz_mu, qz_var, pz_mu, pz_var)

    def generative_forward(self, z_sample):
        """Forward function of the generative network. It receives the samples of the latent space (z_sample) and outputs the feature representation of the spectrogram reconstruction (e_hat_s)."""
        # p(e_hat_s | z): the output of the GMVAE will be the reconstruction of the input, i.e., the encoded spectrogram (N*33, Channels*Height).
        e_hat_s = self.pes_z(z_sample)
        return e_hat_s

    def spec_dec_forward(self, e_hat_s):
        """Forward function of the spectrogram decoder network. It receives the encoded spectrogram reconstruction (e_hat_s) and outputs the spectrogram reconstruction (x_hat)."""
        x_hat = self.pes_z(e_hat_s)
        return x_hat

    def forward(self, x):
        """Forward function of the SpeechTherapist. It receives the spectrogram (x) and outputs the spectrogram reconstruction (x_hat)."""
        e_s = self.spec_encoder_forward(x)
        z_sample, y_logits, y, qz_mu, qz_logvar, pz_mu, pz_var = self.inference_forward(
            e_s
        )
        e_hat_s = self.generative_forward(z_sample)
        x_hat = self.spec_dec_forward(e_hat_s)
        return (
            x_hat,
            pz_mu,
            pz_var,
            y_logits,
            y,
            qz_mu,
            qz_logvar,
            z_sample,
            e_s,
            e_hat_s,
        )

    def metric_loss(self, z_mu, manner_classes):
        """This function computes the metric loss of the GMVAE. It receives the latent space representation of each window of the spectrogram (z_mu) and the manner class of each phoneme (manner_classes).
        Currently we are ignoring the silences and affricates as they are minor in the datase, but we should find a way to include them in the metric loss.
        """

        # Silences and affricates should be not used to compute the metric loss
        manner_classes = manner_classes.reshape(-1)
        idx = np.where((manner_classes != 6) & (manner_classes != 7))[
            0
        ]  # 6: affricates, 7: silences
        z_mu = z_mu[idx]
        manner_classes = manner_classes[idx]

        # Oversample labels and x_hat to have the same number of samples
        unique_labels, count_labels = np.unique(manner_classes, return_counts=True)
        max_count = np.max(count_labels)

        # Generate indices for oversampling
        idx_sampled = np.concatenate(
            [
                np.random.choice(np.where(manner_classes == label)[0], max_count)
                for label in unique_labels
            ]
        )

        # Apply oversampling
        manner_classes = manner_classes[idx_sampled]
        z_mu = z_mu[idx_sampled]

        # Miner
        hard_pairs = self.miner(z_mu, manner_classes)

        # Loss
        loss = self.metric_loss_func(z_mu, manner_classes, hard_pairs)

        return loss

    def loss(self, x, manner_classes):
        """This function computes the loss of the SpeechTherapist. It receives the spectrogram (x) and the manner class of each phoneme (manner_classes)."""
        (
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
        ) = self.forward(x)

        # Reconstruction loss
        recon_loss = self.mse_loss(x_hat, x)

        # Gaussian loss
        gaussian_loss = torch.sum(
            self.log_normal(z_sample, qz_mu, qz_var)  # q_phi(z|x,y)
            - self.log_normal(z_sample, pz_mu, pz_var)  # p_theta(z|y)
        )

        # Categorical loss
        categorical_loss = self.categorical_loss(y_logits, y)

        # Metric loss
        metric_loss = self.metric_loss(z_sample, manner_classes)

        return recon_loss, gaussian_loss, categorical_loss, metric_loss


class GMVAE(torch.nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        n_gaussians=10,
        hidden_dims=[20, 10],
        class_dims=[64, 32, 10],
        n_manner=8,
        ss=False,
        supervised=False,
        weights=[1, 3, 10, 10, 10],
        cnn=True,
        device=None,
        cnn_classifier=False,
    ):
        super().__init__()
        self.device = device

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.k = n_gaussians
        self.ss = ss
        self.num_labeled = 0
        self.ss_mask = None
        self.supervised = supervised
        self.hidden_dims = hidden_dims
        self.cnn = cnn
        self.cnn_classifier = cnn_classifier
        self.manner = n_manner
        self.usage = np.zeros(self.k)

        # Spectrogram networks
        self.spec_nets = Spectrogram_networks_manner(
            x_dim=self.x_dim, hidden_dims=self.hidden_dims, cnn=cnn
        )

        # Inference
        self.inference_networks(cnn=cnn)

        # Generative
        self.generative_networks(cnn=cnn)

        # Classifier: initialize it always but only use it if supervised
        self.class_dims = class_dims
        self.classifier()

        self.w1, self.w2, self.w3, self.w4, self.w5 = weights

        # ===== Loss =====
        self.mse_loss = torch.nn.MSELoss(reduction="sum")
        if self.supervised:
            self.cross_entropy_loss = torch.nn.BCELoss(reduction="sum")

        # weight initialization
        for m in self.modules():
            if (
                type(m) == torch.nn.Linear
                or type(m) == torch.nn.Conv2d
                or type(m) == torch.nn.ConvTranspose2d
            ):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    torch.nn.init.constant_(m.bias, 0)

        print("Device used for training: ", self.device)
        self.to(self.device)

    def classifier(self):
        self.hmc = torch.nn.Embedding(self.manner, self.z_dim)

        self.clf_cnn = torch.nn.Sequential(
            # 2DConv with kernel_size = (32, 3), stride=2, padding=1
            torch.nn.Conv2d(
                1,
                self.class_dims[0],
                kernel_size=[3, 32],
                stride=[2, 1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.class_dims[0]),
            torch.nn.Conv2d(
                self.class_dims[0],
                self.class_dims[1],
                kernel_size=[3, 1],
                stride=[2, 1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.class_dims[1]),
            # Flatten
            ClassifierFlatten(),
            # Linear
            torch.nn.Linear(32 * 7, self.class_dims[2]),
            torch.nn.ReLU(),
            # Dropout
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(self.class_dims[2], 1),
            torch.nn.Sigmoid(),
        )

        self.clf_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def inference_networks(self, cnn=False):
        # ===== Inference =====
        # Deterministic x_hat = g(x)
        self.inference_gx = self.spec_nets.inference_gx.to(self.device)
        self.flatten = self.spec_nets.flatten.to(self.device)
        self.x_hat_shape_before_flat = self.spec_nets.x_hat_shape_before_flatten
        self.x_hat_shape_after_flat = self.spec_nets.x_hat_shape_after_flatten

        # q(y | x_hat)
        if cnn:
            self.inference_qy_x = torch.nn.Sequential(
                torch.nn.Linear(self.x_hat_shape_after_flat[1], self.hidden_dims[1]),
            ).to(self.device)
        else:
            raise NotImplementedError

        # Gumbel softmax
        self.gumbel_softmax = torch.nn.Sequential(
            GumbelSoftmax(self.hidden_dims[1], self.k, device=self.device)
        )

        # y_hat = h(qy)
        self.hqy = torch.nn.Linear(self.k, self.x_hat_shape_after_flat[1]).to(
            self.device
        )

        # q(z | x_hat, y)
        if cnn:
            self.inference_qz_xy = torch.nn.Sequential(
                torch.nn.Linear(self.x_hat_shape_after_flat[1], self.z_dim * 2),
            ).to(self.device)
        else:
            raise NotImplementedError

    def generative_networks(self, cnn=False):
        # ===== Generative =====
        # p(z | y)
        self.generative_pz_y = torch.nn.Sequential(
            torch.nn.Linear(self.k, self.z_dim * 2),
        ).to(self.device)

        # p(x_hat | z)
        if cnn:
            self.generative_pxhat_z = torch.nn.Sequential(
                torch.nn.Linear(self.z_dim, self.x_hat_shape_after_flat[1]),
            ).to(self.device)

            self.unflatten = self.spec_nets.unflatten.to(self.device)

            self.generative_fxhat = self.spec_nets.generative_fxhat.to(self.device)
        else:
            raise NotImplementedError

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def infere(self, x: torch.Tensor) -> torch.Tensor:
        # x_hat = g(x)
        x_hat_unflatten = self.inference_gx(x)

        # Flatten
        x_hat = self.flatten(x_hat_unflatten)

        # q(y | x)
        qy_logits, probs, qy = self.gumbel_softmax(self.inference_qy_x(x_hat))

        # q(z | x, y)
        # Transform Y in the same shape as X
        if self.cnn:
            y_hat = self.hqy(qy)
            xy = x_hat + y_hat
        else:
            raise NotImplementedError
        qz_mu, qz_logvar = torch.chunk(self.inference_qz_xy(xy), 2, dim=1)

        z = self.reparametrize(qz_mu, qz_logvar)

        return z, qy_logits, qy, qz_mu, qz_logvar, x_hat, x_hat_unflatten

    def log_normal(self, x, mu, var):
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1
        )

    def generate(self, z, y) -> torch.Tensor:
        # p(z | y)
        z_mu, z_logvar = torch.chunk(self.generative_pz_y(y), 2, dim=1)
        z_var = torch.nn.functional.softplus(z_logvar)

        # p(x_hat | z)
        x_hat = self.generative_pxhat_z(z)

        # Unflatten x_hat
        x_hat = self.unflatten(x_hat)

        # x_rec = f(x_hat)
        x_rec = self.generative_fxhat(x_hat)

        return x_rec, z_mu, z_var

    def classifier_forward(self, z, manner, labels):
        manner = manner.to(self.device)
        hmc = self.hmc_cnn(manner)
        window_size = copy(hmc.shape[1])
        # Calculate torch embedding. Transform manner (shape: (batch, window)) to (batch, window, z_dim)
        if not self.cnn_classifier:
            # For MLP, swap window and z_dim: now is (batch, z_dim, window)
            hmc = hmc.permute(0, 2, 1)
            # Now flatten: now is (batch, z_dim*window)
            hmc = hmc.reshape(hmc.shape[0], -1)

        # Now z is (batch*window, z_dim), reshape to: (batch, window, z_dim)
        z = z.reshape(hmc.shape[0], window_size, self.z_dim)

        # \tilde{z} = z + h(mc)
        if self.cnn_classifier:
            # Directly sum them and add a dimension for the channel. From (batch, window, z_dim) to (batch, 1, window, z_dim)
            z_hat = (z + hmc).unsqueeze(1)
        else:
            # For MLP, flatten both z and hmc from (batch, window, z_dim) to (batch, window*z_dim)
            z = z.reshape(z.shape[0], -1)
            hmc = hmc.reshape(hmc.shape[0], -1)
            z_hat = z + hmc

        # Oversample z_hat and labels to have the same number of samples in each class
        # Oversample z_hat and labels to have the same number of samples in each class
        unique_labels, count_labels = np.unique(labels, return_counts=True)
        max_count = np.max(count_labels)

        # Generate indices for oversampling
        idx_sampled = np.concatenate(
            [
                np.random.choice(np.where(labels == label)[0], max_count)
                for label in unique_labels
            ]
        )

        # Apply oversampling
        labels = labels[idx_sampled]
        z_hat = z_hat[idx_sampled]

        if self.cnn_classifier:
            y_pred = self.clf_cnn(z_hat)
        else:
            y_pred = self.clf_mlp(z_hat)

        return y_pred, labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, qy_logits, qy, qz_mu, qz_logvar, x_hat, x_hat_unflatten = self.infere(x)
        x_rec, z_mu, z_var = self.generate(z, qy)

        return (
            x_rec,
            z_mu,
            z_var,
            qy_logits,
            qy,
            qz_mu,
            qz_logvar,
            z,
            x_hat,
            x_hat_unflatten,
        )

    def metric_loss(self, x_hat, labels):
        sumreducer = reducers.SumReducer()
        miner = miners.MultiSimilarityMiner(epsilon=0.1)
        loss_func = losses.GeneralizedLiftedStructureLoss(reducer=sumreducer).to(
            self.device
        )

        # Silences and affricates should be not used to compute the metric loss
        labels = labels.reshape(-1)
        idx = np.where((labels != 6) & (labels != 7))[0]  # 6: affricates, 7: silences
        x_hat = x_hat[idx]
        labels = labels[idx]

        # Oversample labels and x_hat to have the same number of samples
        unique_labels, count_labels = np.unique(labels, return_counts=True)
        max_count = np.max(count_labels)

        # Generate indices for oversampling
        idx_sampled = np.concatenate(
            [
                np.random.choice(np.where(labels == label)[0], max_count)
                for label in unique_labels
            ]
        )

        # Apply oversampling
        labels = labels[idx_sampled]
        x_hat = x_hat[idx_sampled]

        # Miner
        hard_pairs = miner(x_hat, labels)

        # Loss
        metric_loss = loss_func(x_hat, labels, hard_pairs)

        return metric_loss

    def loss(
        self, x: torch.Tensor, labels=None, manner=None, e=0, idx_sampled=[]
    ) -> torch.Tensor:
        (
            x_rec,
            z_mu,
            z_var,
            qy_logits,
            qy,
            qz_mu,
            qz_logvar,
            z,
            x_hat,
            x_hat_unflatten,
        ) = self.forward(x)

        # reconstruction loss
        rec_loss = self.mse_loss(x_rec, x)

        # Gaussian loss = KL divergence between q(z|x,y) and p(z|y)
        gaussian_loss = torch.sum(
            self.log_normal(z, qz_mu, torch.exp(qz_logvar))  # q_phi(z|x,y)
            - self.log_normal(z, z_mu, z_var)  # p_theta(z|y)
        )

        # Cat loss = KL divergence between q(y|x) and p(y)
        # KL Divergence between the posterior and a prior uniform distribution (U(0,1))
        #  loss = (1/n) * Σ(qx * log(qx/px)), because we use a uniform prior px = 1/k
        #  loss = (1/n) * Σ(qx * (log(qx) - log(1/k)))
        log_q = torch.log_softmax(qy_logits, dim=-1)
        log_p = torch.log(1 / torch.tensor(self.k))
        cat_loss = torch.sum(qy * (log_q - log_p), dim=-1)
        cat_loss = torch.sum(cat_loss)

        if self.supervised:
            y_pred, labels = self.classifier_forward(z, manner, labels)
            # Reshape labels tensor and make it float
            labels = labels.view(-1, 1).float().to(self.device)
            clf_loss = self.cross_entropy_loss(y_pred, labels)
        else:
            y_pred = qy
            clf_loss = 0

        # Metric embedding loss: lifted structured loss

        metric_loss = self.metric_loss(qz_mu, manner)

        # if e <= 20:
        #     w1, w2, w3, w4, w5 = 1, 0.1, 0.1, 0.1, 0.1
        # elif e > 20:
        #     w1, w2, w3, w4, w5 = 1, 50, 50, 1, 100
        w1, w2, w3, w4, w5 = self.w1, self.w2, self.w3, self.w4, self.w5

        # Total loss
        if self.supervised:
            total_loss = clf_loss
        else:
            total_loss = (
                w1 * rec_loss + w2 * gaussian_loss + w3 * cat_loss + w5 * metric_loss
            )

        return (
            total_loss,
            rec_loss,
            gaussian_loss,
            cat_loss,
            clf_loss,
            metric_loss,
            x,
            x_rec,
            qy,
            y_pred,
        )


# Borrowed from https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Layers.py
class GumbelSoftmax(torch.nn.Module):
    def __init__(self, f_dim, c_dim, device=None):
        super(GumbelSoftmax, self).__init__()
        self.logits = torch.nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
        self.device = device

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.to(self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return torch.nn.functional.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        # categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


class Spectrogram_networks_manner(torch.nn.Module):
    def __init__(self, x_dim, hidden_dims, cnn=False):
        super().__init__()

        self.x_dim = x_dim
        self.hidden_dims = hidden_dims
        self.cnn = cnn

        # ===== Inference =====
        # Deterministic x_hat = g(x)
        if cnn:
            print(self.x_dim)
            self.inference_gx = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.x_dim,
                    self.hidden_dims[0],
                    kernel_size=[3, 3],
                    padding=[0, 1],
                    stride=[2, 1],
                ),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(self.hidden_dims[0]),
                torch.nn.Conv2d(
                    self.hidden_dims[0],
                    self.hidden_dims[1],
                    kernel_size=[3, 3],
                    padding=[0, 1],
                    stride=[2, 1],
                ),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(self.hidden_dims[1]),
                torch.nn.Conv2d(
                    self.hidden_dims[1],
                    self.hidden_dims[2],
                    kernel_size=[3, 3],
                    padding=[0, 1],
                    stride=[2, 1],
                ),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(self.hidden_dims[2]),
            )
            self.flatten = Flatten()
            self.x_hat_shape_before_flatten = self.inference_gx(
                torch.zeros(1, 1, 65, 33)
            ).shape
            self.x_hat_shape_after_flatten = self.flatten(
                self.inference_gx(torch.zeros(1, 1, 65, 33))
            ).shape
        else:
            raise NotImplementedError

        # ===== Generative =====
        # Deterministic x = f(x_hat)
        if cnn:
            self.unflatten = UnFlatten()
            self.generative_fxhat = torch.nn.Sequential(
                # ConvTranspose
                torch.nn.ConvTranspose2d(
                    self.hidden_dims[0],
                    self.hidden_dims[1],
                    kernel_size=[5, 3],
                    padding=[0, 1],
                    stride=[2, 1],
                ),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(self.hidden_dims[1]),
                # ConvTranspose
                torch.nn.ConvTranspose2d(
                    self.hidden_dims[1],
                    self.hidden_dims[2],
                    kernel_size=[5, 3],
                    padding=[0, 1],
                    stride=[2, 1],
                ),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(self.hidden_dims[2]),
                # ConvTranspose
                torch.nn.ConvTranspose2d(
                    self.hidden_dims[2],
                    self.x_dim,
                    stride=[2, 1],
                    kernel_size=[3, 3],
                    padding=[5, 1],
                ),
            )
        else:
            raise NotImplementedError


class Flatten(torch.nn.Module):
    def forward(self, x):
        # x_hat is shaped as (B, C, H, W), lets conver it to (B, W, C, H)
        x = x.permute(0, 3, 1, 2)

        # Flatten the two first dimensions so now is (B*W, C, H)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

        # Flatten now the last two dimension so is (B*W, C*H)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

        return x


class ClassifierFlatten(torch.nn.Module):
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)

        return x


class UnFlatten(torch.nn.Module):
    def forward(self, x):
        # x_hat is shaped as (B*W, C*H), lets conver it to (B*W, C, H)
        x = x.reshape(-1, 64, 7)

        # x is now shaped (B*W, C, H), lets conver it to (B, W, C, H)
        x = x.reshape(-1, 33, 64, 7)

        # x is now shaped (B, W, C, H), lets conver it to (B, C, H, W)
        x = x.permute(0, 2, 3, 1)

        return x
