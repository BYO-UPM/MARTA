import torch
import timm
import numpy as np
import copy
import time
from utils.utils import log_normal, KL_cat
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

    def __init__(
        self,
        x_dim,
        z_dim=32,
        hidden_dims_spectrogram=[64, 1024, 64],
        hidden_dims_gmvae=[256],
        n_gaussians=10,
        n_manner=8,
        weights=[1, 1, 1, 10],
        classifier="mlp",
        device="cpu",
        reducer="mean",
    ):
        super().__init__()
        # ============ Device ============
        self.device = device
        # ============ GMVAE parameters ============
        self.k = n_gaussians
        self.w = weights
        self.manner = n_manner
        self.z_dim = z_dim
        self.x_dim = x_dim[0]
        self.window_size = x_dim[-1]
        self.mel_bins = x_dim[-2]
        self.class_dims = [64, 32, 16]
        self.classifier_type = classifier
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

        # ============ Losses ============
        self.reducer = reducer
        # Reconstruction loss
        self.mse_loss = torch.nn.MSELoss(reduction=self.reducer)
        # BCE loss
        self.bce_loss = torch.nn.BCELoss(reduction=self.reducer)

        # ============ Initialize the networks ============
        self.init_weights()

        # ============ Move the networks to the device ============
        self.to(self.device)

    def init_weights(self):
        """Initialize the weights of the networks."""
        # Initialize the weights of the networks
        for m in self.modules():
            if (
                type(m) == torch.nn.Linear
                or type(m) == torch.nn.Conv2d
                or type(m) == torch.nn.ConvTranspose2d
            ):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    torch.nn.init.constant_(m.bias, 0)

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
            torch.nn.BatchNorm2d(self.hidden_dims_spectrogram[0]),
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

        self.x_hat_shape_after_flat = self.spec_enc(
            torch.zeros(1, 1, self.mel_bins, self.window_size)
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
            torch.nn.Linear(self.x_hat_shape_after_flat[1], self.hidden_dims_gmvae[0]),
            torch.nn.ReLU(),  # Check if makes sense to have a ReLU here
            GumbelSoftmax(self.hidden_dims_gmvae[0], self.k, device=self.device),
        ).to(self.device)

        # y_hat = h(y): the output of the GumbelSoftmax is the probability of each gaussian component given the encoded spectrogram (e_s).
        # This network will upsample the K components to match the dimensionality of the encoded spectrogram (e_s), i.e., N*33, Channels*Height.
        self.hy = torch.nn.Linear(self.k, self.x_hat_shape_after_flat[1]).to(
            self.device
        )

        # q(z | e_s, y): the GaussianMixture network will get the encoded spectrogram (e_s) and the GumbelSoftmax output (y) and will output the parameters of the GaussianMixture, i.e., mu and logvar.
        self.qz_es_y = torch.nn.Sequential(
            torch.nn.Linear(self.x_hat_shape_after_flat[1], self.hidden_dims_gmvae[0]),
            torch.nn.ReLU(),  # Check if makes sense to have a ReLU here
            torch.nn.Linear(
                self.hidden_dims_gmvae[0], self.z_dim * 2
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

    def classifier(self):
        """This function adds a classifier to the network.
        First, the manner_class (N x 33 x 1) are embedded to a z_dim vector (N x win_size x z_dim).
        Then, two options are possible:
        1. Use a CNN classifier whose input is: z + h(manner_class) (N x win_size x z_dim)
        2. Use a MLP classifier whose input is: z + h(manner_class) flattened (N x (win_size*z_dim))
        """
        self.hmc = torch.nn.Embedding(self.manner, self.z_dim)

        self.clf_cnn = torch.nn.Sequential(
            # 2DConv with kernel_size = (3, 32), stride=2, padding=1
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
            torch.nn.Linear(self.z_dim * self.window_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def spec_encoder_forward(self, x):
        """Forward function of the spectrogram encoder network. It receives the spectrogram (x) and outputs the encoded spectrogram (e_s)."""
        e_s = self.spec_enc(x)
        return e_s

    def inference_forward(self, e_s):
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
        x_hat = self.spec_dec(e_hat_s)
        return x_hat

    def classifier_forward(self, x, manner):
        """Forward function of the classifier network. It receives the spectrogram (x), the manner class of each phoneme (manner) and the labels (labels).
        It outputs the predictions (y_pred) and the labels (labels)."""
        # Extract the features of the spectrogram
        e_s = self.spec_encoder_forward(x)
        # Get the latent space representation of each window of the spectrogram
        _, _, _, qz_mu, _, _, _ = self.inference_forward(e_s)
        # Let us use the mean of the latent space representation as the samples
        z = qz_mu

        # Embed manner class from (batch, win_size) to (batch, win_size, z_dim)
        hmc = self.hmc(manner)

        # Now z is (batch*window, z_dim), reshape to: (batch, window_size, z_dim)
        z = z.reshape(hmc.shape[0], self.window_size, self.z_dim)

        if self.classifier_type == "mlp":
            # For the MLP classifier, flatten both z and hmc from (batch, window, z_dim) to (batch, window*z_dim)
            hmc = hmc.reshape(hmc.shape[0], -1)
            z = z.reshape(z.shape[0], -1)

        # \tilde{z} = z + h(mc)
        if self.classifier_type == "cnn":
            # Sum them and reshape to (batch, 1, window_size, z_dim) for the CNN
            z_hat = (z + hmc).unsqueeze(1)
        else:
            # Sum them, now the shape is (batch, window_size*z_dim)
            z_hat = z + hmc

        if self.classifier_type == "cnn":
            y_pred = self.clf_cnn(z_hat)
        else:
            y_pred = self.clf_mlp(z_hat)

        return y_pred

    def forward(self, x):
        """Forward function of the SpeechTherapist. It receives the spectrogram (x) and outputs the spectrogram reconstruction (x_hat)."""
        e_s = self.spec_encoder_forward(x)
        z_sample, y_logits, y, qz_mu, qz_var, pz_mu, pz_var = self.inference_forward(
            e_s
        )
        e_hat_s = self.generative_forward(z_sample)
        x_hat = self.spec_dec_forward(e_hat_s)
        return (
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
        )

    def metric_loss(self, z_mu, manner_classes):
        """This function computes the metric loss of the GMVAE. It receives the latent space representation of each window of the spectrogram (z_mu) and the manner class of each phoneme (manner_classes).
        Currently we are ignoring the silences and affricates as they are minor in the datase, but we should find a way to include them in the metric loss.
        """
        if self.reducer == "mean":
            reducer = reducers.MeanReducer()
        else:
            reducer = reducers.SumReducer()

        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self.metric_loss_func = losses.GeneralizedLiftedStructureLoss(
            reducer=reducer
        ).to(self.device)

        # Silences and affricates should be not used to compute the metric loss
        manner_classes = manner_classes.reshape(-1)
        # idx = np.where((manner_classes != 6) & (manner_classes != 7))[
        #     0
        # ]  # 6: affricates, 7: silences
        # z_mu = z_mu[idx]
        # manner_classes = manner_classes[idx]

        # Oversample labels and x_hat to have the same number of samples
        # unique_labels, count_labels = np.unique(manner_classes, return_counts=True)
        # max_count = np.max(count_labels)

        # # Generate indices for oversampling
        # idx_sampled = np.concatenate(
        #     [
        #         np.random.choice(np.where(manner_classes == label)[0], max_count)
        #         for label in unique_labels
        #     ]
        # )

        # # Apply oversampling
        # manner_classes = manner_classes[idx_sampled]
        # z_mu = z_mu[idx_sampled]

        # Miner
        hard_pairs = self.miner(z_mu, manner_classes)

        # Loss
        loss = self.metric_loss_func(z_mu, manner_classes, hard_pairs)

        return loss

    def loss(
        self,
        x,
        manner_classes,
        x_hat,
        z_sample,
        qz_mu,
        qz_var,
        pz_mu,
        pz_var,
        y,
        y_logits,
    ):
        # Reconstruction loss
        recon_loss = self.mse_loss(x_hat, x)

        # Gaussian loss
        gaussian_loss = log_normal(
            z_sample, qz_mu, qz_var
        ) - log_normal(  # q(z | e_s, y):
            z_sample, pz_mu, pz_var
        )  # p(z | y)
        if self.reducer == "mean":
            gaussian_loss = torch.mean(gaussian_loss)
        else:
            gaussian_loss = torch.sum(gaussian_loss)

        # Categorical loss
        categorical_loss = KL_cat(y, y_logits, self.k, self.reducer)

        # Metric loss
        metric_loss = self.metric_loss(qz_mu, manner_classes)

        # Total loss is the weighted sum of the four losses
        complete_loss = (
            self.w[0] * recon_loss
            + self.w[1] * gaussian_loss
            + self.w[2] * categorical_loss
            + self.w[3] * metric_loss
        )

        return complete_loss, recon_loss, gaussian_loss, categorical_loss, metric_loss

    def classifier_loss(self, y_pred, labels):
        """This function computes the loss of the classifier. It receives the predictions (y_pred) and the labels (labels)."""
        # Convert both to N, 1 shape if they are not
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(1)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        # BCE loss
        loss = self.bce_loss(y_pred, labels)
        return loss, 0, 0, 0, 0


# ========================================= START of utils for SpeechTherapist =========================================
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


class Flatten(torch.nn.Module):
    def forward(self, x):
        # x_hat is shaped as (B, C, H, W), lets conver it to (B, W, C, H)
        x = x.permute(0, 3, 1, 2)

        # Flatten the two first dimensions so now is (B*W, C, H)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

        # Flatten now the last two dimension so is (B*W, C*H)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

        return x


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


# ========================================= END of util for SpeechTherapist =========================================
