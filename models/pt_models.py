import torch
import timm
import numpy as np
import copy
from pytorch_metric_learning.losses import LiftedStructureLoss


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


class VAE_images(torch.nn.Module):
    def __init__(
        self,
        embedding_input,
        latent_dim=2,
        hidden_dims_enc=[20, 10],
        hidden_dims_dec=[20],
        supervised=False,
        n_classes=5,
    ):
        super(VAE, self).__init__()
        self.embedding_input = embedding_input[0]
        self.latent_dim = copy.deepcopy(latent_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes

        encoder_layers = []
        for i in range(len(hidden_dims_enc)):
            # CNN layers
            encoder_layers.append(
                torch.nn.Conv2d(embedding_input, hidden_dims_enc[i], 3)
            )
            encoder_layers.append(torch.nn.ReLU())
            encoder_layers.append(torch.nn.MaxPool2d(2))
            embedding_input = hidden_dims_enc[i]
        encoder_layers.append(torch.nn.Flatten())

        self.enc = torch.nn.Sequential(*encoder_layers)
        self.output_shape = self.enc(torch.zeros(1, *embedding_input)).shape

        self.fc_mu = torch.nn.Linear(hidden_dims_enc[-1], self.latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dims_enc[-1], self.latent_dim)

        decoder_layers = []
        for i in range(len(hidden_dims_dec)):
            # CNN layers
            if i == 0:
                decoder_layers.append(
                    torch.nn.Linear(self.latent_dim, hidden_dims_dec[i])
                )
            decoder_layers.append(
                torch.nn.ConvTranspose2d(hidden_dims_dec[i], hidden_dims_dec[i], 3)
            )
            decoder_layers.append(torch.nn.ReLU())
            decoder_layers.append(torch.nn.Upsample(scale_factor=2))
            latent_dim = hidden_dims_dec[i]
        self.dec = torch.nn.Sequential(
            *decoder_layers,
            torch.nn.ConvTranspose2d(hidden_dims_dec[-1], self.embedding_input),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Sigmoid(),
        )
        self.supervised = supervised
        if self.supervised:
            if self.n_classes == 2:
                self.dec_sup = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(5, 1),
                    torch.nn.Sigmoid(),
                )
            else:
                self.dec_sup = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(5, n_classes),
                )

        self.to(self.device)

    def encoder(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h) + 1e-6  # logvar > 0
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        return self.dec(z)

    def decoder_supervised(self, z, mu):
        x_hat = self.dec(z)
        y_hat = self.dec_sup(
            z
        )  # Works better with z than with mu, TODO: develop this by formula
        return x_hat, y_hat

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if self.supervised:
            x_hat, y_hat = self.decoder_supervised(z, mu)
            return x_hat, y_hat, mu, logvar
        else:
            x_hat = self.decoder(z)
            return x_hat, mu, logvar


class VAE(torch.nn.Module):
    def __init__(
        self,
        embedding_input,
        latent_dim=2,
        hidden_dims_enc=[20, 10],
        hidden_dims_dec=[20],
        supervised=False,
        n_classes=5,
    ):
        super(VAE, self).__init__()
        self.embedding_input = embedding_input
        self.latent_dim = copy.deepcopy(latent_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes

        encoder_layers = []
        for i in range(len(hidden_dims_enc)):
            encoder_layers.append(torch.nn.Linear(embedding_input, hidden_dims_enc[i]))
            encoder_layers.append(torch.nn.ReLU())
            embedding_input = hidden_dims_enc[i]

        self.enc = torch.nn.Sequential(*encoder_layers)

        self.fc_mu = torch.nn.Linear(hidden_dims_enc[-1], self.latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dims_enc[-1], self.latent_dim)

        decoder_layers = []
        for i in range(len(hidden_dims_dec)):
            decoder_layers.append(torch.nn.Linear(latent_dim, hidden_dims_dec[i]))
            decoder_layers.append(torch.nn.ReLU())
            latent_dim = hidden_dims_dec[i]
        self.dec = torch.nn.Sequential(
            *decoder_layers,
            torch.nn.Linear(hidden_dims_dec[-1], self.embedding_input),
        )
        self.supervised = supervised
        if self.supervised:
            if self.n_classes == 2:
                self.dec_sup = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(5, 1),
                    torch.nn.Sigmoid(),
                )
            else:
                self.dec_sup = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(5, n_classes),
                )

        self.to(self.device)

    def encoder(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h) + 1e-6  # logvar > 0
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        return self.dec(z)

    def decoder_supervised(self, z, mu):
        x_hat = self.dec(z)
        y_hat = self.dec_sup(
            z
        )  # Works better with z than with mu, TODO: develop this by formula
        return x_hat, y_hat

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if self.supervised:
            x_hat, y_hat = self.decoder_supervised(z, mu)
            return x_hat, y_hat, mu, logvar
        else:
            x_hat = self.decoder(z)
            return x_hat, mu, logvar


class VectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        usage_threshold: float,
    ):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.commitment_cost = commitment_cost
        self.usage_threshold = usage_threshold

        self.embedding = torch.nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        self.register_buffer("cluster_size", torch.zeros(self.K))
        self.register_buffer("embed_avg", torch.zeros(self.K, self.D))
        self.register_buffer("usage", torch.ones(self.K), persistent=False)

    def random_restart(self):
        # Get dead embeddings
        dead_embeddings = torch.where(self.embed_usage < self.usage_threshold)[0]
        # Reset dead embeddings
        rand_codes = torch.randperm(self.K)[: len(dead_embeddings)]
        with torch.no_grad():
            self.embedding.weight[dead_embeddings] = self.embedding.weight[rand_codes]
            self.embed_usage[dead_embeddings] = self.embed_usage[rand_codes]
            self.cluster_size[dead_embeddings] = self.cluster_size[rand_codes]
            self.embed_avg[dead_embeddings] = self.embed_avg[rand_codes]

    def reset_usage(self):
        self.embed_usage = torch.zeros(self.K)
        self.usage.zero_()  #  reset usage between epochs

    def update_usage(self, min_enc):
        self.usage[min_enc] = self.usage[min_enc] + 1  # if code is used add 1 to usage

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        latents_shape = z.shape
        # Flatten batch inputs
        flat_latents = z.view(-1, self.D)

        # Calclate L2 distance between latents and embedding weights
        dist = torch.cdist(flat_latents, self.embedding.weight)

        # Get the index of the closest embedding
        enc_idx = torch.argmin(dist, dim=1)

        # Convert to one-hot encodings
        encoding_ohe = torch.nn.functional.one_hot(enc_idx, self.K).type_as(
            flat_latents
        )

        # Quantize the latents
        z_q = torch.matmul(encoding_ohe, self.embedding.weight).view(latents_shape)

        # Update usage
        self.update_usage(enc_idx)

        # Compute VQ loss
        e_loss = torch.nn.functional.mse_loss(z_q.detach(), z, reduction="sum")
        q_loss = torch.nn.functional.mse_loss(z_q, z.detach(), reduction="sum")

        # Loss
        vq_loss = q_loss + self.commitment_cost * e_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Calculate avg
        avg_probs = torch.mean(encoding_ohe, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, vq_loss, enc_idx


class VQVAE(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        K=128,
        hidden_dims_enc=[20, 10],
        hidden_dims_dec=[20],
        commitment_cost=0.25,
        supervised=False,
    ):
        super().__init__()
        self.latent_dim = copy.deepcopy(latent_dim)
        self.K = copy.deepcopy(K)
        self.input_dim = copy.deepcopy(input_dim)
        self.commitment_cost = commitment_cost
        self.supervised = supervised
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.usage = torch.ones(self.K).to(self.device)

        encoder_layers = []
        for i in range(len(hidden_dims_enc)):
            encoder_layers.append(torch.nn.Linear(input_dim, hidden_dims_enc[i]))
            encoder_layers.append(torch.nn.ReLU())
            input_dim = hidden_dims_enc[i]

        self.enc = torch.nn.Sequential(*encoder_layers)
        self.fc_mu = torch.nn.Linear(hidden_dims_enc[-1], self.latent_dim)

        self.vq = VectorQuantizer(
            self.K, self.latent_dim, self.commitment_cost, usage_threshold=1e-3
        )

        latent_dim = self.latent_dim
        decoder_layers = []
        for i in range(len(hidden_dims_dec)):
            decoder_layers.append(torch.nn.Linear(latent_dim, hidden_dims_dec[i]))
            decoder_layers.append(torch.nn.ReLU())
            latent_dim = hidden_dims_dec[i]
        self.dec = torch.nn.Sequential(
            *decoder_layers,
            torch.nn.Linear(hidden_dims_dec[-1], self.input_dim),
        )

        if self.supervised:
            self.clf = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1),
                torch.nn.Sigmoid(),
            )

        self.to(self.device)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        mu = self.fc_mu(z)
        return mu

    def reset_usage(self):
        self.vq.reset_usage()

    def decoder(self, z_q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.dec(z_q)  # Decode with quantized latents
        if self.supervised:
            y = self.clf(z)  # Predict class from quantized latents
        else:
            y = None
        return x, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_q, vq_loss, enc_idx = self.vq(z)
        x_hat, y_hat = self.decoder(z_q, z)
        self.usage = self.vq.usage
        return x_hat, y_hat, vq_loss, z, z_q, enc_idx


class Spectrogram_networks(torch.nn.Module):
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
            self.x_hat_shape = self.inference_gx(torch.zeros(1, 1, 65, 41)).shape
        else:
            raise NotImplementedError

        # ===== Generative =====
        # Deterministic x = f(x_hat)
        if cnn:
            self.generative_fxhat= torch.nn.Sequential(
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


class GMVAE(torch.nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        n_gaussians=10,
        hidden_dims=[20, 10],
        ss=False,
        supervised=False,
        weights=[1, 3, 10, 10, 10],
        cnn=False,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.k = n_gaussians
        self.ss = ss
        self.num_labeled = 0
        self.ss_mask = None
        self.supervised = supervised
        self.hidden_dims = hidden_dims
        self.cnn = cnn
        self.usage = np.zeros(self.k)

        # Spectrogram networks
        self.spec_nets = Spectrogram_networks(x_dim=self.x_dim, hidden_dims=self.hidden_dims, cnn=cnn)

        # Inference
        self.inference_networks(cnn=cnn)

        # Generative
        self.generative_networks(cnn=cnn)

        self.w1, self.w2, self.w3, self.w4, self.w5 = weights

        # ===== Loss =====
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self.mse_loss = torch.nn.MSELoss(reduction="sum")
        self.lifted_struct_loss = LiftedStructureLoss(pos_margin=0, neg_margin=1)

        self.to(self.device)

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

    def inference_networks(self, cnn=False):
        # ===== Inference =====
        # Deterministic x_hat = g(x)
        self.inference_gx = self.spec_nets.inference_gx.to(self.device)
        self.x_hat_shape = self.spec_nets.x_hat_shape
        
        # q(y | x_hat)
        if cnn:
            self.inference_qy_x = torch.nn.Sequential(
                torch.nn.Linear(self.x_hat_shape[1], self.hidden_dims[1]),
            ).to(self.device)
        else:
            raise NotImplementedError
        
        # Gumbel softmax
        self.gumbel_softmax = torch.nn.Sequential(
            GumbelSoftmax(self.hidden_dims[1], self.k)
        )

        # q(z | x_hat, y)
        if cnn:
            self.inference_qz_xy = torch.nn.Sequential(
                torch.nn.Linear(self.x_hat_shape[1], self.z_dim * 2),
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
                # Unflatten
                torch.nn.Linear(self.z_dim, self.hidden_dims[0] * 3 * 3),
                torch.nn.ReLU(),
                torch.nn.Unflatten(1, (self.hidden_dims[0], 3, 3)),).to(self.device)
            
            self.generative_fxhat = self.spec_nets.generative_fxhat.to(self.device)
        else:
            raise NotImplementedError

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def infere(self, x: torch.Tensor) -> torch.Tensor:
        # x_hat = g(x)
        x_hat = self.inference_gx(x)

        # q(y | x)
        qy_logits, probs, qy = self.gumbel_softmax(self.inference_qy_x(x_hat))

        # q(z | x, y)
        # Transform Y in the same shape as X
        if self.cnn:
            y_hat = torch.nn.Linear(self.k, self.x_hat_shape[1]).to(self.device)(qy)
            xy = x_hat + y_hat
        else:
            y_hat = torch.nn.Linear(self.k, self.hidden_dims[2]).to(self.device)(qy)
            xy = x_hat + y_hat
        qz_mu, qz_logvar = torch.chunk(self.inference_qz_xy(xy), 2, dim=1)

        z = self.reparametrize(qz_mu, qz_logvar)

        return z, qy_logits, qy, qz_mu, qz_logvar, x_hat

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

        # x_rec = f(x_hat)
        x_rec = self.generative_fxhat(x_hat)

        return x_rec, z_mu, z_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, qy_logits, qy, qz_mu, qz_logvar, x_hat = self.infere(x)
        x_rec, z_mu, z_var = self.generate(z, qy)

        return x_rec, z_mu, z_var, qy_logits, qy, qz_mu, qz_logvar, z, x_hat

    def loss(self, x: torch.Tensor, labels=None, combined=None, e=0) -> torch.Tensor:
        x_rec, z_mu, z_var, qy_logits, qy, qz_mu, qz_logvar, z, x_hat = self.forward(x)

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
            if self.n_gaussians < 10:
                # Supervised loss
                clf_loss = torch.nn.CrossEntropyLoss(reduction="sum")(
                    qy_logits, labels.type(torch.int64)
                )
            else:
                # Supervised loss
                clf_loss = torch.nn.CrossEntropyLoss(reduction="sum")(
                    qy_logits, combined.type(torch.float32)
                )
        else:
            clf_loss = 0

        # Metric embedding loss: lifted structured loss
        metric_loss = self.lifted_struct_loss(x_hat, labels)

        # if e <= 20:
        #     w1, w2, w3, w4, w5 = 1, 0.1, 0.1, 0.1, 0.1
        # elif e > 20:
        #     w1, w2, w3, w4, w5 = 1, 50, 50, 1, 100
        w1, w2, w3, w4, w5 = self.w1, self.w2, self.w3, self.w4, self.w5

        # Total loss
        total_loss = (
            w1 * rec_loss
            + w2 * gaussian_loss
            + w3 * cat_loss
            + w4 * clf_loss
            + w5 * metric_loss
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
        )


# Borrowed from https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Layers.py
class GumbelSoftmax(torch.nn.Module):
    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = torch.nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
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
