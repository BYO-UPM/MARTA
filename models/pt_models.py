import torch
import timm
import copy


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


class VAE(torch.nn.Module):
    def __init__(
        self,
        embedding_input,
        latent_dim=2,
        hidden_dims_enc=[20, 10],
        hidden_dims_dec=[20],
        supervised=False,
    ):
        super(VAE, self).__init__()
        self.embedding_input = embedding_input
        self.latent_dim = copy.deepcopy(latent_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            self.dec_sup = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, 5),
                torch.nn.Sigmoid(),
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
        y_hat = self.dec_sup(mu)
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
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        self.K = embedding_dim
        self.D = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, continous_latents: torch.Tensor) -> torch.Tensor:
        latents_shape = continous_latents.shape
        flat_latents = continous_latents.view(-1, self.D)

        # Calclate L2 distance between latents and embedding weights
        dist = torch.cdist(flat_latents, self.embedding.weight)

        # Get the index of the closest embedding
        enc_idx = torch.argmin(dist, dim=1)

        # Convert to one-hot encodings
        encoding_ohe = torch.nn.functional.one_hot(enc_idx, self.K).type_as(
            flat_latents
        )

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_ohe, self.embedding.weight).view(
            latents_shape
        )

        # Compute VQ loss
        commitment_loss = torch.nn.functional.mse_loss(
            quantized_latents.detach(), continous_latents
        )
        embedding_loss = torch.nn.functional.mse_loss(
            quantized_latents, continous_latents.detach()
        )

        # Loss
        vq_loss = embedding_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized_latents = (
            continous_latents + (quantized_latents - continous_latents).detach()
        )

        return quantized_latents, vq_loss


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
        self.K = K
        self.input_dim = copy.deepcopy(input_dim)
        self.commitment_cost = commitment_cost
        self.supervised = supervised
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        encoder_layers = []
        for i in range(len(hidden_dims_enc)):
            encoder_layers.append(torch.nn.Linear(input_dim, hidden_dims_enc[i]))
            encoder_layers.append(torch.nn.ReLU())
            input_dim = hidden_dims_enc[i]

        self.enc = torch.nn.Sequential(*encoder_layers)

        self.vq = VectorQuantizer(self.K, self.latent_dim, self.commitment_cost)

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
                torch.nn.Linear(self.latent_dim, 1),
                torch.nn.Sigmoid(),
            )

        self.to(self.device)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec(z)
        if self.supervised:
            y = self.clf(z)
        else:
            y = None
        return x, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)
        x_hat, y_hat = self.decoder(z_q)
        return x_hat, y_hat, vq_loss


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
