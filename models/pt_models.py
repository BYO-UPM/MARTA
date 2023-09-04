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

        # Inference
        self.inference_networks()

        # Generative
        self.generative_networks()

        self.w1, self.w2, self.w3, self.w4, self.w5 = weights

        self.to(self.device)

    def inference_networks(self):
        # ===== Inference =====
        # Deterministic x_hat = g(x)
        self.inference_gx = torch.nn.Sequential(
            torch.nn.Linear(self.x_dim, self.hidden_dims[0]),
        ).to(self.device)

        # q(y | x_hat)
        self.inference_qy_x = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0], self.k),
        ).to(self.device)

        # q(z | x_hat, y)
        self.inference_qz_xy = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0] + self.k, self.hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[1], self.hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[1], self.z_dim * 2),
        ).to(self.device)

    def generative_networks(self):
        # ===== Generative =====
        # p(x | z)
        self.generative_px_z = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, self.hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[1], self.x_dim),
        ).to(self.device)

        # p(z | y)
        self.generative_pz_y = torch.nn.Sequential(
            torch.nn.Linear(self.k, self.z_dim * 2),
        ).to(self.device)

    def bincount_matrix(self, x, num_classes):
        x = x.int()
        max_x_plus_1 = num_classes
        ids = x + max_x_plus_1 * torch.arange(x.shape[0]).unsqueeze(1).to(self.device)
        flattened_ids = ids.view(-1)
        out = torch.bincount(flattened_ids, minlength=max_x_plus_1 * x.shape[0])
        out = out.view(-1, num_classes)
        return out

    def assign_labels_semisupervised(
        self, features, labels, batch_size, num_classes, knn
    ):
        """Assign labels to unlabeled data based on the k-nearest-neighbors.

        Code adapted from https://github.com/jariasf/semisupervised-vae-metric-embedding/blob/master/utils/assignment.py

        Args:
            features: (array) corresponding array containing the features of the input data
            labels: (array) corresponding array containing the labels of the labeled data
            batch_size: (int) training batch size
            num_classes: (int) number fo classification classes
            knn: (int) number of k-nearest neighbors to use

        Returns:
            output: (array) corresponding array containing the labels assigned to all the data
        """
        num_classes = torch.Tensor([num_classes]).type(torch.int64).to(self.device)
        self.ss_mask = torch.ones(labels.shape).to(self.device)
        self.ss_mask[-int(self.ss_mask.shape[0] * 0.1) :] = 0
        self.num_labeled = int(torch.sum(self.ss_mask))

        # Make dot product between all features and then get only the daigonal.
        dot_product = torch.mm(features, features.t())
        square_norm = torch.diag(dot_product)
        # compute pairwise distance matrix
        distances = (
            square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
        )
        # Clamp it to 0 in case of numerical error
        distances = torch.clamp(distances, min=0.0)
        # Get the mask of the labeled data and the unlabeled data
        distances = distances[
            : batch_size - self.num_labeled, batch_size - self.num_labeled :
        ]
        neg_distances = -distances
        # Get the top K nearest neighbors
        _, idx = torch.topk(neg_distances, knn, dim=1)
        # Get the labels of the nearest neighbors
        knn_labels = labels[idx].type(torch.int64)
        # count repeated labels and get the most common one
        assignment = torch.argmax(
            torch.bincount(knn_labels.view(-1), minlength=num_classes), dim=0
        )
        # Concatenate the assignment with the labels of the labeled data
        output_labels = torch.cat([labels[: self.num_labeled], assignment])
        return output_labels

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def infere(self, x: torch.Tensor) -> torch.Tensor:
        # x_hat = g(x)
        x_hat = self.inference_gx(x)

        # q(y | x)
        qy_logits = self.inference_qy_x(x_hat)
        # gumbel softmax (if hard=True, it becomes one-hot vector, otherwise soft)
        qy = torch.nn.functional.gumbel_softmax(qy_logits, tau=1.0, hard=True)

        # q(z | x, y)
        # concat x and y
        xy = torch.cat([x_hat, qy], dim=1)
        qz_mu, qz_logvar = torch.chunk(self.inference_qz_xy(xy), 2, dim=1)

        z = self.reparametrize(qz_mu, qz_logvar)

        return z, qy_logits, qy, qz_mu, qz_logvar

    def log_normal(self, x, mu, var):
        return -0.5 * (
            torch.log(var) + torch.log(2 * torch.tensor(torch.pi)) + (x - mu) ** 2 / var
        )

    def generate(self, z, y) -> torch.Tensor:
        # p(z | y)
        z_mu, z_logvar = torch.chunk(self.generative_pz_y(y), 2, dim=1)
        z_var = torch.nn.functional.softplus(z_logvar)

        # p(x | z)
        x_rec = self.generative_px_z(z)

        return x_rec, z_mu, z_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, qy_logits, qy, qz_mu, qz_logvar = self.infere(x)
        x_rec, z_mu, z_var = self.generate(z, qy)

        return x_rec, z_mu, z_var, qy_logits, qy, qz_mu, qz_logvar, z

    def loss(self, x: torch.Tensor, labels=None, combined=None) -> torch.Tensor:
        x_rec, z_mu, z_var, qy_logits, qy, qz_mu, qz_logvar, z = self.forward(x)

        # reconstruction loss
        rec_loss = torch.nn.functional.mse_loss(x_rec, x, reduction="sum")

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

        if self.ss:
            import pytorch_metric_learning

            # Semi-supervised loss
            # Assign labels to unlabeled data based on the k-nearest-neighbors
            labels = self.assign_labels_semisupervised(
                z, labels, batch_size=x.shape[0], num_classes=self.k, knn=10
            )
            # Cross entropy loss
            clf_loss = torch.nn.functional.cross_entropy(qy_logits, labels)

            # Metric embedding loss: lifted structured loss
            lifted_loss = pytorch_metric_learning.losses.LiftedStructureLoss(
                pos_margin=0.5, neg_margin=0.5
            )
            metric_loss = lifted_loss(z, labels)
        else:
            clf_loss = 0
            metric_loss = 0

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
            metric_loss = 0

        # Total loss
        total_loss = (
            self.w1 * rec_loss
            + self.w2 * gaussian_loss
            + self.w3 * cat_loss
            + self.w4 * clf_loss
            + self.w5 * metric_loss
        )

        # obtain predictions
        _, y_pred = torch.max(qy_logits, dim=-1)

        return total_loss, rec_loss, gaussian_loss, cat_loss, clf_loss, x, x_rec, y_pred
