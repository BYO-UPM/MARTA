import torch
import torch.nn.functional as F

class HingeLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        hinge_loss = self.margin - target * output
        hinge_loss = torch.clamp(hinge_loss, min=0.0)
        return torch.mean(hinge_loss)

class SimCLR(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(SimCLR, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float(),
        )

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = torch.nn.functional.normalize(emb_i, dim=1)
        z_j = torch.nn.functional.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.nn.functional.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature
        )

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


# Generalized End-to-End Loss
class GE2E_Loss(torch.nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        super(GE2E_Loss, self).__init__()
        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ["softmax", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    def Estimate_cosine_sim(self, e, y):
        batch = e.shape[0]
        classes = torch.unique(y)
        centroids = []
        n_samples = []
        S = torch.zeros(batch, classes.shape[0])
        for i in classes:
            centroids.append(torch.sum(e[y == i, :], dim=0))
            n_samples.append(e[y == i, :].shape[0])

        if 0 in n_samples:
            raise ("Not enough number of samples per class included in training batch")
        for i in range(batch):
            for k in classes:
                if k == y[i]:
                    S[i, k] = F.cosine_similarity(
                        torch.unsqueeze(e[i, :], 0),
                        torch.unsqueeze(
                            (centroids[k] - e[i, :]) / (n_samples[k] - 1), 0
                        ),
                    )
                else:
                    S[i, k] = F.cosine_similarity(
                        torch.unsqueeze(e[i, :], 0),
                        torch.unsqueeze(centroids[k] / n_samples[k], 0),
                    )
        return S

    def embed_loss_contrast(self, cos_sim_matrix, y):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N = cos_sim_matrix.shape[0]
        sig_cos_sim_matrix = torch.sigmoid(cos_sim_matrix)
        L = []
        for j in range(N):
            centroids_sigmoids = sig_cos_sim_matrix[j, y[j]]
            excl_centroids_sigmoids = torch.cat(
                (sig_cos_sim_matrix[j, : y[j]], sig_cos_sim_matrix[j, y[j] + 1 :])
            )
            L.append(1.0 - centroids_sigmoids + torch.max(excl_centroids_sigmoids))
        return torch.stack(L)

    def embed_loss_softmax(self, cos_sim_matrix, y):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N = cos_sim_matrix.shape[0]
        soft_cos_sim_matrix = -F.log_softmax(cos_sim_matrix, dim=0)
        L = []
        for j in range(N):
            L.append(soft_cos_sim_matrix[j, y[j]])
        return torch.stack(L)

    def forward(self, emb, y):
        # normalization
        e = F.normalize(emb, p=2, dim=1)

        Sim = self.Estimate_cosine_sim(e, y)

        Sim = Sim * self.w + self.b

        L = self.embed_loss(Sim, y)

        return L.sum()
