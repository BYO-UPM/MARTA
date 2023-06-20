import torch
from models.pt_models import VAE


# Unit test for the VAE
def test_vae():
    # Create a random input tensor
    x = torch.randn(1, 39)

    # Create a VAE model
    vae = VAE(embedding_input=39)

    # Run the VAE model
    x_hat, mu, logvar = vae(x)

    # Check the output shape
    assert x_hat.shape == x.shape

    # Check the mu shape
    assert mu.shape == (1, 2)

    # Check the logvar shape
    assert logvar.shape == (1, 2)


if __name__ == "__main__":
    test_vae()
