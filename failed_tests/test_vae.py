import torch
from models.pt_models import VAE
from training.pt_training import VAE_trainer


# Unit test for the VAE
def test_vae():
    # Create a random input tensor
    x = torch.randn(1, 39)

    # Create a VAE model
    vae = VAE(
        embedding_input=39,
        hidden_dims_enc=[30, 30, 30, 30],
        hidden_dims_dec=[30, 30, 30, 30],
        latent_dim=2,
    )

    # Run the VAE model
    x_hat, mu, logvar = vae(x.to(vae.device))

    # Check the output shape
    assert x_hat.shape == x.shape

    # Check the mu shape
    assert mu.shape == (1, 2)

    # Check the logvar shape
    assert logvar.shape == (1, 2)


# Create a unitary test to check the VAE_trainer function
def test_VAE_trainer():
    # Create a VAE model
    vae = VAE(
        embedding_input=39,
        hidden_dims_enc=[30, 30, 30, 30],
        hidden_dims_dec=[30, 30, 30, 30],
        latent_dim=2,
    )

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=torch.randn(100, 39), batch_size=10, shuffle=True
    )

    # Create a valid dataloader
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=torch.randn(100, 39), batch_size=10, shuffle=True
    )

    # Run the VAE model
    (
        elbo_training,
        kl_div_training,
        rec_loss_training,
        elbo_validation,
        kl_div_validation,
        rec_loss_validation,
    ) = VAE_trainer(vae, dataloader, valid_dataloader, 10, 1, 0.001, False)

    print(elbo_training)

    # Check the output shape
    assert len(elbo_training) == 10

    # Check
    assert len(kl_div_training) == 10

    # Check
    assert len(rec_loss_training) == 10

    print("VAE_trainer test passed!")


if __name__ == "__main__":
    test_vae()
    test_VAE_trainer()
