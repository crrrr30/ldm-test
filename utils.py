from tqdm.notebook import tqdm
from inspect import isfunction
from prettytable import PrettyTable

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def count_parameters(model, model_name=None):
    table = PrettyTable(["Modules", "#Parameters"]);
    table.align["Modules"] = 'l'
    table.align["#Parameters"] = 'r'
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, f"{params:,}"])
        total_params+=params
    print(table)
    if model_name:
        print(f"===> Total Trainable Params: {total_params:,}.")
    else:
        print(f"===> Total {model_name} Trainable Params: {total_params:,}.")


# DDPM Misc Utils

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_hyperparams(timesteps, device, schedule="cosine"):
    if schedule == "cosine":
        betas = cosine_beta_schedule(timesteps=timesteps).to(device)
    elif schedule == "linear":
        betas = linear_beta_schedule(timesteps=timesteps).to(device)

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        "betas": betas, "alphas": alphas, "alphas_cumprod": alphas_cumprod, \
            "alphas_cumprod_prev": alphas_cumprod_prev, "sqrt_recip_alphas": sqrt_recip_alphas, \
                "sqrt_alphas_cumprod": sqrt_alphas_cumprod, "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod, \
                    "posterior_variance": posterior_variance, "timesteps": timesteps
    }


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Forward Diffusion
def q_sample(hyperparams, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(hyperparams["sqrt_alphas_cumprod"], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        hyperparams["sqrt_one_minus_alphas_cumprod"], t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(hyperparams, denoise_model, x_start, t, loss_type="l1"):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(hyperparams, x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss

@torch.no_grad()
def p_sample(hyperparams, model, x, t, t_index):
    betas_t = extract(hyperparams["betas"], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        hyperparams["sqrt_one_minus_alphas_cumprod"], t, x.shape
    )
    sqrt_recip_alphas_t = extract(hyperparams["sqrt_recip_alphas"], t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(hyperparams["posterior_variance"], t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(hyperparams, model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in tqdm(reversed(range(0, hyperparams["timesteps"])), desc='Sampling loop time step', total=hyperparams["timesteps"]):
        img = p_sample(hyperparams, model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop_final(hyperparams, model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(0, hyperparams["timesteps"])), desc='Sampling loop time step', total=hyperparams["timesteps"]):
        img = p_sample(hyperparams, model, img, torch.full((b,), i, device=device, dtype=torch.long), i).clamp_(-1., 1.)
    return img

@torch.no_grad()
def sample(hyperparams, model, image_size, batch_size, channels=3):
    return p_sample_loop_final(hyperparams, model, shape=(batch_size, channels, image_size, image_size))

def sample_and_save(hyperparams, model, vae, iteration, image_size, num=16):
    batch_size = 2
    assert num % batch_size == 0, f"Batch size {batch_size} must divide number of samples {num}."
    all_images_list = [sample(hyperparams, model.cuda(), image_size, batch_size=batch_size, channels=4) for _ in range(num // batch_size)]
    all_images = torch.cat(all_images_list, dim=0)
    all_images = vae.cuda().decode(all_images / 0.18215).sample
    all_images = (all_images + 1) * 0.5
    save_image(all_images, f"sample-{iteration:06}.png")
    