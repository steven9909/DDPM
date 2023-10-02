import torch
import logging
import tqdm

class SimpleDiffusion:
    def __init__(
        self, img_size, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"
    ):
        self.img_size = img_size
        self.num_timesteps = num_timesteps
        self.device = device
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self._get_schedule()
        self.alpha = 1.0 - self.beta

        self.alpha_sqrt = torch.sqrt(self.alpha)
        self.alpha_cum = torch.cumprod(self.alpha, dim=0)
        self.alpha_sqrt_cum = torch.sqrt(self.alpha_cum)
        self.alpha_one_minus_sqrt_cum = torch.sqrt(1 - self.alpha_cum)

    def _get_schedule(self):
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
    
    def sample_timestamps(self, num_samples):
        return torch.randint(low = 1, high=self.num_timesteps, size=(num_samples,)).to(self.device)
    
    def noise_image(self, img, t):
        noise = torch.randn_like(img, device=self.device)
        return self.alpha_sqrt_cum[t] * img + self.alpha_one_minus_sqrt_cum[t] * noise, noise

    def sample(self, model, batch_size):
        logging.info(f"Sampling {batch_size} from model...")
        model.eval()

        with torch.no_grad():
            x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(
                self.device
            )
            
            for i in tqdm(reversed(range(1, self.num_timesteps)), position=0):
                if i > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                x = 1/self.alpha_sqrt[i] * (x - (1 - self.alpha[i])/self.alpha_one_minus_sqrt_cum[i] * model(x, i)) + torch.sqrt(self.beta[i]) * z

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

if __name__ == "__main__":
    from PIL import Image
    from numpy import asarray
    import numpy as np
    import matplotlib.pyplot as plt

    sd = SimpleDiffusion(64)
    example_img = Image.open("./dataset/flowers/daisy/5547758_eea9edfd54_n.jpg")
    example_img = asarray(example_img.resize((64, 64)))
    example_img = 2 * ((example_img - example_img.min()) / (example_img.max() - example_img.min())) - 1
    example_img = np.expand_dims(example_img, axis=0)
    example_img = np.moveaxis(example_img, -1, 1)

    example_img, _ = sd.noise_image(torch.Tensor(example_img), 0)

    example_img = example_img.squeeze().numpy()
    example_img = ((example_img + 1) * 127.5).astype(np.uint8)
    example_img = np.moveaxis(example_img, 0, -1)

    plt.imshow(example_img)
    plt.show()