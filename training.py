import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
import wandb
from tqdm import tqdm
import os

class SFTTrainer:
    def __init__(
        self,
        unet,
        vae,
        text_encoder,
        noise_scheduler,
        train_dataloader,
        val_dataloader,
        mask_temperature=1.0,
        device="cuda:0",
        save_dir="./checkpoints",
    ):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.mask_temperature = mask_temperature
        self.device = device
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir

    def loss_step(self, batch):
        
        weight_dtype = torch.float32
        batch_size = batch["pixel_values"].shape[0]
        
        # encode the image
        latents = self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype, device=self.device)).latent_dist.sample()

        # encode the masked image
        masked_image_latents = self.vae.encode(batch["masked_image_values"].to(dtype=weight_dtype, device=self.device)).latent_dist.sample()
        
        # scale the latents
        masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        latents = latents * self.vae.config.scaling_factor

        # scale the mask
        mask = F.interpolate(
                    batch["mask"].to(dtype=weight_dtype, device=self.device),
                    scale_factor=1 / self.vae_scale_factor,
                )

        # add noise to the latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            int(self.noise_scheduler.config.num_train_timesteps * 1.0),
            (batch_size,),
            device=latents.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        latent_model_input = torch.cat(
                [noisy_latents, mask, masked_image_latents], dim=1
            )


        encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.device))[0]
        
        model_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if batch.get("mask", None) is not None:
            
            mask = (
                batch["mask"]
                .to(model_pred.device)
                .reshape(
                    model_pred.shape[0], 1, model_pred.shape[2] * self.vae_scale_factor, model_pred.shape[3] * self.vae_scale_factor
                )
            )
            # resize to match model_pred
            mask = F.interpolate(
                mask.float(),
                size=model_pred.shape[-2:],
                mode="nearest",
            )

            mask = mask.pow(self.mask_temperature)

            mask = mask / mask.max()

            model_pred = model_pred * mask

            target = target * mask

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss


    def train_step(self, loss):  
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()
        torch.cuda.empty_cache()

    def train_epoch(self):
        "Train for one epoch, log metrics and save model"
        self.unet.train()
        pbar = tqdm(self.train_dataloader, leave=False, desc="Training")
        
        for step, batch in enumerate(self.train_dataloader):
            with torch.autocast("cuda"):
                loss = self.loss_step(batch)

            self.train_step(loss)
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({
                "loss": loss.item(),
                "lr": self.lr_scheduler.get_last_lr()[0]
                })
                
        pbar.close()

    def val_epoch(self):
        "Validate for one epoch, log metrics and save model"
        self.unet.eval()
        pbar = tqdm(self.val_dataloader, leave=False, desc="Validation")
        with torch.no_grad():
            for step, batch in enumerate(self.val_dataloader):
                loss = self.loss_step(batch)
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({
                "val_loss": loss.item(),
                })
        pbar.close()

    def prepare(self, config):
        self.unet.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.epochs)
        self.scaler = torch.cuda.amp.GradScaler()
        wandb.init(project="sd-sft", entity='arked', config=config)
        wandb.watch(self.unet, log="all")

    def fit(self, config):
        self.prepare(config)
        for epoch in range(config.epochs):
            self.train_epoch()
            if epoch % config.val_epochs == 0:
                self.val_epoch()
                self.save(epoch)
        self.save(epoch)

    def save_model(self, epoch):
        save_path = os.path.join(self.save_dir, f"unet_{epoch}.pt")
        torch.save(self.unet.state_dict(), save_path)
        # torch.save(self.vae.state_dict(), f"vae_{epoch}.pt")
        # torch.save(self.text_encoder.state_dict(), f"text_encoder_{epoch}.pt")

    def save_optimizer(self, epoch):
        save_path = os.path.join(self.save_dir, f"optimizer_{epoch}.pt")
        torch.save(self.optimizer.state_dict(), save_path)

    def save_scheduler(self, epoch):
        save_path = os.path.join(self.save_dir, f"scheduler_{epoch}.pt")
        torch.save(self.lr_scheduler.state_dict(), save_path)

    def load_model(self, epoch):
        load_path = os.path.join(self.save_dir, f"unet_{epoch}.pt")
        self.unet.load_state_dict(torch.load(load_path))
        # self.vae.load_state_dict(torch.load(f"vae_{epoch}.pt"))
        # self.text_encoder.load_state_dict(torch.load(f"text_encoder_{epoch}.pt"))

    def load_optimizer(self, epoch):
        load_path = os.path.join(self.save_dir, f"optimizer_{epoch}.pt")
        self.optimizer.load_state_dict(torch.load(load_path))

    def load_scheduler(self, epoch):
        load_path = os.path.join(self.save_dir, f"scheduler_{epoch}.pt")
        self.lr_scheduler.load_state_dict(torch.load(load_path))

    def load(self, load_path):
        self.load_model(load_path)
        self.load_optimizer(load_path)
        self.load_scheduler(load_path)

    def save(self, epoch):
        self.save_model(epoch)
        self.save_optimizer(epoch)
        self.save_scheduler(epoch)    

    def print_model_info(self):
        total_params = sum(p.numel() for p in self.unet.parameters())
        total_trainable_params = sum(
            p.numel() for p in self.unet.parameters() if p.requires_grad
        )
        print(f"Unet: {total_params:,} total parameters.")
        print(f"Unet: {total_trainable_params:,} training parameters.")