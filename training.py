import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
import wandb

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

    def loss_step(self, batch):
        
        weight_dtype = torch.float32
        batch_size = batch["pixel_values"].shape[0]
        
        # encode the image
        latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents.to(dtype=weight_dtype, device=self.device)

        # encode the masked image
        masked_image_latents = self.vae.encode(batch["masked_image_values"]).latent_dist.sample()
        masked_image_latents.to(dtype=weight_dtype, device=self.device)

        # scale the latents
        masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        latents = latents * self.vae.config.scaling_factor

        # scale the mask
        mask = F.interpolate(
                    batch["mask"].to(dtype=weight_dtype, device=unet.device),
                    scale_factor=1 / self.vae_scale_factor,
                )

        # add noise to the latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            int(noise_scheduler.config.num_train_timesteps * t_mutliplier),
            (batch_size,),
            device=latents.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        latent_model_input = torch.cat(
                [noisy_latents, mask, masked_image_latents], dim=1
            )


        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
        encoder_hidden_states = encoder_hidden_states.to(text_encoder.device)

        model_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        if batch.get("mask", None) is not None:
            
            mask = (
                batch["mask"]
                .to(model_pred.device)
                .reshape(
                    model_pred.shape[0], 1, model_pred.shape[2] * vae_scale_factor, model_pred.shape[3] * vae_scale_factor
                )
            )
            # resize to match model_pred
            mask = F.interpolate(
                mask.float(),
                size=model_pred.shape[-2:],
                mode="nearest",
            )

            mask = mask.pow(mask_temperature)

            mask = mask / mask.max()

            model_pred = model_pred * mask

            target = target * mask

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss


    def train_step(self, loss):  
        lr_scheduler.step()
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.lr_scheduler.step()

    def train_epoch(self):
        "Train for one epoch, log metrics and save model"
        self.unet.train()
        pbar = tqdm(self.train_dataloader, leave=False, desc="Training")
        
        for step, batch in enumerate(self.train_dataloader):
            with torch.autocast(self.device):
                loss = self.loss_step(batch)

            self.train_step(loss)
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({
                "loss": loss.item(),
                "lr": self.scheduler.get_last_lr()[0]
                })
                
        pbar.close()

    def val_epoch(self):
        "Validate for one epoch, log metrics and save model"
        self.unet.eval()
        pbar = tqdm(self.val_dataloader, leave=False, desc="Validation")
        with torch.no_grad():
            for step, batch in enumerate(self.val_dataloader):
                with torch.autocast(self.device):
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
        self.noise_scheduler.to(self.device)
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.epochs)
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
        torch.save(self.unet.state_dict(), f"unet_{epoch}.pt")
        # torch.save(self.vae.state_dict(), f"vae_{epoch}.pt")
        # torch.save(self.text_encoder.state_dict(), f"text_encoder_{epoch}.pt")

    def save_optimizer(self, epoch):
        torch.save(self.optimizer.state_dict(), f"optimizer_{epoch}.pt")

    def save_scheduler(self, epoch):
        torch.save(self.scheduler.state_dict(), f"scheduler_{epoch}.pt")

    def load_model(self, epoch):
        self.unet.load_state_dict(torch.load(f"unet_{epoch}.pt"))
        # self.vae.load_state_dict(torch.load(f"vae_{epoch}.pt"))
        # self.text_encoder.load_state_dict(torch.load(f"text_encoder_{epoch}.pt"))

    def load_optimizer(self, epoch):
        self.optimizer.load_state_dict(torch.load(f"optimizer_{epoch}.pt"))

    def load_scheduler(self, epoch):
        self.scheduler.load_state_dict(torch.load(f"scheduler_{epoch}.pt"))

    def load(self, epoch):
        self.load_model(epoch)
        self.load_optimizer(epoch)
        self.load_scheduler(epoch)

    def save(self, epoch):
        self.save_model(epoch)
        self.save_optimizer(epoch)
        self.save_scheduler(epoch)    

    def print_model_info(self):
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"Unet: {total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.unet.parameters() if p.requires_grad
        )
        print(f"Unet: {total_trainable_params:,} training parameters.")