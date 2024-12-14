'''
 # @ Author: Meet Patel
 # @ Create Time: 2024-11-07 05:30:00
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-11-09 09:05:07
 # @ Description:
 '''

import os
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


from checkpoint_handler import CheckpointHandler
from model import GeneratorNet, DiscriminatorNet
from dataset_helper import DehazingImageDataset
from metrics import get_psnr, get_ssim
from vgg_perceptual_loss import VGGIntermediate
from logger import SingletonLogger

class CGANDehaze(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, ndf, input_hw, output_hw, 
                 batch_size, num_epochs, dataset_path, sample_size, random_flip, 
                 use_gram_matrix_for_perceptual_loss, normalize_gram_matrix, vgg_layers_to_extract,
                 learned_loss_multipliers, gan_loss_lambda, perceptual_loss_lambda, l1_loss_lambda, grad_loss_lambda,
                 use_amp, output_dir, generator_start_lr, discriminator_start_lr,
                 gen_drop_prob, weight_decay):
        super().__init__()
        self.generator = GeneratorNet(input_nc, output_nc, ngf, gen_drop_prob)
        self.generator = torch.compile(self.generator) 
        self.discriminator = DiscriminatorNet(input_nc, output_nc, ndf)
        self.discriminator = torch.compile(self.discriminator) 
        if perceptual_loss_lambda != 0:
            self.vgg_layers_to_extract = vgg_layers_to_extract
            self.custom_vgg = VGGIntermediate(self.vgg_layers_to_extract)
            self.custom_vgg = torch.compile(self.custom_vgg)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.input_hw = input_hw
        self.output_hw = output_hw
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.random_flip = random_flip
        self.use_gram_matrix_for_perceptual_loss = use_gram_matrix_for_perceptual_loss
        self.normalize_gram_matrix = normalize_gram_matrix
        self.learned_loss_multipliers = learned_loss_multipliers
        self.gan_loss_lambda = gan_loss_lambda
        self.perceptual_loss_lambda = perceptual_loss_lambda
        self.l1_loss_lambda = l1_loss_lambda
        self.grad_loss_lambda = grad_loss_lambda
        self.use_amp = use_amp
        self.generator_start_lr = generator_start_lr
        self.discriminator_start_lr = discriminator_start_lr
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.learned_loss_multipliers:
            # Initialize weights as learnable parameters
            self.gan_loss_lambda = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # Initial weights can be any value
            self.perceptual_loss_lambda = nn.Parameter(torch.tensor(1.0, requires_grad=True))
            self.l1_loss_lambda = nn.Parameter(torch.tensor(1.0, requires_grad=True))
            self.grad_loss_lambda = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        logger_instance = SingletonLogger(log_dir=self.output_dir, log_file="training.log")
        self.logger = logger_instance.get_logger()
        
        self.gpu = torch.device("cuda:0")
        self.host = torch.device("cpu:0")
        
    def bce_loss(self, prediction, label):
        # prediction : [1, 1, 256, 256]     --> Sigmoid is applied in Discriminator model.
        # label      : [1, 1, 256, 256]
        if self.learned_loss_multipliers:
            multiplier = torch.exp(self.gan_loss_lambda)
        else:
            multiplier = self.gan_loss_lambda
        return multiplier * F.binary_cross_entropy_with_logits(prediction, label)
    
    def perceptual_loss(self, generator_clear_image, real_clear_image):
        if self.perceptual_loss_lambda == 0:
            return 0
        
        self.custom_vgg.eval()

        for param in self.custom_vgg.parameters():
            param.requires_grad = False
            
        # Adding 1 to the input which is feed to custom_vgg, because these
        # tensors are generated in the range of [-1, 1]
        outputs_generated = self.custom_vgg((generator_clear_image + 1) / 2)
        outputs_real = self.custom_vgg((real_clear_image.detach() + 1) / 2)

        perceptual_loss = 0
        for output_real, output_gen in zip(outputs_real, outputs_generated):
            batch, channel, height, width = output_real.shape\
            
            if self.use_gram_matrix_for_perceptual_loss:
                with autocast(self.gpu.type, enabled=False):
                    output_real = torch.reshape(output_real, [batch, channel, -1])
                    output_real_transpose = torch.transpose(output_real, 2, 1)
                    if self.normalize_gram_matrix or self.use_amp:
                        # For AMP training, normalizing is must otherwise loss becomes nan.
                        output_real = output_real / (channel * height * width)
                    output_real_gram = torch.bmm(output_real, output_real_transpose)
                    # [B, C, C]
                    
                    output_gen = torch.reshape(output_gen, [batch, channel, -1])
                    output_gen_transpose = torch.transpose(output_gen, 2, 1)
                    if self.normalize_gram_matrix or self.use_amp:
                        # For AMP training, normalizing is must otherwise loss becomes nan.
                        output_gen = output_gen / (channel * height * width)
                    output_gen_gram = torch.bmm(output_gen, output_gen_transpose)
                    # [B, C, C]

                    perceptual_loss += torch.mean((output_gen_gram - output_real_gram) ** 2)
            else:
                perceptual_loss += F.mse_loss(output_gen, output_real)
                    
        perceptual_loss /= len(output_real)
        if self.learned_loss_multipliers:
            multiplier = torch.exp(self.perceptual_loss_lambda)
        else:
            multiplier = self.perceptual_loss_lambda
        return multiplier * perceptual_loss
            
    def tv_loss(self, gen_img_B):
        if self.grad_loss_lambda == 0:
            return 0
        
        B, C, H, W = gen_img_B.size()
        
        # x_diff = torch.abs(gen_img_B[:, :, :H-1, :W-1] - gen_img_B[:, :, :H-1, 1:W])
        # y_diff = torch.abs(gen_img_B[:, :, :H-1, :W-1] - gen_img_B[:, :, 1:H, :W-1])
        # tv_loss = torch.sum(x_diff + y_diff)
        x_diff = gen_img_B[:, :, :H-1, :W-1] - gen_img_B[:, :, :H-1, 1:W]
        y_diff = gen_img_B[:, :, :H-1, :W-1] - gen_img_B[:, :, 1:H, :W-1]
        
        res = torch.zeros_like(gen_img_B)
        res[:, :, :H-1, :W-1] = x_diff + y_diff
        res[:, :, :H-1, 1:W] -= x_diff
        res[:, :, 1:H, :W-1] -= y_diff
        tv_loss = torch.mean(torch.abs(res))
        if self.learned_loss_multipliers:
            multiplier = torch.exp(self.grad_loss_lambda)
        else:
            multiplier = self.grad_loss_lambda
        return multiplier * tv_loss

    def l1_loss(self, gen_img_B, gt_img_B):
        if self.l1_loss_lambda == 0:
            return 0
        l1_loss = nn.L1Loss()
        if self.learned_loss_multipliers:
            multiplier = torch.exp(self.l1_loss_lambda)
        else:
            multiplier = self.l1_loss_lambda
        return multiplier * l1_loss(gen_img_B, gt_img_B)

    def train(self):
        dataset = DehazingImageDataset(self.dataset_path, sample_size=self.sample_size, random_flip=self.random_flip)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        gen_ckpt_handler = CheckpointHandler(self.output_dir, "generator", max_to_keep=3)
        disc_ckpt_handler = CheckpointHandler(self.output_dir, "discriminator", max_to_keep=3)
    
        self.generator.to(self.gpu)
        self.discriminator.to(self.gpu)
        if self.perceptual_loss_lambda != 0:
            self.custom_vgg.to(self.gpu)
        
        kwargs = {}
        # if self.use_amp:
        #     kwargs["amsgrad"] = True
        if self.weight_decay:
            kwargs["weight_decay"] = self.weight_decay
            
        generator_params = list(self.generator.parameters())
        if self.learned_loss_multipliers:
            generator_params.extend([self.gan_loss_lambda, self.perceptual_loss_lambda, self.grad_loss_lambda, self.l1_loss_lambda])
        
        opt_gen = optim.RMSprop(generator_params, lr=self.generator_start_lr, **kwargs) #betas=(0.5, 0.999))
        opt_dis = optim.RMSprop(self.discriminator.parameters(), lr=self.discriminator_start_lr, **kwargs) #, betas=(0.5, 0.999))
        
        total_steps = self.num_epochs * len(data_loader)
        scheduler_gen = CosineAnnealingLR(opt_gen, T_max=total_steps, eta_min=1e-5)
        scheduler_dis = CosineAnnealingLR(opt_dis, T_max=total_steps, eta_min=1e-5)
        
        global_step = 1
        scaler = GradScaler("cuda", enabled=self.use_amp)
        test_img = Image.open(config.test_img_path).convert("RGB")
        test_img = test_img.resize((512, 512))
        
        for epoch in range(self.num_epochs):
            mean_ssim = 0
            mean_psnr = 0
            self.generator.train()
            self.discriminator.train()
        
            for batch_num, data in enumerate(data_loader):
                gt_img_A, gt_img_B, _, hazy_img_B = data
                gt_img_A = gt_img_A.to(self.gpu)
                gt_img_B = gt_img_B.to(self.gpu)
                # hazy_img_A = hazy_img_A.to(self.gpu)
                hazy_img_B = hazy_img_B.to(self.gpu)
                batch_size = gt_img_A.shape[0]
                
                with autocast("cuda", enabled=self.use_amp):
                    gen_img_B_detached = self.generator(hazy_img_B).detach()
                    
                    # Train Discriminator
                    opt_dis.zero_grad()
                    # real input and real shall be output
                    real_output = self.discriminator(torch.cat([gt_img_A, gt_img_B], dim=1))
                    real_label = torch.ones([batch_size, 1, self.output_hw, self.output_hw], dtype=torch.float32).to(self.gpu)
                    real_loss = self.bce_loss(real_output, real_label)
                    
                    # fake input and fake shall be output
                    fake_output = self.discriminator(torch.cat([gt_img_A, gen_img_B_detached], dim=1))
                    fake_label = torch.zeros([batch_size, 1, self.output_hw, self.output_hw], dtype=torch.float32).to(self.gpu)
                    fake_loss = self.bce_loss(fake_output, fake_label)
                    
                    total_discriminator_loss = real_loss + fake_loss
                
                scaler.scale(total_discriminator_loss).backward()
                scaler.step(opt_dis)
                scaler.update()
                
                # Train Generator
                opt_gen.zero_grad()
                with autocast("cuda", enabled=self.use_amp):
                    gen_img_B = self.generator(hazy_img_B)
                    fake_output = self.discriminator(torch.cat([gt_img_A, gen_img_B], dim=1))
                    gen_gan_loss = self.bce_loss(fake_output, real_label)
                    perceptual_loss = self.perceptual_loss(gen_img_B, gt_img_B)
                    l1_loss = self.l1_loss(gen_img_B, gt_img_B)
                    tv_loss = self.tv_loss(gen_img_B)
                    total_generator_loss = gen_gan_loss + perceptual_loss + l1_loss + tv_loss
                
                scaler.scale(total_generator_loss).backward()
                scaler.step(opt_gen)
                scaler.update()
                
                scheduler_gen.step()
                scheduler_dis.step()
                self.logger.info("--"*30)
                self.logger.info(f"Epoch: {epoch+1}/{self.num_epochs}, Batch: {batch_num}/{len(data_loader)}, LR: {scheduler_gen.get_last_lr()[0]:.4f}")
                self.logger.info(f"Gen. GAN Loss: {gen_gan_loss:.4f}, Gen. Perceptual Loss: {perceptual_loss:.4f}, Gen. L1 Loss: {l1_loss:.4f}, Gen. TV Loss: {tv_loss:.4f}")
                self.logger.info(f"Total Gen. Loss: {total_generator_loss:.4f}, Disc. Loss: {total_discriminator_loss:.4f}")
                self.logger.info(f"Gen. GAN lambda: {self.gan_loss_lambda.item():.4f}, Gen. Perceptual lambda: {self.perceptual_loss_lambda.item():.4f}, Gen. L1 lambda: {self.l1_loss_lambda.item():.4f}, Gen. TV lambda: {self.grad_loss_lambda.item():.4f}")
                
                img_1 = gen_img_B.detach().to(self.host)
                img_2 = gt_img_B.detach().to(self.host)
                
                # converting data from [-1, 1] into [0, 255] and with uint8 dtype
                img_1 = ((img_1 + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                img_2 = ((img_2 + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                psnr = get_psnr(img_1, img_2, data_range=255.0)      
                ssim = get_ssim(img_1.to(torch.float32), img_2.to(torch.float32), data_range=255.0)      # adding 1 so that range becomes [0, 2] from [-1, 1]
                mean_ssim += ssim
                mean_psnr += psnr
                
                global_step += 1
            mean_ssim /= len(data_loader)
            mean_psnr /= len(data_loader)
            self.logger.info(f"Mean PSNR: {mean_psnr:.4f}, Mean SSIM: {mean_ssim:.4f}")
            self.logger.info(f"Epoch: {epoch+1}/{self.num_epochs} Completed.")
            
            gen_checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "gen_loss": total_generator_loss,
                "disc_loss": total_discriminator_loss,
                "model": self.generator.state_dict(),\
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
            }
            gen_ckpt_handler.save(gen_checkpoint)
            disc_checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "gen_loss": total_generator_loss,
                "disc_loss": total_discriminator_loss,
                "model": self.discriminator.state_dict(),
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
            }
            disc_ckpt_handler.save(disc_checkpoint)
    
    
            self.generator.eval()
            output_img = self.run_single_img(test_img, resize_wh=None)
            self.generator.train()

            test_img_w, test_img_h = test_img.size
            comb_img = Image.new("RGB", (test_img_w * 2, test_img_h))
            comb_img.paste(test_img, (0, 0))
            comb_img.paste(output_img, (test_img_w, 0))
            comb_img_path = os.path.join(self.output_dir, f"comb_img_epoch_{epoch}.jpg")
            comb_img.save(comb_img_path)


    def run_single_img(self, test_img, resize_wh=(256, 256)):
        if resize_wh:
            test_img = test_img.resize(resize_wh)
        
        test_img = transforms.ToTensor()(test_img)      # converts [H, W, C] into [C, H, W] and divides image by 255.
        test_img = (test_img * 2) - 1                   # converts [0-1] into [-1, 1] 
        test_img = torch.unsqueeze(test_img, 0)
        test_img = test_img.to(self.gpu)
        
        output_img = self.generator(test_img)
        
        output_img = torch.clamp(output_img, -1, 1)

        output_img = (output_img + 1) / 2               # Now in [0, 1]

        transform_to_pil = transforms.ToPILImage()
        output_img = transform_to_pil(output_img.squeeze(0))
        return output_img
    
    def test(self, gen_model_path, test_img_path, test_output_path, save_output=True):
        self.generator.load_state_dict(torch.load(gen_model_path))
        self.generator.eval()
        self.generator.to(self.gpu)
        
        test_img = Image.open(test_img_path).convert("RGB")
        output_img = self.run_single_img(test_img, resize=True)
        
        if save_output:
            output_img.save(test_output_path)
        
        self.generator.train()
    
if __name__ == "__main__":
    import config
    
    model = CGANDehaze(config.input_nc, config.output_nc, config.base_channels_gen, config.base_channels_dis,
                    config.input_hw, config.output_hw, config.batch_size, config.num_epochs, 
                    config.dataset_path, config.sample_size, config.random_flip, 
                    config.use_gram_matrix_for_perceptual_loss,
                    config.normalize_gram_matrix, config.vgg_layers_to_extract,
                    config.learned_loss_multipliers, config.gan_loss_lambda, config.perceptual_loss_lambda, config.l1_loss_lambda, config.grad_loss_lambda,
                    config.use_amp, config.output_dir,
                    config.generator_start_lr, config.discriminator_start_lr, config.gen_drop_prob,
                    config.weight_decay)
    
    if not config.test_mode:
        model.train()
    else:
        model.test(config.gen_model_path, config.test_img_path, config.test_output_path)  