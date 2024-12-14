'''
 # @ Author: Meet Patel
 # @ Create Time: 2024-11-07 09:19:00
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-11-09 09:08:16
 # @ Description:
 '''


# Model Architecture
input_nc = 3
output_nc = 3
input_hw = 256
output_hw = 256
base_channels_gen = 64
base_channels_dis = 48

# Dataset
dataset_path = "/mnt/d/d/DeepLearning/Workfoster-CGAN/Dense_Haze_NTIRE19/"
random_flip = True
sample_size = [input_hw, input_hw]

# Training
num_epochs = 200
batch_size = 2
vgg_layers_to_extract = [4, 9, 16, 23]
use_gram_matrix_for_perceptual_loss = False
normalize_gram_matrix = True

learned_loss_multipliers = True
gan_loss_lambda = 20
perceptual_loss_lambda = 2
l1_loss_lambda = 50
grad_loss_lambda = 0
use_amp = True
output_dir = "./experiment_files"
generator_start_lr = 1e-3
discriminator_start_lr = 1e-3
gen_drop_prob = 0.1
weight_decay = 1e-4

# Testing
test_mode = False
gen_model_path = "/mnt/d/d/DeepLearning/Workfoster-CGAN/CGAN-Dehazing/saved_models/generator_epoch_9.pth"
test_img_path = "/mnt/d/d/DeepLearning/Workfoster-CGAN/Dense_Haze_NTIRE19/hazy/02_hazy.png"
test_output_path = "./test_result.jpg"

