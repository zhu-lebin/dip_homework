import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from pix2pix_dataset import pix2pixDataset
from FCN_network2 import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F



def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

# --- Discriminator ---
#返回值不是单个数字，而是一个二维特征图
#判别器的输出形状为 [batch_size, 1, h, w]
#这种方法的优点是判别器对图像的局部特征进行判断
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            # Adjusting the final layer to output a single value per image (for binary classification)
            # nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),  # Output one channel for each image
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1),  # Output one channel for each image
            nn.Sigmoid()  # Sigmoid activation to output a probability
        )

    # def forward(self, input_image, generated_image):
    #     combined = torch.cat((input_image, generated_image), dim=1)  # Concatenate along channel dimension
    #     return self.model(combined)  # Output a single value for each image
    def forward(self, input): 
        return self.model(input)  # Output a single value for each image
    

# --- Loss Functions ---
#这个损失函数会计算每个预测值和目标值之间的二元交叉熵，然后取平均值。
def gan_loss(pred, target):
    return nn.BCELoss()(pred, target)


def l1_loss(generated, target):
    return torch.mean(torch.abs(generated - target))



def train_one_epoch(generator, discriminator,dataloader,criterion, optimizer_g,optimizer_d, device, epoch, num_epochs,output_dir,history,log_file_path):
    """
    Train the model for one epoch.

    Args:
        # model (nn.Module): The neural network model.
        # dataloader (DataLoader): DataLoader for the training data.
        # optimizer (Optimizer): Optimizer for updating model parameters.
        # criterion (Loss): Loss function.
        # device (torch.device): Device to run the training on.
        # epoch (int): Current epoch number.
        # num_epochs (int): Total number of epochs.
    """
    generator.train()
    discriminator.train()
    train_loss_g = 0.0
    train_loss_d = 0.0
    train_loss_g_l1 = 0.0
    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)
        real_A=image_rgb
        real_B=image_semantic
        
        
       # 训练判别器
        fake_B = generator(real_A)
        d_real=discriminator(torch.cat((real_A, real_B), 1))
        d_fake=discriminator(torch.cat((real_A, fake_B.detach()), 1))
        loss_d_real = gan_loss(d_real, torch.ones_like(d_real))
        loss_d_fake = gan_loss(d_fake, torch.zeros_like(d_fake))
        loss_d = (loss_d_real + loss_d_fake) / 2
        train_loss_d += loss_d.item()
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # 训练生成器
        d_fake=discriminator(torch.cat((real_A, fake_B.detach()), 1))
        loss_g_gan = gan_loss(d_fake, torch.ones_like(d_fake))
        loss_g_l1 = criterion(fake_B, real_B)
        loss_g = loss_g_gan + 50* loss_g_l1
        train_loss_g += loss_g.item()
        train_loss_g_l1 += loss_g_l1.item()

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        
         # Save sample images every 5 epochs
        if epoch % 10 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_B, f'{output_dir}/train_results', epoch)
        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss_g: {loss_g.item():.4f}')
    # Calculate average training loss
    avg_train_loss_g = train_loss_g / len(dataloader)
    avg_train_loss_d = train_loss_d / len(dataloader)
    avg_train_loss_g_l1 = train_loss_g_l1/ len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {avg_train_loss_g:.4f}, Loss_g_l1:{avg_train_loss_g_l1:.4f}, Discriminator Loss: {avg_train_loss_d:.4f}')
    # 保存日志
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {avg_train_loss_g:.4f},  Loss_g_l1:{avg_train_loss_g_l1:.4f},Discriminator Loss: {avg_train_loss_d:.4f}\n")
    history["train_loss_g"].append(avg_train_loss_g)
    history["train_loss_d"].append(avg_train_loss_d)


def validate(model, dataloader, criterion, device, epoch, num_epochs, output_dir,history,log_file_path):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 10 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, f'{output_dir}/val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    history["val_loss"].append(avg_val_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    # 保存日志
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}\n")
# Function to create a unique output directory
def create_output_dir(base_dir="output-cgan-50"):
    # Ensure the base output directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create a unique subdirectory using timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir)
    os.makedirs(output_dir + '/train_results')
    os.makedirs(output_dir + '/val_results')
    os.makedirs(output_dir + '/checkpoints')
    print(f"Output directory created: {output_dir}")
    return output_dir

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = pix2pixDataset(dataset_dir='datasets/cityscapes/train')
    val_dataset = pix2pixDataset(dataset_dir='datasets/cityscapes/val')
    output_dir = create_output_dir()
    log_file_path = os.path.join(output_dir, "training_log.txt")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    generator = FullyConvNetwork().to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # 定义优化器
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.L1Loss()
    # Add a learning rate scheduler for decay
    scheduler_g = StepLR(optimizer_g, step_size=50, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, step_size=50, gamma=0.5)
    # Training loop
    num_epochs = 800
    history = {
        "train_loss_g": [],
        "train_loss_d": [],
        "val_loss": []
    }

    for epoch in range(num_epochs):
        train_one_epoch(generator,discriminator, train_loader, criterion,optimizer_g,optimizer_d, device, epoch, num_epochs,output_dir,history,log_file_path) 
        if epoch % 1 == 0 :
            validate(generator, val_loader, criterion, device, epoch, num_epochs,output_dir,history,log_file_path)

        # Step the scheduler after each epoch
        scheduler_g.step()
        scheduler_d.step()
        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'{output_dir}/checkpoints/pix2pix_generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'{output_dir}/checkpoints/pix2pix_discriminator_epoch_{epoch + 1}.pth')
if __name__ == '__main__':
    main()
