import torch
import torch.nn as nn
import model
import dataloader
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import peak_signal_noise_ratio
import os
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
# Argument parser is used to select whether test or train
def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Denoising Autoencoder")
    parser.add_argument("--image_path", type=str, help="Path to the test image", required=False)
    parser.add_argument("--model_path", type=str, help="Path to trained model", required=False)
    parser.add_argument("--mode", type=str, help="Purpose of the script: 'train' for training or 'test' for testing", required=False)
    return parser.parse_args()

# PIL image loader function
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
#Converting PIL Image to torch tensor
def convert_pil_to_torch(img):
    transform = transforms.ToTensor()
    img = transform(img)
    img.unsqueeze(0)
    return img
#converting torch to PIL and save img

def convert_torch_to_pil_save(tensor, img_name):
    image = transforms.ToPILImage()(tensor)
    image.save('./results/test_results/' + img_name)

#sketching outputs together to save and show
def plot_imgs(org_img, noisy_img,autoencoder_denoised_img, wavelet_denoised_img, epoch_number, img_name = '',
              mode='train'):
    if (mode !='train'):
        plt.figure(figsize=(16, 16))
    else:
        plt.figure(figsize=(18,18))
    number_of_img = autoencoder_denoised_img.shape[0]
    for i, image in enumerate(autoencoder_denoised_img):
        plt.subplot(4,number_of_img, i+1)
        plt.imshow(org_img[i].permute(1, 2, 0).cpu())
        plt.title('Orginal Image', fontsize=10)
        plt.subplot(4,number_of_img, number_of_img + i+1)
        plt.imshow(noisy_img[i].permute(1, 2, 0).cpu())
        plt.title('Noisy Image', fontsize=10)
        plt.subplot(4, number_of_img, number_of_img*2 + i+1)
        plt.imshow(autoencoder_denoised_img[i].permute(1, 2, 0).cpu())
        plt.title('Autoencoder Denoised Img', fontsize=10)
        plt.subplot(4,number_of_img, number_of_img*3 + i+1)
        plt.imshow(wavelet_denoised_img[i])
        plt.title('Wavelet Denoised Img', fontsize=10)

    #plt.show()
    if(mode == 'train'):
        plt.savefig('./results/val_results/' + str(epoch_number) + '.png')
    elif(mode == 'test'):
        plt.savefig('./results/test_results/' + img_name)
        convert_torch_to_pil_save(autoencoder_denoised_img[0], img_name =('autoencoder_denoised_img' +'.png'))
        convert_torch_to_pil_save(noisy_img[0], img_name =('noisy_img' + '.png'))
        Image.fromarray((wavelet_denoised_img[0]*255).astype(np.uint8)).save("./results/test_results/wavelet_denoised"
                                                                             ".png")
#denosing noisy img using wavelet bayes method(thresholding different levels with different threshold, better than
#visushrink)
def wavelet_denoiser(org_img, noisy_img):
    psnr_noisy = 0
    psnr_bayes = 0
    denoised_wavelet = []
    for img_idx in range(org_img.shape[0]):
        noisy_single_img = noisy_img[img_idx].permute(1, 2, 0).numpy()
        img_bayes = denoise_wavelet(noisy_single_img, method='BayesShrink', mode='soft',
                                    wavelet_levels=3, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
        img_bayes = img_bayes.clip(min=0, max=1)
        psnr_noisy += peak_signal_noise_ratio(org_img[img_idx].permute(1, 2, 0).numpy(), noisy_single_img)
        psnr_bayes += peak_signal_noise_ratio(org_img[img_idx].permute(1, 2, 0).numpy(), img_bayes)
        denoised_wavelet.append(img_bayes)
    psnr_noisy = psnr_noisy/org_img.shape[0]
    psnr_bayes = psnr_bayes/org_img.shape[0]
    print(f'PSNR Noisy = {psnr_noisy}, PSNR_Bayes = {psnr_bayes}\n')
    return denoised_wavelet, psnr_bayes
# class method to train/validate/test the network, init args can be implemented as config or argument
class trainTestModel():
    def __init__(self, purpose='train',number_of_epoch=50, batch_size=8, learning_rate=0.001):
        self.model = model.DenoisingAutoEnc()
        self.loss_func = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (purpose != "test"):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.number_of_epoch = number_of_epoch
            self.batch_size = batch_size
            self.data_loader = dataloader.ImageDataLoader(self.batch_size)
            self.current_epoch = 0
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
            self.loss_autoenc_denoised_list_val = []
            self.loss_autoenc_denoised_list_train = []
            self.epoch_list = []


    # adding gaussin noise with a given standart deviation and mean
    def add_noise(self, img, mean=0, std=0.1):
        noise = torch.randn_like(img) * std + mean
        noisy_img = img + noise
        noisy_image_clamped = torch.clamp(noisy_img, min=0, max=1)
        return noisy_image_clamped
    # training function
    def model_train(self): # Training
        self.model.to(self.device)
        self.model.train()
        for epoch_idx in range (self.number_of_epoch):
            total_loss = 0
            total_batch_epoch = 0
            for batch_idx, org_img in enumerate(self.data_loader.train_loader):
                org_img = org_img[0]
                noisy_img = self.add_noise(org_img)
                noisy_img = noisy_img.to(self.device)
                self.optimizer.zero_grad()
                denoised_img = self.model(noisy_img)
                org_img = org_img.to(self.device)
                loss = self.loss_func(denoised_img, org_img)
                total_loss += loss/self.batch_size
                total_batch_epoch += 1
                PSNR = 10.0 * torch.log10(1.0 / torch.tensor(loss.item()/self.batch_size))
                loss.backward()
                self.optimizer.step()
                if (batch_idx %1500 == 0):
                    print("Epoch {}, Batch {}: Loss= {}, PSNR= {}".format(epoch_idx, batch_idx,loss/self.batch_size, PSNR))

            self.scheduler.step(loss)
            self.current_epoch += 1
            if (epoch_idx % 1 == 0 or 1):
                self.epoch_list.append(int(epoch_idx+1))
                avg_loss = total_loss / total_batch_epoch
                self.loss_autoenc_denoised_list_train.append(avg_loss.cpu())
                print("\n Epoch {}: Avg Loss= {}".format(epoch_idx, avg_loss))
                self.save_checkpoint(avg_loss)
                self.model_eval()
    # evaluation function
    def model_eval(self): #Validation
        self.model.to(self.device)
        self.model.eval()
        small_batch = 4
        total_loss = 0
        total_batch = 0
        # avoiding gradient calculation
        with torch.no_grad():
            for batch_idx, org_img in enumerate(self.data_loader.test_loader):
                org_img = org_img[0]
                noisy_img = (self.add_noise(org_img)).to(self.device)
                org_img = org_img.to(self.device)
                autoenc_denoised_img = self.model(noisy_img)
                loss = self.loss_func(autoenc_denoised_img, org_img)
                total_loss += loss/self.batch_size
                total_batch = batch_idx
                if (batch_idx == 0):
                    loss_small_batch = self.loss_func(autoenc_denoised_img[0:small_batch,:,:,:], org_img[0:small_batch,:,:,:])
                    psnr_autoenc_denoised = 10.0 * torch.log10(1.0 / torch.tensor((loss_small_batch/small_batch).item()))
                    wavelet_denoised_imgs, psnr_wavelet_denoised = wavelet_denoiser(org_img[0:small_batch,:,:,:].cpu(), noisy_img[0:small_batch,:,:,:].cpu())
                    plot_imgs(org_img[0:small_batch, :, :, :], noisy_img[0:small_batch, :, :, :], autoenc_denoised_img[0:small_batch, :, :, :],
                                   wavelet_denoised_imgs, self.current_epoch, mode='train')
                    print(f"Validation Autoencoder Denoiser PSNR: {psnr_autoenc_denoised}")
                    print(f"Validation Wavelet Denoiser PSNR: {psnr_wavelet_denoised}")

            avg_loss_val = total_loss/total_batch
            self.loss_autoenc_denoised_list_val.append(avg_loss_val.cpu())
            plt.figure(figsize=(16, 16))
            plt.plot(self.epoch_list, self.loss_autoenc_denoised_list_val, label = 'Validation')
            plt.plot(self.epoch_list, self.loss_autoenc_denoised_list_train, label = 'Train')
            plt.xticks(range(min(self.epoch_list), max(self.epoch_list) + 1))
            plt.legend()
            plt.savefig("./results/loss_curve.png")

    # test function: denoise the specified image by loading the model in model_path
    def model_test(self, test_path, model_path): # Test
        self.load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        org_img = pil_loader(test_path)
        org_img = convert_pil_to_torch(org_img).unsqueeze(0)
        img_name = os.path.basename(test_path)
        with torch.no_grad():
            noisy_img = (self.add_noise(org_img)).to(self.device)
            org_img = org_img.to(self.device)
            autoenc_denoised_img = self.model(noisy_img)
            loss = self.loss_func(autoenc_denoised_img, org_img)
            psnr_autoenc_denoised = 10.0 * torch.log10(1.0 / torch.tensor(loss.item()))
            wavelet_denoised_imgs, psnr_wavelet_denoised = wavelet_denoiser(org_img.cpu(), noisy_img.cpu())
            plot_imgs(org_img, noisy_img, autoenc_denoised_img,
                           wavelet_denoised_imgs, 0, mode='test', img_name= img_name)
            print(f"Autoencoder Denoiser PSNR: {psnr_autoenc_denoised}")
            print(f"Wavelet Denoiser PSNR: {psnr_wavelet_denoised}")


    # Saving trained weights
    def save_checkpoint(self, loss, path= './weights/'):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, (path+ str(self.current_epoch)+'.pth'))

    # Loading trained weights
    def load_checkpoint(self, path):

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        if (self.device == "cuda"):
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.model.train()


if __name__ == '__main__':
    args = parse_arguments()
    test_path = args.image_path
    model_path = args.model_path
    mode = args.mode

    Model = trainTestModel(purpose= mode)
    if (mode == "train"):
        Model.model_train()
        print('Training is completed.')

    elif (mode == "test"):
        Model.model_test(test_path, model_path)
        print("Test is completed, you can check ./results/test_results for output")
