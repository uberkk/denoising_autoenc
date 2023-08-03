import torch.nn as nn
# I used this class to create an autoencoder, decoder and encoder can be implemented as different classes
class DenoisingAutoEnc(nn.Module):
    def __init__(self):
        super(DenoisingAutoEnc, self).__init__()
        self.img_size = 32
        self.create_encoder()
        self.create_decoder()

    # downsampling to obtain latent space(encoding)
    def create_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
    # upsampling model to recreate the original resolution from the latent space representation
    def create_decoder(self):
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2,  padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,  padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2,  padding=1, output_padding=1),
            nn.Sigmoid()
        )
    # model workflow
    def forward(self,input_image):
        latent_space = self.encoder(input_image)
        output_image = self.decoder(latent_space)
        return output_image



