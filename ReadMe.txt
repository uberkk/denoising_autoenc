for testing:
python main.py --model_path $Trained_model_path$ --image_path $Input_image_path$ --mode test

-> image_path = path to single image which will be denoised
-> Denoised images can be seen in /results/test_results, there are 4 outputs noisy image, denoised images and all images together(with the original name)
-> PSNR values can be seen from terminal.
-> Two different sized images are put in test_image for testing.
-> Two pretrained weights are put in ./pretrained_weights and training validation loss graph for each epoch can be seen.

Sample:
python main.py --model_path ./pretrained_weights/20.pth --image_path ./test_image/christian-holzinger-1262_rndcrop_256x256_05.png --mode test


for training:
python main.py --mode train

Weights can be obtained under /weights

python main.py --model_path /home/berk/Desktop/inter/weights/10.pth --image_path /home/berk/Desktop/inter/test_image/kodim01.png --mode test


for code testing(unit test):

python unit_test.py

for dependencies:
use 
pip install -r requirements.txt