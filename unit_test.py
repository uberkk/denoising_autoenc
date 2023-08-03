import unittest
import dataloader
import torch
import model

class Test(unittest.TestCase):
    # tests whether datashapes are as expected, to use all parameters multiple times(to avoid
    # unnecessary initializations) @classmethod is used
    @classmethod
    def setUpClass(self):
        torch.manual_seed(42)
        self.model = model.DenoisingAutoEnc()
        self.batch_size = 8
        self.img_size = 32
        self.shape = [self.batch_size,3,self.img_size,self.img_size]
        self.custom_dataset_flag = 1
        self.training_dataset="./code_test/train"
        self.test_dataset="./code_test/test"
        self.val_dataset="./code_test/val"
        self.test_input = torch.randn(self.shape)
        self.dataloader = dataloader.ImageDataLoader(self.batch_size, 0)

    # checks org shape and the dataloader's output same or not
    def test_dataloader_shape(self):
        for batch_idx, org_img in enumerate(self.dataloader.train_loader):
            self.assertEqual(org_img[0].shape[0], self.shape[0])
            self.assertEqual(org_img[0].shape[1], self.shape[1])
            self.assertEqual(org_img[0].shape[2], self.shape[2])
            self.assertEqual(org_img[0].shape[3], self.shape[3])

    # checks the max and min value ranges of dataset
    def test_range(self):
        for batch_idx, org_img in enumerate(self.dataloader.train_loader):
            self.assertGreaterEqual(1, org_img[0].max())
            self.assertLessEqual(0, org_img[0].min())

    # checks autoencoder gives an output with same dimension as input
    def test_network_shape(self):
        output = self.model(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)


if __name__ == '__main__':
    unittest.main()