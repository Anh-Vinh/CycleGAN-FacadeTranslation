# CycleGAN-FacadeTranslation
This project trains a CycleGAN model to translate images from the CMP Facade dataset to their corresponding building segmentation masks and vice versa. The CycleGAN model is designed for unpaired image-to-image translation, making it suitable for tasks where paired datasets are limited.

# Dataset
The CMP Facade dataset consists of images of building facades and their corresponding segmentation masks. The dataset is used to train the CycleGAN model to learn mappings between real facade images and their segmentation counterparts.

# Training
The model was trained on Kaggle for 200 epochs, using Adam optimizer with learning rate of 2e-4 for the first 100 epoch and linear decay for the next 100 like in the paper. Generator G maps segmentation masks to building facade images, while Generator F performs the reverse mapping.

# Result
|SSIM                 |Training|Testing|
|---------------------|--------|-------|
|Segmentation -> Image| 0.0177 | 0.383 |
|Image -> Segmentation| 0.0285 | 0.396 |

# References
- CycleGAN Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- CMP Facade Dataset: http://cmp.felk.cvut.cz/%7Etylecr1/facade/
