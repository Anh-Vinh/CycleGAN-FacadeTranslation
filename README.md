# CycleGAN-FacadeTranslation
This project trains a CycleGAN model to translate images from the CMP Facade dataset to their corresponding building segmentation masks and vice versa. The CycleGAN model is designed for unpaired image-to-image translation, making it suitable for tasks where paired datasets are limited.

# Dataset
The CMP Facade dataset consists of images of building facades and their corresponding segmentation masks. The dataset is used to train the CycleGAN model to learn mappings between real facade images and their segmentation counterparts.

# Result
|        |SSIM|
|--------|----|
|Training|0.38|
|Testing |0.40|

# References
- CycleGAN Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- CMP Facade Dataset: http://cmp.felk.cvut.cz/%7Etylecr1/facade/
