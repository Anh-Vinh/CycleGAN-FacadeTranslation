# CycleGAN-FacadeTranslation
This project trains a CycleGAN model to translate images from the CMP Facade dataset to their corresponding building segmentation masks and vice versa. The CycleGAN model is designed for unpaired image-to-image translation, making it suitable for tasks where paired datasets are limited.

# Dataset
The CMP Facade dataset consists of images of building facades and their corresponding segmentation masks. The dataset is used to train the CycleGAN model to learn mappings between real facade images and their segmentation counterparts.

# Training
The model was trained on Kaggle for 200 epochs, using Adam optimizer with learning rate of 2e-4 for the first 100 epoch and linear decay for the next 100 like in the paper. Generator G maps segmentation masks to building facade images, while Generator F performs the reverse mapping.

# Result
|SSIM                 |Training|Testing|
|---------------------|--------|-------|
|Segmentation -> Image| 0.018  | 0.384 |
|Image -> Segmentation| 0.029  | 0.397 |

Some result from the test dataset
![image](https://github.com/user-attachments/assets/6b8c5227-a71d-47b4-9c5d-a95cacd1b980)
![image](https://github.com/user-attachments/assets/4c5af0fe-28ac-44e2-9068-19b08be7d3ff)
![image](https://github.com/user-attachments/assets/d66bcf27-d938-4fd1-ba32-c0355b434ac1)

Result from the original paper for references:
![image](https://github.com/user-attachments/assets/c2d4cdae-4835-488d-b379-f70206a813ef)
More original paper's result in: https://taesung.me/cyclegan/2017/03/25/facades.html

# References
- CycleGAN Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- CMP Facade Dataset: http://cmp.felk.cvut.cz/%7Etylecr1/facade/
- Author's result with the same dataset: https://taesung.me/cyclegan/2017/03/25/facades.html
