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
- Segmentation to image:
![test1](https://github.com/user-attachments/assets/41de77d6-374f-41e0-961b-8976704ad121)
![test2](https://github.com/user-attachments/assets/72f30a1b-a7c1-42eb-8653-3241d32a348f)
![test3](https://github.com/user-attachments/assets/c5f09d9a-8dd8-46a1-bb21-e3f6152061bd)
- Image to segmentation:
![test1](https://github.com/user-attachments/assets/90ba3696-b80e-42a8-841d-91e5fefe6109)
![test2](https://github.com/user-attachments/assets/c4ab2ccf-9c9e-4198-9cdb-a326189a64c6)
![test3](https://github.com/user-attachments/assets/d1353b21-5ac3-42c0-a19a-1ee526d18ab5)


# References
- CycleGAN Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- CMP Facade Dataset: http://cmp.felk.cvut.cz/%7Etylecr1/facade/
- Author's result with the same dataset: https://taesung.me/cyclegan/2017/03/25/facades.html
