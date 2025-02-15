from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim_scores(model_G, model_F, dataset):
    stoi = []
    itos = []

    for image, y_true  in dataset:
        y_pred_a = model_G.predict([image], verbose=0)  # Add batch dimension
        y_pred_b = model_F.predict([y_true], verbose=0)
        
        y_true = y_true[0].numpy().astype(np.float32)
        stoi.append(ssim(y_true,
                      y_pred_a[0],
                      data_range=y_pred_a[0].max() - y_pred_a[0].min(),
                      channel_axis=-1))
        
        image = image[0].numpy().astype(np.float32)
        itos.append(ssim(image,
                      y_pred_b[0],
                      data_range=y_pred_b[0].max() - y_pred_b[0].min(),
                      channel_axis=-1))
        
    return {
        "Segmentation -> Image": np.mean(stoi),
        "Image -> Segmentation": np.mean(itos)
    }