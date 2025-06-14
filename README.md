## Malaria Cell Classification Using Vision Transformer (ViT)

**Project Objective** 🔬

This project aims to build a deep learning model capable of classifying images of red blood cells as either **"Parasitized"** (infected with malaria) or **"Uninfected"**. The model employed is a **Vision Transformer (ViT)**, specifically the `google/vit-base-patch16-224` pretrained model, which has been adapted for this binary classification task.

**Methodology** ⚙️
1.  **Dataset**: The project uses a dataset of red blood cell images from Kaggle, divided into two classes. The data is split into a training set (80%), a validation set (10%), and a test set (10%).
2.  **Model**: A pre-trained Vision Transformer model (`ViTForImageClassification`) from Hugging Face is used. The model's classifier head was replaced to match the two target classes.
3.  **Preprocessing & Augmentation**: Images are processed to meet the input requirements of the ViT model. Data augmentation techniques such as rotation, cropping, flipping, and color jittering are applied to the training data to improve model robustness.
4.  **Training**: The model is trained using the `CrossEntropyLoss` function and optimized with various combinations of optimizers (Adam and AdamW), learning rates, and batch sizes. An *Early Stopping* mechanism is implemented to prevent overfitting and save the model with the best validation accuracy.
5.  **Evaluation**: The model's performance is evaluated on the previously unseen test set using metrics such as accuracy, precision, recall, f1-score, and a confusion matrix.

---

### Hyperparameter Testing Results

Here is a summary of the results from four different testing scenarios to find the optimal configuration of optimizer, learning rate, and batch size.

| Test | Optimizer | Learning Rate | Batch Size | Epochs Run (out of 50) | Best Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **AdamW** | **$10^{-4}$** | **32** | 11 | **97.31%** | **97.46%** |
| **2** | **AdamW** | $10^{-3}$ | 64 | 20 | 96.08% | 96.12% |
| **3** | **Adam** | **$10^{-4}$** | **32** | **17** | **97.64%** | **97.71%** |
| **4** | **Adam** | $10^{-3}$ | 64 | 17 | 96.08% | 95.75% |

---

### Analysis and Conclusion

Based on the results table above, several conclusions can be drawn:

1.  **Lower Learning Rate Performs Better**: Tests using a lower learning rate ($10^{-4}$) consistently produced higher validation and test accuracies (above 97%) compared to the higher learning rate ($10^{-3}$), which resulted in accuracies around 95-96%. This suggests that a smaller learning rate allowed the model to converge to a more optimal solution.

2.  **Adam Slightly Outperforms AdamW**: At the $10^{-4}$ learning rate, the **Adam Optimizer (Test 3)** demonstrated slightly superior performance with a **test accuracy of 97.71%**, compared to AdamW (Test 1) with a test accuracy of 97.46%. Although the difference is not significant, Adam yielded the best result in this scenario.

3.  **Smaller Batch Size Works Better**: The combination of a low learning rate and a smaller batch size (32) delivered the best results. Using a larger batch size (64) with a high learning rate appeared to make the training less stable and resulted in lower accuracy.

**Final Conclusion:**
The best configuration in this series of tests was the **Adam optimizer** with a **learning rate of $10^{-4}$** and a **batch size of 32**. This setup successfully trained the Vision Transformer model to achieve an accuracy of **97.71%** on the test data, demonstrating its excellent capability in distinguishing between malaria-infected and uninfected cells.

model : https://huggingface.co/docs/transformers/en/model_doc/vit#vision-transformer-vit

dataset : https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
