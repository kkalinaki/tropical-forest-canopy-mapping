# Tropical Forest Canopy mapping in High-resolution multisource satellite Imary Using Attention recurrent Residual U-Net
This repository contains the codes and supplementary datasets for the study undertaken to map forest canopy changes in the tropical regions of Brunei Using an improved attention recurrent residual U-Net and Multisource High-resolution satellite imagery.

# Files
-models.py 
-AttResUnet_training.py.
-create_patches.py

The Create patches file (create_patches.py)
-This file contains the code for creating the patches. Choose the patch size you would like to use. we used 256px by 256px

# The Models file (models.py)
-This file contains the models and loss functions. All three models—Standard U-Net, Attention U-Net, and Attention Recurrent Residual U-Net—have been defined.

# The Attention Recurrent Residual U-Net file (AttRecResUnet_training.py)
-This file contains the codes to run the three models mentioned above in the Model file.

How to use the files
-For your image datasets, they should be patched with sizes of 256px by 256px, along with the corresponding masks
-Follow the folder structure in the AttRecResUnet_training.py.

How to run the code
-After annotating your images and patching them.
-Run the model.py file from your preferred environment (I use Spyder from Anaconda)
-Then run the AttRecResUnet_training.py file, which has all the models
