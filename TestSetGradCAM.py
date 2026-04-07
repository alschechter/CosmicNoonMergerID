import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import torch
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE, Isomap
import glob
from astropy.visualization import simple_norm
import os
#from torchvision.io import read_image
from tqdm import tqdm
import pandas as pd
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchsummary import summary
from SetRandomSeed import set_random_seeds
from skimage.transform import resize
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier


set_random_seeds(626)
#class_for_gradcam = 'nonmerger'
class_for_gradcam = 'merger'

learning_rate = 3.154957151204336e-07
lr_decay = 0.8814595270497834
dropout = 0.1205273604259237
hidden_size = 256
num_classes = 2

# Load the saved model state
checkpoint = torch.load('nano_1_trial1_lr3.154957151204336e-07_epoch120.pth', map_location=torch.device('cpu'))

model = FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', learning_rate=learning_rate,
    layer_decay=lr_decay,
    num_classes=num_classes
)
model.head = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(640, hidden_size),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, num_classes)
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load the model weights from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)  # Ensure the model is on the correct device (GPU/CPU)
model.eval()  # Switch to evaluation mode


from BinaryMergerDataset import BinaryMergerDataset, get_transforms
path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/NormalizedDataset/'
BATCH_SIZE = 64

test_mergers_dataset_orig = BinaryMergerDataset(path, 'test', mergers = True, transform = get_transforms(aug=False), codetest=False)
test_nonmergers_dataset_orig = BinaryMergerDataset(path, 'test', mergers = False, transform = get_transforms(aug=False), codetest=False)

test_dataset_full = torch.utils.data.ConcatDataset([test_mergers_dataset_orig, test_nonmergers_dataset_orig])
test_dataloader = DataLoader(test_dataset_full, shuffle = True, num_workers = 0, batch_size=BATCH_SIZE)#num workers used to be 4

all_labels = []
all_preds = []
all_names = []
all_probabilities = []

with torch.no_grad():  # No need to track gradients during inference
    for images, labels, names in tqdm(test_dataloader):
        #print(type(names))
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        # Forward pass
        outputs = model(images)
        # print(type(outputs))
        # print(outputs.shape)
        probabilities = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)   # Convert to binary (0 or 1)
        pred = pred.to(device=device) #dtype=torch.float32
        maxvals, pred_index = torch.max(outputs, 1)
        # Collect labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_names.extend(names)
        all_probabilities.extend(probabilities.cpu().numpy())
        
# 4. Compute accuracy or other evaluation metrics (e.g., confusion matrix)
# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.squeeze(np.array(all_preds))
all_names = np.array(all_names)
all_probabilities = np.array(all_probabilities)


# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_names = np.array(all_names)



if class_for_gradcam == 'merger':
    targetclass = 1
else:
    targetclass = 0
    
print('targetclass', targetclass)
#save_dir = 'gradcam/' + CNNName + '/test/predicted_label_' + class_for_gradcam
save_dir = '/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/gradcam/zoobot'
for subdir in ['TruePositive', 'TrueNegative', 'FalsePositive', 'FalseNegative']:
    os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)

#### GradCAM with help from github and chat GPT
# We have to specify the target we want to generate the CAM for.
# targets = [ClassifierOutputTarget(0)] #target class is mergers
# print(targets)
#targets = [ClassifierOutputTarget(targetclass)] #target class is mergers
targets = [0, 1]
# print(targets)
print('ClassifierOutputTarget', ClassifierOutputTarget(targetclass), type(ClassifierOutputTarget(targetclass)))
path_test = []
targets_GradCAM = [model.encoder.stages[-1].blocks[-1]]
layer_names_GradCAM = ['stages_last_block']

# Build lookup from name -> predicted class using already-computed batch predictions
name_to_pred = dict(zip(all_names, all_preds))

# Create GradCAM once and reuse across all images/targets
cam = GradCAM(model=model, target_layers=targets_GradCAM)

# Select the first validation image to visualize Grad-CAM
sample_index_choices = np.arange(0, len(test_dataloader.dataset))#len(test_dataloader.dataset))#  # Change this index to visualize different images
for sample_index in sample_index_choices:
    sample_image = test_dataloader.dataset[sample_index][0].unsqueeze(0)  # Add batch dimension
    sample_label = test_dataloader.dataset[sample_index][1]
    sample_name = test_dataloader.dataset[sample_index][2]
    
    # z1path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground_z1/' + sample_name + 'wfc3_ir_f125w.npy'
    # z15path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground_z15/' + sample_name + 'wfc3_ir_f125w.npy'
    # if os.path.exists(z1path):
    #     #path_test.append('z1')
    #     all = glob.glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground_z1/' + sample_name + '*.npy')
    #     print(len(all))
    #     i0 = resize(np.load(all[0]), (224,224))
    #     i1 = resize(np.load(all[1]), (224,224))
    #     i2 = resize(np.load(all[2]), (224,224))
    #     i3 = resize(np.load(all[3]), (224,224))
    #     ibig = i0 + i1 + i2 + i3
    #     ibig = ibig * 1e20
    #     norm = simple_norm(ibig, 'log', log_a=2e5)
    #     ibig = norm(ibig)
    #     mean = np.mean(ibig)
    #     std = np.std(ibig) 
    #     contour_levels = [mean + i * std for i in [1,3,5]]
    # elif os.path.exists(z15path):
    #     #path_test.append('z15')
    #     all = glob.glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground_z15/' + sample_name + '*.npy')
    #     print(len(all))
    #     i0 = resize(np.load(all[0]), (224,224))
    #     print(np.shape(np.load(all[0])))
    #     i1 = resize(np.load(all[1]), (224,224))
    #     i2 = resize(np.load(all[2]), (224,224))
    #     i3 = resize(np.load(all[3]), (224,224))
    #     ibig = i0 + i1 + i2 + i3
    #     ibig = ibig * 1e20
    #     norm = simple_norm(ibig, 'log', log_a=2e5)
    #     ibig = norm(ibig)
    #     mean = np.mean(ibig)
    #     std = np.std(ibig) 
    #     contour_levels = [mean + i * std for i in [1,3,5]]
    # else:
    #     path_test.append('uhhh')
    #print(str(sample_name))
    # Move to device
    sample_image = sample_image.to(torch.float32).to(device)

    #make labels words
    if sample_label == 1.:
        sample_label_word = 'merger'
    else:
        sample_label_word = 'nonmerger'
        
    
    # Get the original image for visualization
    original_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
    predicted_class = int(name_to_pred[sample_name])  # look up from batch inference, avoids extra forward pass

    if predicted_class == 1:
        predicted_class_word = 'merger'
    else:
        predicted_class_word = 'nonmerger'

    # Determine classification category for save directory
    # In this file: label/pred 1 = merger (positive), 0 = nonmerger (negative)
    if sample_label == 1. and predicted_class == 1:    # true merger, predicted merger
        classification_dir = os.path.join(save_dir, 'TruePositive')
    elif sample_label == 0. and predicted_class == 0:  # true nonmerger, predicted nonmerger
        classification_dir = os.path.join(save_dir, 'TrueNegative')
    elif sample_label == 0. and predicted_class == 1:  # true nonmerger, predicted merger
        classification_dir = os.path.join(save_dir, 'FalsePositive')
    else:                                               # true merger, predicted nonmerger
        classification_dir = os.path.join(save_dir, 'FalseNegative')

    fig, axes = plt.subplots(1, len(targets)+1, figsize=(15, 6))  # Create subplots for each target class and original image
    axes = axes.ravel()  # Flatten axes array if it's multi-dimensional
    plt.subplots_adjust(wspace=0, hspace=0)
    # Original image subplot
    ax = axes[0]
    im_sum = np.sum(original_img, axis=2)
    #extent = [0, im_sum.shape[1], im_sum.shape[0], 0]  # [x_min, x_max, y_max, y_min]
    mean = np.mean(im_sum)
    std = np.std(im_sum)
    # contour_levels2 = [mean + i * std for i in [1,3,5]]
    contour_levels3 = [mean + i * std for i in [3,5]]
    #extent = [0, im_sum.shape[1], im_sum.shape[0], 0]  # [x_min, x_max, y_max, y_min]
    im_original = ax.imshow(im_sum, cmap='magma', norm=LogNorm())  ##, extent = extent Log scale #vmin=np.min(im_sum)*1e-7, vmax=1
    #ax.contour(ibig, colors = 'white', levels = contour_levels, zorder = 1, origin = 'upper') #, extent = extent
    ax.contour(im_sum, colors = 'white', levels = contour_levels3, zorder = 1)
    ax.set_title(f"Original Image - True: {sample_label_word}", fontsize = 'medium')
    ax.axis('off')
    # cbar_o = fig.add_axes([.91,.124,.04,.754])
    # cbar_original = fig.colorbar(im_original, ax=cbar_o, orientation='vertical',)
    # cbar_original.set_label('Intensity (log scale)')
    # #cbar_original.set_ticks([1e-1,1e0,10**(0.3)])
    # cbar_original.locator = MaxNLocator(nbins=3)  # Set number of ticks
    # cbar_original.update_ticks()  # Apply the new tick settings
    # Generate the Grad-CAM visualization
    #targets = [ClassifierOutputTarget(int(sample_label))]  # Use the true label as target
    counter = 1 #gradCAMS start in second subplot
    for target_class in targets:
            
        # if target_class == 'merger':
        #     targetclass = 0
        # else:
        #     targetclass = 1
        if target_class == 0:
            target_class_word = 'nonmerger'    
        else:
            target_class_word = 'merger'
        target = [ClassifierOutputTarget(target_class)] #setting to merger or nonmerger
        grayscale_cam = cam(input_tensor=sample_image, targets=target)
        grayscale_cam = grayscale_cam[0, :]  # Get the CAM for the first image in the batch
        # Convert the input image to a format suitable for visualization
        rgb_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
        #grayscale_cam = grayscale_cam#.numpy()

        # Visualize the Grad-CAM output
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)



        # Plot the Grad-CAM
        #print(np.min(original_img), np.max(original_img)) #0s and 1s cause it's normalized 
        #print(np.shape(np.sum(original_img, axis = 2)))
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns


        # Plot the Grad-CAM
        ax = axes[counter]
        im = ax.imshow(visualization) #, extent = extent
        #ax.contour(ibig, colors = 'white', levels = contour_levels, zorder=1,  origin = 'upper') #, extent = extent
        ax.contour(im_sum, colors = 'white', levels = contour_levels3, zorder=1) #, extent = extent
        ax.set_title(f"Class Activated: {target_class_word} - Pred: {predicted_class_word}", fontsize = 'medium')
        ax.axis('off')

        # Optionally, add a colorbar for each subplot


        # if counter == 2:
        #     cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink = 0.60)
        #     cbar.set_label('Grad-CAM Intensity')
        #     cbar.set_ticks([])

        counter += 1
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.85, 0.01, 0.4])
    # cbar_grad = fig.colorbar(im, cax=cbar_ax)
    # cbar_grad.set_ticks([])
    plt.tight_layout()
    plt.savefig(f"{classification_dir}/gradcam_test_image_{sample_name}.png") #, dpi=300
    #plt.show()
    plt.clf()
    plt.close()

# save_dir2 = 'gradcam/' + CNNName + '/test/guided'
# if os.path.exists(save_dir2):
#     print('yes this directory exists')
# # print(targets)
# # Select the first validation image to visualize Grad-CAM
# sample_index_choices = np.arange(0, len(test_dataloader.dataset))#len(test_dataloader.dataset))#  # Change this index to visualize different images
# for sample_index in sample_index_choices:
#     sample_image = test_dataloader.dataset[sample_index][0].unsqueeze(0)  # Add batch dimension
#     sample_label = test_dataloader.dataset[sample_index][1]
#     sample_name = test_dataloader.dataset[sample_index][2]
    
#     z1path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground_z1/' + sample_name + 'wfc3_ir_f125w.npy'
#     z15path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground_z15/' + sample_name + 'wfc3_ir_f125w.npy'
#     if os.path.exists(z1path):
#         #path_test.append('z1')
#         all = glob.glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground_z1/' + sample_name + '*.npy')
#         print(len(all))
#         i0 = resize(np.load(all[0]), (224,224))
#         i1 = resize(np.load(all[1]), (224,224))
#         i2 = resize(np.load(all[2]), (224,224))
#         i3 = resize(np.load(all[3]), (224,224)) #need to leave i3 in because we dont know what order glob grabbed in
#         ibig = i0 + i1 + i2 + i3
#         ibig = ibig * 1e20
#         norm = simple_norm(ibig, 'log', log_a=2e5)
#         ibig = norm(ibig)
#         mean = np.mean(ibig)
#         std = np.std(ibig) 
#         contour_levels = [mean + i * std for i in [3,5]]
#     elif os.path.exists(z15path):
#         #path_test.append('z15')
#         all = glob.glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground_z15/' + sample_name + '*.npy')
#         print(len(all))
#         i0 = resize(np.load(all[0]), (224,224))
#         i1 = resize(np.load(all[1]), (224,224))
#         i2 = resize(np.load(all[2]), (224,224))
#         i3 = resize(np.load(all[3]), (224,224))
#         ibig = i0 + i1 + i2 + i3
#         ibig = ibig * 1e20
#         norm = simple_norm(ibig, 'log', log_a=2e5)
#         ibig = norm(ibig)
#         mean = np.mean(ibig)
#         std = np.std(ibig) 
#         contour_levels = [mean + i * std for i in [3,5]]
#     else:
#         path_test.append('uhhh')
#     #print(str(sample_name))
#     # Move to device
#     sample_image = sample_image.to(torch.float32).to(device)

#     #make labels words
#     if sample_label == 1.:
#         sample_label_word = 'nonmerger'
#     else:
#         sample_label_word = 'merger'
        

#     # Get the original image for visualization
#     #am I uing predictions when I need to be using probabilities?
#     original_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
#     predicted_class = torch.argmax((model(sample_image)), dim=1) 
#     predicted_label = f"Predicted: {predicted_class}, True: {sample_label_word}"
    
#     if predicted_class == 1.:
#         predicted_class_word = 'nonmerger'
#     else:
#         predicted_class_word = 'merger'
        
#     fig, axes = plt.subplots(1, len(targets)+1, figsize=(15, 6))  # Create subplots for each target class and original image
#     axes = axes.ravel()  # Flatten axes array if it's multi-dimensional
#     plt.subplots_adjust(wspace=0, hspace=0)
#     # Original image subplot
#     ax = axes[0]
#     im_sum = np.sum(original_img, axis=2)
#     # mean = np.mean(im_sum)
#     # std = np.std(im_sum)
#     # contour_levels = [mean + i * std for i in [1,3,5]]
#     im_original = ax.imshow(im_sum, cmap='magma', norm=LogNorm())  # Log scale #vmin=np.min(im_sum)*1e-7, vmax=1
#     ax.contour(ibig, colors = 'white', levels = contour_levels, zorder=1)
#     ax.set_title(f"Original Image - True: {sample_label_word}", fontsize = 'medium')
#     ax.axis('off')
#     # cbar_o = fig.add_axes([.91,.124,.04,.754])
#     # cbar_original = fig.colorbar(im_original, ax=cbar_o, orientation='vertical',)
#     # cbar_original.set_label('Intensity (log scale)')
#     # #cbar_original.set_ticks([1e-1,1e0,10**(0.3)])
#     # cbar_original.locator = MaxNLocator(nbins=3)  # Set number of ticks
#     # cbar_original.update_ticks()  # Apply the new tick settings
#     # Generate the Grad-CAM visualization
#     #targets = [ClassifierOutputTarget(int(sample_label))]  # Use the true label as target
#     counter = 1 #gradCAMS start in second subplot
#     for target_class in targets:
            
#         # if target_class == 'merger':
#         #     targetclass = 0
#         # else:
#         #     targetclass = 1
#         if target_class == 0:
#             target_class_word = 'merger'    
#         else:
#             target_class_word = 'nonmerger'
#         target = [ClassifierOutputTarget(target_class)] #setting to merger or nonmerger
#         target_layer_GradCAM = targets_GradCAM[0]
#         # for target_layer_GradCAM in targets_GradCAM:    
#         #     layername = layer_names_GradCAM[counter]
#         #     counter += 1
#         #print('target', target_layer_GradCAM)
#         with HiResCAM(model=model, target_layers=[target_layer_GradCAM]) as cam:
#             #print('hello??')
#             #print('shapes', np.shape(sample_image), np.shape(targets))
#             grayscale_cam = cam(input_tensor=sample_image, targets=target)
#             #print('hi?')
#             print(np.shape(grayscale_cam)) 
#             grayscale_cam = grayscale_cam[0, :]  # Get the CAM for the first image in the batch
#             #print('greyscale2', grayscale_cam)
#         # Convert the input image to a format suitable for visualization
#         rgb_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
#         #grayscale_cam = grayscale_cam#.numpy()

#         # Visualize the Grad-CAM output
#         visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)



#         # Plot the Grad-CAM
#         #print(np.min(original_img), np.max(original_img)) #0s and 1s cause it's normalized 
#         #print(np.shape(np.sum(original_img, axis = 2)))
#         #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns


#         # Plot the Grad-CAM
#         ax = axes[counter]
#         im = ax.imshow(visualization)
#         ax.contour(ibig, colors = 'white', levels = contour_levels, zorder=1)
#         ax.set_title(f"Class Activated: {target_class_word} - Pred: {predicted_class_word}", fontsize = 'medium')
#         ax.axis('off')

#         # Optionally, add a colorbar for each subplot


#         # if counter == 2:
#         #     cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink = 0.60)
#         #     cbar.set_label('Grad-CAM Intensity')
#         #     cbar.set_ticks([])

#         counter += 1
#     # fig.subplots_adjust(right=0.8)
#     # cbar_ax = fig.add_axes([0.85, 0.85, 0.01, 0.4])
#     # cbar_grad = fig.colorbar(im, cax=cbar_ax)
#     # cbar_grad.set_ticks([])
#     plt.tight_layout()
#     plt.savefig(f"{save_dir2}/gradcam_test_image_{sample_name}.png") #, dpi=300
#     #plt.show()
#     plt.clf()
#     plt.close()

# # ##### OLD ONE CLASS AT A TIME GRADCAM
# # # targets = [ClassifierOutputTarget(targetclass)] #target class is mergers
# # # # print(targets)
# # # print('ClassifierOutputTarget', ClassifierOutputTarget(targetclass), type(ClassifierOutputTarget(targetclass)))

# # # targets_GradCAM = [model.layer4[-1]]
# # # layer_names_GradCAM = ['layer4_n1']
# # # # Select the first validation image to visualize Grad-CAM
# # # sample_index_choices = np.arange(0, len(test_dataloader.dataset))#  # Change this index to visualize different images
# # # for sample_index in sample_index_choices:
# # #     sample_image = test_dataloader.dataset[sample_index][0].unsqueeze(0)  # Add batch dimension
# # #     sample_label = test_dataloader.dataset[sample_index][1]
# # #     sample_name = test_dataloader.dataset[sample_index][2]
# # #     #print(str(sample_name))
# # #     # Move to device
# # #     sample_image = sample_image.to(torch.float32).to(device)

    
# # #     # Get the original image for visualization
# # #     #am I uing predictions when I need to be using probabilities?
# # #     original_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
# # #     predicted_class = torch.argmax((model(sample_image)), dim=1) 
# # #     predicted_label = f"Predicted: {predicted_class}, True: {sample_label}"


# # #     # Generate the Grad-CAM visualization
# # #     #targets = [ClassifierOutputTarget(int(sample_label))]  # Use the true label as target
# # #     counter = 0
# # #     for target_layer_GradCAM in targets_GradCAM:    
# # #         layername = layer_names_GradCAM[counter]
# # #         counter += 1
# # #         #print('target', target_layer_GradCAM)
# # #         with GradCAM(model=model, target_layers=[target_layer_GradCAM]) as cam:
# # #             #print('hello??')
# # #             #print('shapes', np.shape(sample_image), np.shape(targets))
# # #             grayscale_cam = cam(input_tensor=sample_image, targets=targets)
# # #             #print('hi?')
# # #             print(np.shape(grayscale_cam)) 
# # #             grayscale_cam = grayscale_cam[0, :]  # Get the CAM for the first image in the batch
# # #             #print('greyscale2', grayscale_cam)
# # #         # Convert the input image to a format suitable for visualization
# # #         rgb_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
# # #         #grayscale_cam = grayscale_cam#.numpy()

# # #         # Visualize the Grad-CAM output
# # #         visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# # #         #make labels words
# # #         if sample_label == 1.:
# # #             sample_label_word = 'nonmerger'
# # #         else:
# # #             sample_label_word = 'merger'
            
# # #         if predicted_class == 1.:
# # #             predicted_class_word = 'nonmerger'
# # #         else:
# # #             predicted_class_word = 'merger'
# # #         # Plot the Grad-CAM
# # #         #print(np.min(original_img), np.max(original_img)) #0s and 1s cause it's normalized 
# # #         #print(np.shape(np.sum(original_img, axis = 2)))
# # #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
# # #         # Original image subplot
# # #         im1 = ax1.imshow(np.sum(original_img, axis = 2), cmap = 'magma', norm = 'log') #assign image for colorbar
# # #         ax1.set_title(f"Original Image - True: {sample_label_word}")
# # #         ax1.axis('off')  # Turn off axis
# # #         cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', norm=LogNorm(vmin = 0, vmax = 1)) #vmin=np.min(original_img), vmax=np.max(original_img)
# # #         cbar1.set_label('Intensity (log scale)')
# # #         #GradCAM
# # #         im2 = ax2.imshow(visualization)
# # #         #ax2.title(f'Grad-CAM for Validation Image {sample_index} Layer {layername} (True Label: {sample_label})')
# # #         ax2.set_title(f"Predicted: {predicted_class_word}")
# # #         ax2.axis('off')
# # #         cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical')
# # #         cbar2.set_label('Grad-CAM Intensity')
        
# # #         plt.tight_layout()
# # #         plt.savefig(f"{save_dir}/gradcam_test_image_{sample_name}.png") #, dpi=300
# # #         #plt.show()
# # #         plt.clf()
# # #         plt.close()
            
