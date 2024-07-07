import tensorflow as tf
import numpy as np
import os
from os import path
import csv
import textwrap
import pandas as pd

from tensorflow.keras.regularizers import l1
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.optimizers import Adam

models = []
class_1_accuracies = []

original_dir = '/WACV_Paper/Dataset/maid-dataset-high-frequency/original'
denoised_dir = '/WACV_Paper/Dataset/maid-dataset-high-frequency/denoised'
csv_path     = '/WACV_Paper/Dataset/maid-dataset-high-frequency/classified_label.csv'
result_file_path = "/WACV_Paper/Result/Result_High_Frequency.csv"

#########################################################################################################################################################################################################################################

def extract_y_channel_from_yuv_with_patch_numbers(yuv_file_path: str, width: int, height: int):
    y_size = width * height
    patches, patch_numbers = [], []

    if not os.path.exists(yuv_file_path):
        print(f"Warning: File {yuv_file_path} does not exist.")
        return [], []

    with open(yuv_file_path, 'rb') as f:
        y_data = f.read(y_size)

    if len(y_data) != y_size:
        print(f"Warning: Expected {y_size} bytes, got {len(y_data)} bytes.")
        return [], []

    y_channel = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))
    patch_number = 0

    for i in range(0, height, 224):
        for j in range(0, width, 224):
            patch = y_channel[i:i+224, j:j+224]
            if patch.shape[0] < 224 or patch.shape[1] < 224:
                patch = np.pad(patch, ((0, 224 - patch.shape[0]), (0, 224 - patch.shape[1])), 'constant')
            patches.append(patch)
            patch_numbers.append(patch_number)
            patch_number += 1

    return patches, patch_numbers

#########################################################################################################################################################################################################################################

def load_data_from_csv(csv_path, original_dir, denoised_dir):
    df = pd.read_csv(csv_path)
    
    all_original_patches = []
    all_denoised_patches = []
    all_scores = []
    denoised_image_names = []
    all_patch_numbers = []

    for _, row in df.iterrows():
        
        original_file_name = f"original_{row['image_name']}.raw"
        denoised_file_name = f"denoised_{row['image_name']}.raw"

        original_path = os.path.join(original_dir, original_file_name)
        denoised_path = os.path.join(denoised_dir, denoised_file_name)
        
        original_patches, original_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(original_path, row['width'], row['height'])
        denoised_patches, denoised_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])

        all_original_patches.extend(original_patches)
        all_denoised_patches.extend(denoised_patches)
        denoised_image_names.extend([row['image_name']] * len(denoised_patches))
        all_patch_numbers.extend(denoised_patch_numbers) 


        scores = np.array([0 if float(score) == 0 else 1 for score in row['patch_score'].split(',')])
        if len(scores) != len(original_patches) or len(scores) != len(denoised_patches):
            print(f"Error: Mismatch in number of patches and scores for {row['image_name']}")
            continue
        all_scores.extend(scores)

    return all_original_patches, all_denoised_patches, all_scores, denoised_image_names, all_patch_numbers

#########################################################################################################################################################################################################################################

def calculate_difference(original, ghosting):
  return [ghost.astype(np.int16) - orig.astype(np.int16) for orig, ghost in zip(original, ghosting)]

#########################################################################################################################################################################################################################################

def prepare_data(data, labels):
    data = np.array(data).astype('float32') / 255.0
    lbl = np.array(labels)
    return data, lbl

#########################################################################################################################################################################################################################################

def save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path):

    if path.exists(result_file_path):
    
        df_existing = pd.read_csv(result_file_path)
        df_new_row = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],
            'Feature Map' : [feature_name],
            'Overall Accuracy': [test_acc],
            'Precision': [weighted_precision],
            'Recall': [weighted_recall],
            'F1-Score': [weighted_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })
        df_metrics = pd.concat([df_existing, df_new_row], ignore_index=True)
    else:
    
        df_metrics = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],            
            'Feature Map' : [feature_name],
            'Overall Accuracy': [test_acc],
            'Precision': [weighted_precision],
            'Recall': [weighted_recall],
            'F1-Score': [weighted_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })

    df_metrics.to_csv(result_file_path, index=False)

##########################################################################################################################################################################

def create_vgg16_model(input_shape=(224,224, 1)):
    input_layer = Input(shape=input_shape)
  
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
  
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)

    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


##########################################################################################################################################################################

original_patches, denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)

diff_patches = calculate_difference(original_patches, denoised_patches)
diff_patches_np, labels_np = prepare_data(diff_patches, labels)

print(f" Total Patches: {len(diff_patches_np)}")
print(f" Total Labels: {len(labels_np)}")

combined = list(zip(diff_patches_np, labels_np, denoised_image_names, all_patch_numbers))
combined = sklearn_shuffle(combined)

total_patches, total_labels, image_names, patch_numbers = zip(*combined)

total_patches = np.array(total_patches)
total_labels = np.array(total_labels)

print(f" Total Train Patches: {len(total_patches)}")
print(f" Total Train Labels: {len(total_labels)}")

##########################################################################################################################################################################
##########################################################################################################################################################################

X_train, X_temp, y_train, y_temp = train_test_split(total_patches, total_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"X_Train Shape: {X_train.shape}")
print(f"y_Train Shape: {y_train.shape}")

print(f"X_Val Shape: {X_val.shape}")
print(f"y_Val Shape: {y_val.shape}")

print(f"X_Test Shape: {X_test.shape}")
print(f"y_Test Shape: {y_test.shape}")


##########################################################################################################################################################################
##########################################################################################################################################################################
                                                            ## Without Class Weight
##########################################################################################################################################################################
##########################################################################################################################################################################

opt = Adam(learning_rate=2e-05)
vgg16_wcw_model = create_vgg16_model()
vgg16_wcw_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
wcw_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='/WACV_Paper/Models_RAW/VGG16_Diff_wCW.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
wcw_model_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, restore_best_weights=True)

wcw_history = vgg16_wcw_model.fit(X_train, y_train, epochs=2, validation_data=(X_val, y_val), callbacks=[wcw_model_checkpoint, wcw_model_early_stopping])


##########################################################################################################################################################################
##########################################################################################################################################################################
                                                            ## With Class Weight
##########################################################################################################################################################################
##########################################################################################################################################################################

ng = len(total_patches[total_labels == 0])
ga =  len(total_patches[total_labels == 1])
total = ng + ga

imbalance_ratio = ng / ga  
weight_for_0 = (1 / ng) * (total / 2.0)
weight_for_1 = (1 / ga) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0 (Non-ghosting): {:.2f}'.format(weight_for_0))
print('Weight for class 1 (Ghosting): {:.2f}'.format(weight_for_1))

opt = Adam(learning_rate=2e-05)
vgg16_cw_model = create_vgg16_model()
vgg16_cw_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

cw_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='/WACV_Paper/Models_RAW/VGG16_Diff_CW.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
cw_model_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, restore_best_weights=True)

cw_history = vgg16_cw_model.fit(X_train, y_train, epochs=2, class_weight=class_weight, validation_data=(X_val, y_val), callbacks=[cw_model_early_stopping, cw_model_checkpoint])


##########################################################################################################################################################################
##########################################################################################################################################################################
                                                                    # Testing
##########################################################################################################################################################################
##########################################################################################################################################################################

X_test = np.array(X_test)
# X_test = X_test.reshape((-1, 224, 224, 1))

##########################################################################################################################################################################
## Without Class Weight

test_loss, test_acc = vgg16_wcw_model.evaluate(X_test, y_test)
test_acc  = test_acc *100

predictions = vgg16_wcw_model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=-1)

precision, recall, _ = precision_recall_curve(true_labels, predictions)

pr_data = pd.DataFrame({'Precision': precision, 'Recall': recall })
file_path = '/WACV_Paper/Plots_RAW/VGG16_Diff_wCW_PR_Curve.csv'
pr_data.to_csv(file_path, index=False)


plt.figure()
plt.plot(recall, precision, linestyle='-', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
precision_recall_curve_path = '/WACV_Paper/Plots_RAW/VGG16_Diff_wCW_PR_Curve.png'

if not os.path.exists(os.path.dirname(precision_recall_curve_path)):
    os.makedirs(os.path.dirname(precision_recall_curve_path))

plt.savefig(precision_recall_curve_path, dpi=300)
plt.close()


report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

conf_matrix = confusion_matrix(true_labels, predicted_labels)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FP
total_class_1 = TP + FN
correctly_predicted_0 = TN
correctly_predicted_1 = TP


accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

if weighted_precision + weighted_recall > 0:
    weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
else:
    weighted_f1_score = 0

weighted_f1_score  = weighted_f1_score*100
weighted_precision = weighted_precision*100
weighted_recall    = weighted_recall*100


model_name = "VGG16"
feature_name = "Difference Map"
technique = "Without Class Weight"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)
print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")


##########################################################################################################################################################################

## With Class Weight

test_loss, test_acc = vgg16_cw_model.evaluate(X_test, y_test)
test_acc  = test_acc *100

predictions = vgg16_cw_model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=-1)

precision, recall, _ = precision_recall_curve(true_labels, predictions)

pr_data = pd.DataFrame({'Precision': precision, 'Recall': recall })
file_path = '/WACV_Paper/Plots_RAW/VGG16_Diff_CW_PR_Curve.csv'
pr_data.to_csv(file_path, index=False)


plt.figure()
plt.plot(recall, precision, linestyle='-', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
precision_recall_curve_path = '/WACV_Paper/Plots_RAW/VGG16_Diff_CW_PR_Curve.png'

if not os.path.exists(os.path.dirname(precision_recall_curve_path)):
    os.makedirs(os.path.dirname(precision_recall_curve_path))

plt.savefig(precision_recall_curve_path, dpi=300)
plt.close()


report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

conf_matrix = confusion_matrix(true_labels, predicted_labels)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FP
total_class_1 = TP + FN
correctly_predicted_0 = TN
correctly_predicted_1 = TP


accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

if weighted_precision + weighted_recall > 0:
    weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
else:
    weighted_f1_score = 0

weighted_f1_score  = weighted_f1_score*100
weighted_precision = weighted_precision*100
weighted_recall    = weighted_recall*100


model_name = "VGG16"
feature_name = "Difference Map"
technique = "Class Weight"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)
print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")
