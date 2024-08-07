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
from tensorflow.keras import layers, models

from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D
from keras.layers import  MaxPooling2D, Flatten, Dense, Dropout, concatenate
from keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.optimizers import Adam

original_dir = '/WACV_Paper/Dataset/maid-dataset-high-frequency/original'
denoised_dir = '/WACV_Paper/Dataset/maid-dataset-high-frequency/denoised'
csv_path     = '/WACV_Paper/Dataset/maid-dataset-high-frequency/classified_label.csv'

result_file_path = "/WACV_Paper/Result/High_Frequency_RAW_Results.csv"

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

def save_metric_details(model_name, technique, test_acc, weighted_precision, weighted_recall, weighted_f1_score, macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path):

    if path.exists(result_file_path):
    
        df_existing = pd.read_csv(result_file_path)
        df_new_row = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],
            'Overall Accuracy': [test_acc],
            'Weight Precision': [weighted_precision],
            'Weight Recall': [weighted_recall],
            'Weight F1-Score': [weighted_f1_score],
            'Macro Precision': [macro_precision],
            'Macro Recall': [macro_recall],
            'Macro F1-Score': [macro_f1_score],
            'Micro Precision': [micro_precision],
            'Micro Recall': [micro_recall],
            'Micro F1-Score': [micro_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })
        df_metrics = pd.concat([df_existing, df_new_row], ignore_index=True)
    else:
 
        df_metrics = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],
            'Overall Accuracy': [test_acc],
            'Weight Precision': [weighted_precision],
            'Weight Recall': [weighted_recall],
            'Weight F1-Score': [weighted_f1_score],
            'Macro Precision': [macro_precision],
            'Macro Recall': [macro_recall],
            'Macro F1-Score': [macro_f1_score],
            'Micro Precision': [micro_precision],
            'Micro Recall': [micro_recall],
            'Micro F1-Score': [micro_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })

    df_metrics.to_csv(result_file_path, index=False)
##########################################################################################################################################################################

def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6

def hard_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

class SEBlock(layers.Layer):
    def __init__(self, input_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(input_channels // reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(input_channels, activation=hard_sigmoid)

    def call(self, inputs):
        x = self.pool(inputs)
        x = tf.expand_dims(tf.expand_dims(x, 1), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return inputs * x

class Bottleneck(layers.Layer):
    def __init__(self, input_channels, out_channels, kernel_size, expansion_factor, stride, use_se, activation):
        super(Bottleneck, self).__init__()
        self.use_se = use_se
        self.stride = stride
        self.activation = activation

        self.expand = models.Sequential([
            layers.Conv2D(int(input_channels * expansion_factor), kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation)
        ]) if expansion_factor != 1 else lambda x: x

        self.depthwise = models.Sequential([
            layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation)
        ])

        self.se = SEBlock(int(input_channels * expansion_factor)) if use_se else lambda x: x

        self.project = models.Sequential([
            layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])

        self.shortcut = (stride == 1 and input_channels == out_channels)

    def call(self, inputs):
        x = self.expand(inputs)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.shortcut:
            x = layers.add([x, inputs])
        return x

class MobileNetV3(models.Model):
    def __init__(self, input_shape=(224, 224, 1), model_type='large'):
        super(MobileNetV3, self).__init__()

        self.model_type = model_type

        self.init_conv = models.Sequential([
            layers.Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False, input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation(hard_swish)
        ])

        self.bottlenecks = self._make_bottlenecks()

        if model_type == 'large':
            self.final_layers = models.Sequential([
                layers.Conv2D(960, kernel_size=1, strides=1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Activation(hard_swish),
                layers.GlobalAveragePooling2D(),
                layers.Dense(1280, activation=hard_swish),
                layers.Dense(1, activation='sigmoid')
            ])
        else:  # 'small'
            self.final_layers = models.Sequential([
                layers.Conv2D(576, kernel_size=1, strides=1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Activation(hard_swish),
                layers.GlobalAveragePooling2D(),
                layers.Dense(1024, activation=hard_swish),
                layers.Dense(1, activation='sigmoid')
            ])

    def _make_bottlenecks(self):
        bottleneck_layers = []
        if self.model_type == 'large':
            bottleneck_layers.extend([
                Bottleneck(16, 16, 3, 1, 1, False, 'relu'),
                Bottleneck(16, 24, 3, 4, 2, False, 'relu'),
                Bottleneck(24, 24, 3, 3, 1, False, 'relu'),
                Bottleneck(24, 40, 5, 3, 2, True, 'relu'),
                Bottleneck(40, 40, 5, 3, 1, True, 'relu'),
                Bottleneck(40, 40, 5, 3, 1, True, 'relu'),
                Bottleneck(40, 80, 3, 6, 2, False, hard_swish),
                Bottleneck(80, 80, 3, 2.5, 1, False, hard_swish),
                Bottleneck(80, 80, 3, 2.3, 1, False, hard_swish),
                Bottleneck(80, 112, 3, 6, 1, True, hard_swish),
                Bottleneck(112, 112, 3, 6, 1, True, hard_swish),
                Bottleneck(112, 160, 5, 6, 2, True, hard_swish),
                Bottleneck(160, 160, 5, 6, 1, True, hard_swish),
                Bottleneck(160, 160, 5, 6, 1, True, hard_swish),
            ])
        else:  # 'small'
            bottleneck_layers.extend([
                Bottleneck(16, 16, 3, 1, 2, True, 'relu'),
                Bottleneck(16, 24, 3, 4.5, 2, False, 'relu'),
                Bottleneck(24, 24, 3, 3.67, 1, False, 'relu'),
                Bottleneck(24, 40, 5, 4, 2, True, hard_swish),
                Bottleneck(40, 40, 5, 6, 1, True, hard_swish),
                Bottleneck(40, 40, 5, 6, 1, True, hard_swish),
                Bottleneck(40, 48, 5, 3, 1, True, hard_swish),
                Bottleneck(48, 48, 5, 3, 1, True, hard_swish),
                Bottleneck(48, 96, 5, 6, 2, True, hard_swish),
                Bottleneck(96, 96, 5, 6, 1, True, hard_swish),
                Bottleneck(96, 96, 5, 6, 1, True, hard_swish),
            ])
        return models.Sequential(bottleneck_layers)

    def call(self, x):
        x = self.init_conv(x)
        x = self.bottlenecks(x)
        x = self.final_layers(x)
        return x

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

mobNet_wcw_model = MobileNetV3(input_shape=(224, 224, 1), model_type='large')
mobNet_wcw_model.build(input_shape=(None, 224, 224, 1))

mobNet_wcw_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
wcw_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='/WACV_Paper/Models_RAW/HIGH_MobileNet_RAW_wCW.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
wcw_model_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, restore_best_weights=True)

wcw_history = mobNet_wcw_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[wcw_model_checkpoint, wcw_model_early_stopping])

wcw_history_df = pd.DataFrame(wcw_history.history)
wcw_history_df.to_csv('/WACV_Paper/History_RAW/HIGH_MobileNet_RAW_wCW.csv', index=False)


##########################################################################################################################################################################
##########################################################################################################################################################################
                                                            ## With Class Weight
##########################################################################################################################################################################
##########################################################################################################################################################################

# ng = len(total_patches[total_labels == 0])
# ga =  len(total_patches[total_labels == 1])
# total = ng + ga

# imbalance_ratio = ng / ga  
# weight_for_0 = (1 / ng) * (total / 2.0)
# weight_for_1 = (1 / ga) * (total / 2.0)
# class_weight = {0: weight_for_0, 1: weight_for_1}

# print('Weight for class 0 (Non-ghosting): {:.2f}'.format(weight_for_0))
# print('Weight for class 1 (Ghosting): {:.2f}'.format(weight_for_1))

# # opt = Adam(learning_rate=2e-05)

# mobNet_cw_model = MobileNetV3(input_shape=(224, 224, 1), model_type='large')
# mobNet_cw_model.build(input_shape=(None, 224, 224, 1))

# mobNet_cw_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# cw_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='/WACV_Paper/Models_RAW/MobileNet_Diff_CW.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
# cw_model_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, restore_best_weights=True)

# cw_history = mobNet_cw_model.fit(X_train, y_train, epochs=50, class_weight=class_weight, validation_data=(X_val, y_val), callbacks=[cw_model_early_stopping, cw_model_checkpoint])

# cw_history_df = pd.DataFrame(cw_history.history)
# cw_history_df.to_csv('/WACV_Paper/History/MobileNet_Diff_CW.csv', index=False)
##########################################################################################################################################################################
##########################################################################################################################################################################
                                                                    # Testing
##########################################################################################################################################################################
##########################################################################################################################################################################

# X_test = np.array(X_test)

##########################################################################################################################################################################
## Without Class Weight

test_loss, test_acc = mobNet_wcw_model.evaluate(X_test, y_test)
test_acc  = test_acc *100

predictions = mobNet_wcw_model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

precision, recall, _ = precision_recall_curve(y_test, predictions[:, 0])

pr_data = pd.DataFrame({'Precision': precision, 'Recall': recall })
file_path = '/WACV_Paper/Plots_CSV/HIGH_MobileNet_RAW_wCW_PR_Curve.csv'
pr_data.to_csv(file_path, index=False)


plt.figure()
plt.plot(recall, precision, linestyle='-', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
precision_recall_curve_path = '/WACV_Paper/Plots_RAW/HIGH_MobileNet_RAW_wCW_PR_Curve.png'

if not os.path.exists(os.path.dirname(precision_recall_curve_path)):
    os.makedirs(os.path.dirname(precision_recall_curve_path))

plt.savefig(precision_recall_curve_path, dpi=300)
plt.close()


report = classification_report(y_test, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

conf_matrix = confusion_matrix(y_test, predicted_labels)
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

macro_precision = (precision_0 + precision_1) / 2
macro_recall = (recall_0 + recall_1) / 2

if macro_precision + macro_recall > 0:
    macro_f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
else:
    macro_f1_score = 0
  
macro_f1_score  = macro_f1_score * 100
macro_precision = macro_precision * 100
macro_recall    = macro_recall * 100


TP_0 = total_class_0 * recall_0
TP_1 = total_class_1 * recall_1
FP_0 = total_class_0 * (1 - precision_0)
FP_1 = total_class_1 * (1 - precision_1)
FN_0 = total_class_0 * (1 - recall_0)
FN_1 = total_class_1 * (1 - recall_1)

micro_precision = (TP_0 + TP_1) / (TP_0 + TP_1 + FP_0 + FP_1)
micro_recall = (TP_0 + TP_1) / (TP_0 + TP_1 + FN_0 + FN_1)

if micro_precision + micro_recall > 0:
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
else:
    micro_f1_score = 0


micro_f1_score  = micro_f1_score * 100
micro_precision = micro_precision * 100
micro_recall    = micro_recall * 100

model_name = "MobileNet"
technique = "Without Class Weight"
save_metric_details(model_name, technique, test_acc, weighted_precision, weighted_recall, weighted_f1_score, macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)
print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Macro Precision: {macro_precision:.4f}, Macro Recall={macro_recall:.4f}, Macro F1-score={macro_f1_score:.4f}, Micro precision: {micro_precision:.4f}, Micro Recall={micro_recall:.4f}, Micro F1-score={micro_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")

##########################################################################################################################################################################

## With Class Weight

# test_loss, test_acc = mobNet_cw_model.evaluate(X_test, y_test)
# test_acc  = test_acc *100

# predictions = mobNet_cw_model.predict(X_test)
# predicted_labels = np.argmax(predictions, axis=1)

# precision, recall, _ = precision_recall_curve(y_test, predictions[:, 0])

# pr_data = pd.DataFrame({'Precision': precision, 'Recall': recall })
# file_path = '/WACV_Paper/Plots_RAW/MobileNet_Diff_CW_PR_Curve.csv'
# pr_data.to_csv(file_path, index=False)

# plt.figure()
# plt.plot(recall, precision, linestyle='-', color='b')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.grid(True)
# precision_recall_curve_path = '/WACV_Paper/Plots_RAW/MobileNet_Diff_CW_PR_Curve.png'

# if not os.path.exists(os.path.dirname(precision_recall_curve_path)):
#     os.makedirs(os.path.dirname(precision_recall_curve_path))

# plt.savefig(precision_recall_curve_path, dpi=300)
# plt.close()


# report = classification_report(y_test, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

# conf_matrix = confusion_matrix(y_test, predicted_labels)
# TN = conf_matrix[0, 0]
# FP = conf_matrix[0, 1]
# FN = conf_matrix[1, 0]
# TP = conf_matrix[1, 1]

# total_class_0 = TN + FP
# total_class_1 = TP + FN
# correctly_predicted_0 = TN
# correctly_predicted_1 = TP


# accuracy_0 = (TN / total_class_0) * 100
# accuracy_1 = (TP / total_class_1) * 100

# precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
# recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
# precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


# weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
# weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

# if weighted_precision + weighted_recall > 0:
#     weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
# else:
#     weighted_f1_score = 0

# weighted_f1_score  = weighted_f1_score*100
# weighted_precision = weighted_precision*100
# weighted_recall    = weighted_recall*100


# model_name = "MobileNet"
# feature_name = "Difference Map"
# technique = "Class Weight"
# save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)
# print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")
