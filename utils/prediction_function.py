import tensorflow as tf
import numpy as np
import cv2
import os
model_path = 'tf_model'
model = tf.saved_model.load(model_path)

classes = ["Pattern", "Solid"]

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.float32)
    image_resized = tf.image.resize(image_tensor, [224, 224])
    image_normalized = image_resized / 255.0
    image_transposed = tf.transpose(image_normalized, perm=[2, 0, 1])
    image_batched = image_transposed[tf.newaxis, :]
    return image_batched

def predict_image(image):
    inputs = {'pixel_values': preprocess_image(image)}
    outputs = model(inputs)
    class_scores = outputs['logits'].numpy().flatten()
    softmax_scores = tf.nn.softmax(class_scores).numpy()
    percentages = softmax_scores * 100
    # formatted_percentages = [f"{p:.2f}%" for p in percentages]
    formatted_percentages = {classes[i]: f"{percentages[i]:.2f}%" for i in range(len(classes))}
    return formatted_percentages, classes[np.argmax(percentages)]
