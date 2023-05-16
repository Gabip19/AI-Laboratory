import numpy as np
import tensorflow as tf


class AbstractClassifier:

    @staticmethod
    def load_images():
        def preprocess_image(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [128, 128])
            image = image / 255.0  # Normalize pixel values between 0 and 1
            return image

        default_image_paths = []
        sepia_image_paths = []

        for i in range(1, 51):
            sepia_img_path = f"data/sepia/image-{i}.jpg"
            default_img_path = f"data/default/image-{i}.jpg"
            sepia_image_paths.append(sepia_img_path)
            default_image_paths.append(default_img_path)

        sepia_images = [preprocess_image(image_path) for image_path in sepia_image_paths]
        default_images = [preprocess_image(image_path) for image_path in default_image_paths]

        return sepia_images, default_images

    @staticmethod
    def load_data():
        sepia_images, default_images = AbstractClassifier.load_images()

        sepia_outlabels = np.ones(len(sepia_images))
        default_outlabels = np.zeros(len(default_images))

        input_data = np.concatenate((sepia_images, default_images), axis=0)
        output_data = np.concatenate((sepia_outlabels, default_outlabels), axis=0)

        return input_data, output_data

    def train_classifier(self, train_inputs, train_outputs):
        raise NotImplementedError()

    def run_classifier(self, train_inputs, train_outputs, test_inputs, test_outputs):
        raise NotImplementedError()
