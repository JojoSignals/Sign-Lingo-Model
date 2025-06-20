from keras.applications import MobileNetV2
from keras import layers, models

def create_asl_model(input_shape=(300, 300, 3), num_classes=26):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
