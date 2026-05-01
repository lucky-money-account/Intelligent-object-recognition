import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tensorflow import keras
from backend.config import DIGIT_MODEL_PATH, DIGIT_MODEL_WEIGHTS_PATH, SAVED_MODELS_DIR
from backend.models import _create_digit_model


def train_digit_model():
    print('=' * 50)
    print('MNIST 数字识别模型训练')
    print('=' * 50)

    print('\n[1/4] 加载 MNIST 数据集...')
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f'  训练集: {x_train.shape[0]} 张图片')
    print(f'  测试集: {x_test.shape[0]} 张图片')

    print('\n[2/4] 创建模型...')
    model = _create_digit_model()
    model.summary()

    print('\n[3/4] 训练模型 (10 epochs)...')
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=1
    )

    print('\n[4/4] 评估模型...')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'  测试准确率: {test_acc * 100:.2f}%')

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model.save(DIGIT_MODEL_PATH)
    print(f'\n模型已保存到: {DIGIT_MODEL_PATH}')
    print('=' * 50)
    print('训练完成!')
    return model


if __name__ == '__main__':
    train_digit_model()
