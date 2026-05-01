import os
import threading
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

from .config import (
    DIGIT_MODEL_PATH,
    IMG_SIZE_GENERAL, IMG_SIZE_DIGIT,
    ANIMAL_KEYWORDS, SCENE_KEYWORDS
)

import tensorflow.keras as keras
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import models as tf_models

_pytorch_model = None
_imagenet_classes = None
_digit_model = None

_models_loaded = False
_models_loading = False
_lock = threading.Lock()

_torch_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

_HARDCODED_IMAGENET_CLASSES = None


def _get_hardcoded_classes():
    global _HARDCODED_IMAGENET_CLASSES
    if _HARDCODED_IMAGENET_CLASSES is None:
        try:
            url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
            import urllib.request
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as f:
                content = f.read().decode('utf-8')
            _HARDCODED_IMAGENET_CLASSES = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception:
            _HARDCODED_IMAGENET_CLASSES = [f'class_{i}' for i in range(1000)]
    return _HARDCODED_IMAGENET_CLASSES


def _load_pytorch_model():
    global _pytorch_model, _imagenet_classes
    if _pytorch_model is None:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        _pytorch_model = models.mobilenet_v2(weights=weights)
        _pytorch_model.eval()
        _imagenet_classes = weights.meta.get('categories', None)
        if not _imagenet_classes:
            _imagenet_classes = _get_hardcoded_classes()
    return _pytorch_model


def _create_digit_model():
    model = tf_models.Sequential([
        tf_layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf_layers.MaxPooling2D((2, 2)),
        tf_layers.Conv2D(64, (3, 3), activation='relu'),
        tf_layers.MaxPooling2D((2, 2)),
        tf_layers.Conv2D(64, (3, 3), activation='relu'),
        tf_layers.Flatten(),
        tf_layers.Dense(64, activation='relu'),
        tf_layers.Dropout(0.5),
        tf_layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def _load_digit_model():
    global _digit_model
    if _digit_model is None:
        if os.path.exists(DIGIT_MODEL_PATH):
            _digit_model = keras.models.load_model(DIGIT_MODEL_PATH)
    return _digit_model


def get_model_status():
    return {
        'loaded': _models_loaded,
        'loading': _models_loading,
        'pytorch': _pytorch_model is not None,
        'digit': _digit_model is not None,
    }


def preload_models():
    global _models_loading, _models_loaded
    with _lock:
        if _models_loading or _models_loaded:
            return
        _models_loading = True

    def _load():
        global _models_loaded, _models_loading
        try:
            _load_pytorch_model()
            _load_digit_model()
            _models_loaded = True
        except Exception:
            pass
        finally:
            _models_loading = False

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()


def _preprocess_pytorch(image_path):
    img = Image.open(image_path).convert('RGB')
    return _torch_transform(img).unsqueeze(0)


def _preprocess_digit(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize(IMG_SIZE_DIGIT, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=(0, -1))


def _filter_predictions(probs, keywords_list):
    results = []
    for idx in range(len(probs)):
        label = _imagenet_classes[idx] if idx < len(_imagenet_classes) else f'class_{idx}'
        label_lower = label.lower().replace('-', ' ').replace('_', ' ')
        for keyword in keywords_list:
            if keyword.lower() in label_lower:
                results.append({
                    'label': label.replace('_', ' ').title(),
                    'confidence': float(probs[idx])
                })
                break
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:10]


def predict_general(image_path):
    model = _load_pytorch_model()
    img = _preprocess_pytorch(image_path)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)
    results = []
    for i in range(5):
        idx = top5_idx[i].item()
        label = _imagenet_classes[idx] if idx < len(_imagenet_classes) else f'class_{idx}'
        results.append({
            'label': label.replace('_', ' ').title(),
            'confidence': float(top5_prob[i].item())
        })
    return results


def predict_digit(image_path):
    model = _load_digit_model()
    if model is None:
        raise RuntimeError('Digit model not trained. Please run train_digits.py first.')
    img = _preprocess_digit(image_path)
    preds = model.predict(img, verbose=0)
    results = []
    for i, conf in enumerate(preds[0]):
        results.append({'label': str(i), 'confidence': float(conf)})
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results


def predict_animal(image_path):
    model = _load_pytorch_model()
    img = _preprocess_pytorch(image_path)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    return _filter_predictions(probs.cpu().numpy(), ANIMAL_KEYWORDS)


def predict_scene(image_path):
    model = _load_pytorch_model()
    img = _preprocess_pytorch(image_path)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    return _filter_predictions(probs.cpu().numpy(), SCENE_KEYWORDS)


MODES = {
    'general': {
        'name': '通用物体识别',
        'description': '识别1000种常见物体，包括工具、食物、植物等',
        'icon': 'globe',
        'predict_fn': predict_general
    },
    'digit': {
        'name': '数字识别',
        'description': '识别手写数字0-9',
        'icon': 'calculator',
        'predict_fn': predict_digit
    },
    'animal': {
        'name': '动物识别',
        'description': '识别各类动物，包括哺乳动物、鸟类、鱼类、昆虫等',
        'icon': 'paw',
        'predict_fn': predict_animal
    },
    'scene': {
        'name': '场景识别',
        'description': '识别自然与城市场景，如山水、建筑、街景等',
        'icon': 'image',
        'predict_fn': predict_scene
    }
}


def get_available_modes():
    modes_info = []
    for mode_id, mode_data in MODES.items():
        info = {
            'id': mode_id,
            'name': mode_data['name'],
            'description': mode_data['description'],
            'icon': mode_data['icon']
        }
        if mode_id == 'digit' and _load_digit_model() is None:
            info['available'] = False
            info['hint'] = '请先运行训练脚本 train_digits.py'
        else:
            info['available'] = True
        modes_info.append(info)
    return modes_info


def load_all_models():
    print('  加载通用识别模型 (PyTorch MobileNetV2)...')
    _load_pytorch_model()
    print('  加载数字识别模型...')
    try:
        model = _load_digit_model()
        if model:
            print('  数字识别模型加载成功')
        else:
            print('  数字识别模型未找到，请先运行 train_digits.py')
    except Exception as e:
        print(f'  数字识别模型加载失败: {e}')
