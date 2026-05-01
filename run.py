import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app
from backend.models import preload_models

if __name__ == '__main__':
    print('=' * 45)
    print('  智能对象识别系统 启动中 ...')
    print('=' * 45)

    print('\n  [服务器] 后端服务启动，端口 5000')
    preload_models()
    print('  [模型] 后台加载中，首次识别时自动可用')
    print('\n  打开浏览器访问: http://127.0.0.1:5000')
    print('=' * 45 + '\n')

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
