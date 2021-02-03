# Light Weight Model Trainer
一个用于跑轻量化模型Baseline的Trainer

## 添加自定义的网络
在```config```目录下新增网络结构
```python
import tensorflow as tf

class MyNet:
    name = "[网络训练的名称]"
    height = [输入数据的高度]
    width = [输入数据的宽度]
    epochs = [训练的次数]

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def init_model(self):
        model = [自定义网络结构]
        self.model = model

```
在```train.py```中引入网络
```python
# train.py
from config.[自定义网络的名称] import MyNet
```
## 可视化训练
```sh
tensorboard --logdir logs/fit 
```
将```logs/fit```替换成在```train.py```中设置的log目录
