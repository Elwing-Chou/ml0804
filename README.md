# 桃園

## 讀取

```python
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy

if not os.path.exists("drive/My Drive/mlp.h5"):
    layers = [
        # 784 * 128 + 128(bias)
        Dense(256, activation="relu", input_dim=784),
        # 128 * 10(連線個數) + 10(bias)
        Dense(10, activation="softmax")
    ]
    model = Sequential(layers)
    model.compile(loss=CategoricalCrossentropy(),
       optimizer="adam",
       metrics=["accuracy"])
else:
    print("Loading...")
    model = load_model("drive/My Drive/mlp.h5") 
# fit
```



## Word2Vec

### 資料集

[PTT小資料集](https://drive.google.com/open?id=1BT4h4-kzrtCS_52P2i7C1rlj1GZgEbs6)

[PTT大資料集](https://drive.google.com/open?id=15byko6d_9VJGPOW7DPAN8YiVsleRiURr)

### 標點符號去除

```python
punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…~/ －＊➜■─★☆=@<>◉é''')
filter(lambda x: x not in punct, jieba.cut(content))
```

### 網址Regex

```python
content = re.sub(r'https?:\/\/.*[\r\n]*', '', content)
```

