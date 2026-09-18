# PyTorch basics

Короткая памятка по базовому циклу обучения и переключению модели между режимами `train` и `eval`.

## 1. `model.train()` и `model.eval()`

В PyTorch модель может работать в двух режимах:

- `model.train()` — режим обучения.
- `model.eval()` — режим оценки / инференса.

Это особенно важно для слоёв, поведение которых зависит от режима, например `Dropout` и `BatchNorm`.

| Mode | Dropout | BatchNorm |
|---|---|---|
| `train()` | активен | обновляет статистики |
| `eval()` | выключен | использует сохранённые статистики |

### Пример

```python
model.train()

for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Для оценки:

```python
model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
```

---

## 2. Зачем нужен `optimizer.zero_grad()`

В PyTorch градиенты по умолчанию накапливаются.

Поэтому перед новым шагом обратного распространения обычно нужно очистить старые градиенты:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Если забыть `zero_grad()`, градиенты от нескольких шагов будут суммироваться.

---

## 3. Что делает `loss.backward()`

`loss.backward()` запускает обратное распространение ошибки и вычисляет градиенты параметров модели.

После этого они доступны в:

```python
parameter.grad
```

Но сами веса ещё не меняются.

Обновление параметров выполняет:

```python
optimizer.step()
```

---

## 4. Базовый цикл обучения

```python
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
```

---

## 5. Оценка модели

Во время validation/test модель не должна обновлять веса.

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
```

`torch.no_grad()` отключает вычисление градиентов, что уменьшает расход памяти и ускоряет инференс.

---

## 6. `DataLoader`

`DataLoader` разбивает датасет на батчи и может перемешивать данные.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

Обычно:

- для `train` используют `shuffle=True`;
- для `validation/test` чаще используют `shuffle=False`.

---

## 7. Размерности тензоров

Для изображений PyTorch обычно использует формат:

```text
[batch, channels, height, width]
```

Например:

```text
[32, 3, 150, 150]
```

означает:

- 32 изображения в батче;
- 3 цветовых канала;
- высота 150;
- ширина 150.

Для отображения изображения через matplotlib часто нужно переставить оси:

```python
img = image.permute(1, 2, 0)
```

Было:

```text
[C, H, W]
```

Стало:

```text
[H, W, C]
```

---

## 8. Типичные ошибки

### Забыли `model.eval()`

Если в модели есть `Dropout` или `BatchNorm`, результаты на validation/test могут быть нестабильными или некорректными.

### Забыли `torch.no_grad()`

Модель всё равно сможет делать предсказания, но будет зря строить вычислительный граф и расходовать память.

### Забыли `optimizer.zero_grad()`

Градиенты начнут накапливаться между батчами.

### Неправильная размерность перед `Linear`

После свёрточных слоёв нужно проверить размер тензора перед `Flatten` / `Linear`.

Полезно временно вывести:

```python
print(x.shape)
```

---

## 9. Быстрая памятка

```text
TRAIN:
model.train()
optimizer.zero_grad()
outputs = model(x)
loss = criterion(outputs, y)
loss.backward()
optimizer.step()

EVAL:
model.eval()
with torch.no_grad():
    outputs = model(x)
```

---

## 10. Мини-проверка

Если `train accuracy` растёт, а `validation/test accuracy` падает, это может быть признаком переобучения.

Что можно проверить:

- augmentation;
- dropout;
- weight decay;
- сложность модели;
- learning rate;
- количество эпох;
- качество и баланс датасета.
