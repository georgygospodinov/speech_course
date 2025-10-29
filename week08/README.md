# Домашнее задание #3: Обучение Speaker Encoder с Angular Margin Loss

- в файле [speaker_recognition.py](./speaker_recognition.py) реализован бейзлайн пайплайна обучения
- вам необходимо:
    - скачать и распаковать [train](https://disk.yandex.ru/d/HBEZ7MbCepdgzw) и [dev](https://disk.yandex.ru/d/LKDitPbiU5XS2A) части датасета
    - реализовать класс `AngularMarginSoftmax`
    - реализовать метод `evaluate` для подсчета `Equal Error Rate`
    - обучить модель с AngularMarginSoftmax, добиться сходимости `train top1 accuracy > 0.99`, `dev EER < 4.5%`
    - подготовить отчет о проделанной работе (с графиками обучения, подобранными параметрами, кодом)

**Дедлайн:** 19 ноября 2025, 23:59 (МСК)  
После указанного срока итоговый балл умножается на **0.7**.