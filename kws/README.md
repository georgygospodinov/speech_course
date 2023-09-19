# Keyword Spotting

[Kaggle Competition](https://www.kaggle.com/competitions/keyword-spotting-mipt-2023)

Ограничения на модель:
* не более `10^4` параметров
* не более `10^6` операций сложения-умножения для записи длительностью 1 секунда

## Baseline

```bash
cd kws
python3 -m venv kws_venv
. kws_venv/bin/activate
pip install -r requirements.txt
# обучение
python run.py ++train_dataloader.dataset.manifest_path=<train_manifest> ++val_dataloader.dataset.manifest_path=<val_manifest> ++predict_dataloader.dataset.manifest_path=<test_manifest>
# анализ логов обучения в tensorboard
tensorboard --logdir ./lightning_logs
# конвертация в ONNX
python to_onnx.py ++init_weights=<path_to_model>
# анализ графа модели в netron
netron ./data/kws.onnx
# предсказания на тестовом датасете
python submit.py ++init_weights=<path_to_model>
```