# QuartzNet5x5-CTC

в этом домашнем задании предлагается имплементировать QuartzNet Encoder из статьи:
[QUARTZNET: DEEP AUTOMATIC SPEECH RECOGNITION WITH 1D TIME-CHANNEL SEPARABLE CONVOLUTIONS](https://arxiv.org/pdf/1910.10261.pdf)

вам понадобятся данные и предобученный чекпоинт: [farfield-golos(~3Gb)](https://drive.google.com/file/d/1TEOR60JXgOkPrC6jSLhuR2Nb6eCegjpd/view?usp=sharing)



если удобнее запускать код в docker'е:
```bash
$ docker pull nvcr.io/nvidia/nemo:23.06
$ docker run -it -v <repo_path>/asr/:/home/asr nvcr.io/nvidia/nemo:23.06
$ cd /home/asr
$ export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 && python main.py trainer.devices=6 trainer.accelerator=gpu ++trainer.strategy=ddp
```

Что требуется сделать:
* Перейти в `<repo_path>/asr`, где написана большая часть кода для обучения
* скачать [данные](https://drive.google.com/file/d/1TEOR60JXgOkPrC6jSLhuR2Nb6eCegjpd/view?usp=sharing) (можно скачать с помощью `download_data.sh`) и распоковать их в `./data`
* разобраться в коде `main.py` и `src/*.py`
* реализовать классы в [`src/encoder.py`](../asr/src/encoder.py)
* проверить, что у реализованного энкодера число параметров совпадает со значением в статье:
```python
sum(p.numel() for p in model.encoder.parameters())
6708096
```
* запустить обучение: `python main.py`
* увидеть подобное в логе:
```log
  | Name     | Type        | Params
-----------------------------------------
0 | encoder  | QuartzNet   | 6.7 M 
1 | decoder  | ConvDecoder | 34.9 K
2 | wer      | WER         | 0     
3 | ctc_loss | CTCLoss     | 0     
-----------------------------------------
6.7 M     Trainable params
0         Non-trainable params
6.7 M     Total params
26.972    Total estimated model params size (MB)
[2023-10-03 13:00:07,101][lightning][INFO] - reference : салют хватит
[2023-10-03 13:00:07,101][lightning][INFO] - prediction: й
```
* запустить обучение с предобученного чекпоинта: `python main.py model.init_weights=<absolute_path>/data/q5x5_ru_stride_4_crowd_epoch_4_step_9794.ckpt`
* увидеть подобное в логе: 
```log
[2023-10-03 12:58:59,306][lightning][INFO] - reference : салют вызов светлане васильевне николенко
[2023-10-03 12:58:59,307][lightning][INFO] - prediction: салют вызод светлане васильевну ниталенко
[2023-10-03 12:58:59,344][lightning][INFO] - reference : джой звонок юрию ивановичу царькову
[2023-10-03 12:58:59,344][lightning][INFO] - prediction: джой званот юрию ивановичу зарькова
```

* посчитать Word Error Rate на датасетах `./data/test_opus/crowd/manifest.jsonl` и `./data/test_opus/farfield/manifest.jsonl` с помощью предобученного чекпоинта: `./data/q5x5_ru_stride_4_crowd_epoch_4_step_9794.ckpt`
* если у вас есть gpu, с помощью датасета `./data/train_opus/manifest.jsonl` и предобученного чекпоинта улучшить качество модели на `test_opus/farfield`-данных
* если у вас нет gpu, взять случайный семпл в 10 минут (~ 100 примеров) из `./data/train_opus/manifest.jsonl` и показать, что со случайных весов модель способна на нем переобучиться