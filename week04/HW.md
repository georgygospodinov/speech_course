# EBranchformer-CTC

в этом домашнем задании предлагается имплементировать E-Branchformer Encoder из статьи:
[E-BRANCHFORMER: BRANCHFORMER WITH ENHANCED MERGING FOR SPEECH RECOGNITION](https://arxiv.org/pdf/2210.00077)

вам понадобятся данные и предобученный чекпоинт: [farfield-golos(~3Gb)](https://drive.google.com/file/d/1iQd89RCOuGryVoDzyWCnSulaj4kcm-I8/view?usp=sharing)



если удобнее запускать код в docker'е:
```bash
$ docker pull nvcr.io/nvidia/nemo:23.06
$ docker run -it -v <repo_path>/asr/:/home/asr nvcr.io/nvidia/nemo:23.06
$ cd /home/asr
$ export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 && python run_quartznet_ctc.py trainer.devices=6 trainer.accelerator=gpu ++trainer.strategy=ddp
```

Что требуется сделать:
* Перейти в `<repo_path>/asr`, где написана большая часть кода для обучения
* скачать [данные](https://drive.google.com/file/d/1iQd89RCOuGryVoDzyWCnSulaj4kcm-I8/view?usp=sharing) (можно скачать с помощью `download_data.py`) и распоковать их в `./data`
* разобраться в коде в `train_model.py` и в `src/`
* реализовать классы в [`src/encoders/e_branchformer.py`](../asr/src/encoders/e_branchformer.py)
* проверить, что у реализованного энкодера число параметров совпадает со значением в статье:
```python
sum(p.numel() for p in model.encoder.parameters())
11266850
```
* запустить обучение: `python train_model.py --config-name ebranchformer_ctc`
* увидеть подобное в логе:
```log
  | Name     | Type                 | Params
--------------------------------------------------
0 | encoder  | EBranchformerEncoder | 11.3 M
1 | decoder  | ConvDecoder          | 8.7 K 
2 | wer      | WER                  | 0     
3 | ctc_loss | CTCLoss              | 0     
--------------------------------------------------
10.0 M    Trainable params
1.3 M     Non-trainable params
11.3 M    Total params
45.067    Total estimated model params size (MB)
[2023-10-03 13:00:07,101][lightning][INFO] - reference : салют хватит
[2023-10-03 13:00:07,101][lightning][INFO] - prediction: й
```
* запустить обучение с предобученного чекпоинта: `python train_model.py --config-name ebranchformer_ctc init_weights=<absolute_path>/data/checkpoints_and_tokenizer/ebranchformer_ckpt.ckpt`
* увидеть подобное в логе: 
```log
[2023-10-03 12:58:59,306][lightning][INFO] - reference : салют вызов светлане васильевне николенко
[2023-10-03 12:58:59,307][lightning][INFO] - prediction: салют вызод светлане васильевну ниталенко
[2023-10-03 12:58:59,344][lightning][INFO] - reference : джой звонок юрию ивановичу царькову
[2023-10-03 12:58:59,344][lightning][INFO] - prediction: джой званот юрию ивановичу зарькова
```

* посчитать Word Error Rate на датасетах `./data/test_opus/crowd/manifest.jsonl` и `./data/test_opus/farfield/manifest.jsonl` с помощью предобученного чекпоинта: `./data/checkpoints_and_tokenizer/ebranchformer_ckpt.ckpt`
* если у вас есть gpu, с помощью датасета `./data/train_opus/manifest.jsonl` и предобученного чекпоинта улучшить качество модели на `test_opus/farfield`-данных
* если у вас нет gpu, взять случайный семпл в 10 минут (~ 100 примеров) из `./data/train_opus/manifest.jsonl` и показать, что со случайных весов модель способна на нем переобучиться