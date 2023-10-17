

## Домашнее задание

Ансамбль CTC/LAS-конформеров:
* скачать те же [данные](https://drive.google.com/file/d/1TEOR60JXgOkPrC6jSLhuR2Nb6eCegjpd/view?usp=sharing), что использовались ранее и распоковать их в `../asr/data`
* оценить Word Error Rate каждого из трех конформеров, веса для которых нужно скачать по [ссылке](https://disk.yandex.ru/d/xKGWdFPGlo6saA) и распаковать в `../asr/data`
* сагрегировать предсказания трех моделей с помощью ROVER'а, оценить WER
* выбрать лучшую гипотезу из трех с помощью MBR-decoding, оценить WER
* может пригодиться [ноутбук](./asr_ensemble.ipynb)
