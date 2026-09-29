# Как добавить свою модель

Первая версия использует модели COCO из MMDetection 3.3. Своё обучение сюда не встроено, но реестр уже читает дополнительные модели из этой папки.

## Куда класть файлы

- `datasets/` — изображения и разметка, например COCO JSON.
- `training/` — конфиги и журналы обучения.
- `custom_models/configs/` — файл конфигурации MMDetection, обычно скопированный из `rtmdet` и с изменённым `num_classes`.
- `checkpoints/` — файл весов `.pth`.
- `custom_models/имя.json` — запись, по которой Model Manager и список Model увидят модель.

## Пример JSON

Создайте `custom_models/printer.json`:

```json
{
  "model_id": "printer_parts",
  "display_name": "Printer Parts",
  "task": "detection",
  "profile": "accurate",
  "mmdet_name": "custom_models/configs/printer.py",
  "checkpoint_url": "",
  "checkpoint_file": "printer_parts.pth",
  "input_size": 640,
  "supports_masks": false,
  "description": "Свои классы: 3D_PRINTER, FILAMENT_SPOOL, PCB, ESP32, ARDUINO, CALIPER, SCREWDRIVER, PRINTED_PART, PACKAGE, FLOWER."
}
```

`mmdet_name` может быть именем модели из MMDetection или путём к локальному `.py` конфигу. `checkpoint_file` — имя файла внутри `checkpoints/`.

Если у модели есть маски экземпляров, поставьте `"supports_masks": true` и `"task": "instance_segmentation"`. Тогда режим Instance Segmentation сможет использовать её без второй модели детекции.

## Классы, которые имеет смысл завести следующим этапом

`3D_PRINTER`, `FILAMENT_SPOOL`, `PCB`, `ESP32`, `ARDUINO`, `CALIPER`, `SCREWDRIVER`, `PRINTED_PART`, `PACKAGE`, `FLOWER`.

Их нет в COCO. Пока в программе находятся только классы выбранной модели, для RTMDet это 80 классов COCO: person, cup, chair, laptop, keyboard, cell phone, bottle и остальные.

## Проверка

1. Положите веса в `checkpoints/`.
2. Откройте MODEL MANAGER. Своя модель должна появиться в таблице.
3. Нажмите VERIFY.
4. Выберите её в списке Model и нажмите START.

Обучение запускается обычными инструментами MMDetection (`tools/train.py`) в этом же `.venv`. Готовый конфиг и чекпоинт подключаются записью JSON, код приложения менять не нужно.
