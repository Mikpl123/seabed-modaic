# seabed-modaic: карта морского дна из кадров

Из папки кадров программа собирает одну карту (PNG и GeoTIFF), рисует путь аппарата и сохраняет его в GeoJSON/CSV.

## Структура проекта
```
seabed-modaic/
  config.json          настройки (меняет пользователь или интерфейс)
  params_schema.json   описание настроек для интерфейса: подписи, границы, группы
  requirements.txt
  src/
    mosaic_orb.py      основная программа
    geo_check.py       проверка траектории COLMAP и петель (для тестировщика)
    extract_frames.py  нарезка кадров из видео
  dataset/             кадры (png, jpg, tif)
  output/              результат
  colmap/              необязательные файлы COLMAP
```

## Запуск
```
pip install -r requirements.txt
python src/mosaic_orb.py                       # настройки из config.json
python src/mosaic_orb.py --config мой.json     # другой файл настроек
```

## Настройки
- Все параметры лежат в `config.json`. Относительные пути считаются от корня проекта.
- Не указанные параметры берут значения по умолчанию из кода.
- Перед обработкой настройки проверяются; при ошибке программа ничего не считает и пишет, что исправить (например, «это не матрица петель, возможно выбран файл траектории», «система координат должна быть в метрах»).
- `LOOP_FILE` (матрица петель COLMAP) и `NAV_FILE` (траектория COLMAP) необязательны. Без них повторные проходы ищутся по виду. Навигацию на Aqualoc лучше не включать: она ухудшает локальные связи.

## Как вызывать из интерфейса
```python
import sys; sys.path.append("src")
import mosaic_orb

mosaic_orb.LOG = lambda *a, **k: show_in_log(" ".join(map(str, a)))   # сообщения о ходе работы
cfg = {"SRC": "dataset", "OUT": "output", "RENDER_STEP": 0.45, "LOOP_FILE": None}  # значения из полей интерфейса
try:
    keys, results = mosaic_orb.main(cfg)      # запускать в отдельном потоке
except mosaic_orb.ConfigError as e:
    show_errors(e.problems)                   # список понятных сообщений
```
- Подписи, границы и группы полей берите из `params_schema.json` (поле `advanced` = прятать во вкладке «Дополнительно»).
- Кнопки «Стоп» пока нет.
- Размер кеша кадров и параметры детектора читаются при импорте и из настроек не меняются.

## Результат (папка output)
`<имя>_part0_nN.png` карта, `..._raw.png` без улучшения, `.tif` GeoTIFF, `.json` параметры, `..._traj.png|geojson|csv` путь аппарата, `poses_*.json` и `debug_labels_*.png` служебные файлы.
Если координаты не заданы (`GEO_EPSG`, `GEO_MPP`), карта и путь в условных пикселях.
