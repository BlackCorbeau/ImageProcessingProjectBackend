#!/bin/bash
# Минимальный рабочий скрипт

# Установите gdown если нет
pip install gdown 2>/dev/null || pip3 install gdown 2>/dev/null

# Скачиваем через gdown (самый надежный способ)
curl -L -o face-mask-12k-images-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/ashishjangra27/face-mask-12k-images-dataset

# Распаковываем
unzip -o face-mask-12k-images-dataset.zip

# Если создалась папка MFSD, перемещаем в корень
[ -d "MFSD" ] && mv MFSD/* ./ && rmdir MFSD

# Удаляем архив
rm -f face-mask-12k-images-dataset.zip

echo "Готово! Проверяем:"
ls -d 1/ 2/ 2>/dev/null || ls | head -10
