# Reconstrucción de rompecabezas y clasificación

Aplicación de línea de comandos que reconstruye un rompecabezas 2D y clasifica
la imagen como `animales`, `frutas`, `objetos` o `personas`.

Utiliza OpenCV para reconstrucción y MobileNetV2 con transfer learning para
clasificación.

## Requisitos

- Python 3.13.
- Modelo incluido en `models/content_classification_coco_500/`.

## Instalación

### macOS o Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Comprobar la instalación:

```bash
python -c "import cv2, numpy, tensorflow; print('Instalación correcta')"
```

## Reconstruir y clasificar

Las piezas de ejemplo están en `input/pieces/imagen1/`.

```bash
python app.py solve
```

Para otra carpeta:

```bash
python app.py solve "/ruta/a/las/piezas"
```

Resultados:

```text
results/imagen1/
├── reconstructed.png
├── reconstructed_debug.png
├── assembly_solution.json
├── reconstruction_report.json
└── classification_result.json
```

## Clasificar una imagen completa

No es necesario convertir una imagen completa en piezas:

```bash
python app.py classify "/ruta/a/imagen.jpg"
```

El comando muestra la clase y confianza, y guarda un JSON en `results/`.

## Evaluar la reconstrucción

```bash
python app.py evaluate \
  data/evaluation/imagen1/imagen_referencia_512.png \
  results/imagen1/reconstructed.png
```

La referencia se usa solamente para evaluación, nunca para resolver el puzzle.

## Modelo entrenado

```text
models/content_classification_coco_500/
├── mobilenetv2_classifier.keras
├── labels.json
└── training_results.json
```

El archivo `.keras` contiene la arquitectura y los pesos entrenados. Para
probar la aplicación no hace falta descargar COCO ni volver a entrenar.

El modelo se construyó mediante transfer learning:

1. MobileNetV2 inició con pesos de ImageNet.
2. Se eliminó su clasificador original de 1000 categorías.
3. Se congeló la base convolucional.
4. Se agregó una salida de cuatro clases.
5. Se entrenó con 2000 imágenes seleccionadas de COCO.

Distribución del dataset utilizado:

```text
Entrenamiento: 350 imágenes por clase
Validación:     75 imágenes por clase
Prueba:         75 imágenes por clase
Total:         500 imágenes por clase
```

Exactitud final en el conjunto de prueba:

```text
92.33 %
```

## Volver a entrenar

Esta parte es opcional y requiere un dataset con la estructura:

```text
dataset/
├── train/{animales,frutas,objetos,personas}/
├── validation/{animales,frutas,objetos,personas}/
└── test/{animales,frutas,objetos,personas}/
```

Comando:

```bash
python -m src.content_classification.trainer \
  --dataset-dir "/ruta/al/dataset" \
  --output-dir models/content_classification_coco_500 \
  --epochs 10 \
  --batch-size 16 \
  --weights imagenet
```

## Dependencias

El proyecto utiliza un solo archivo:

```text
requirements.txt
```

Dependencias:

- NumPy.
- OpenCV.
- TensorFlow.

PyTorch, pytest, COCO y Openverse no son necesarios para ejecutar la entrega.
