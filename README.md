# TIF Gráfica: rompecabezas y reconocimiento

Proyecto académico que recibe una imagen, la divide en piezas, las desordena,
reconstruye la imagen y reconoce su contenido.

## Flujo

`imagen → normalización → piezas → mezcla → reconstrucción → clasificación`

El proyecto soporta cuadrículas regulares, polígonos irregulares con Voronoi,
reconstrucción automática y clasificación mediante transfer learning.

## Instalación

TensorFlow requiere Python 3.10 a 3.13. Para este proyecto se recomienda Python
3.13:

```bash
python3.13 -m venv .venv-tf
source .venv-tf/bin/activate
pip install -r requirements-dev.txt
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generar desde una imagen local

```bash
python -m src.data_generation.generate_dataset generate \
  --input data/raw/dbz.jpg \
  --rows 3 --cols 3 --sides 4 --seed 42 \
  --output data/generated/test_001
```

La salida contiene `original.png`, las piezas PNG, `shuffled_puzzle.png`, `shuffled_puzzle_debug.png` y `metadata.json`.

Formas disponibles:

- `--sides 0`: polígonos irregulares con Voronoi.
- `--sides 3`: dos triángulos por cada celda.
- `--sides 4`: un cuadrilátero por cada celda.
- `--sides 6`: dos hexágonos irregulares por cada celda.

Los triángulos y hexágonos usan transparencia para conservar sus lados rectos sin perder partes de la imagen.

## Descargar imágenes abiertas

```bash
python -m src.data_generation.generate_dataset download-images \
  --query "landscape photo" \
  --limit 3 \
  --output data/raw/openverse_landscape
```

## Generar datos en lote

```bash
python -m src.data_generation.generate_dataset generate-batch \
  --input-dir data/raw/openverse_landscape \
  --rows 3 --cols 3 --sides 0 --seed 100 \
  --output data/generated/openverse_landscape
```

## Estado breve

- Generador de piezas regulares e irregulares: disponible.
- Voronoi: implementado.
- Solver automático: incorporado desde `origin/main`.
- Clasificación MobileNetV2: implementada y probada.
- Clasificación ResNet18: alternativa incorporada.
- Interfaz y prueba integral: pendientes de verificación.

## Datos para clasificación

El clasificador final utilizará cuatro clases: `animales`, `frutas`, `objetos` y
`personas`. Antes de entrenar, el dataset debe tener esta estructura:

```text
data/classification/
├── train/{animales,frutas,objetos,personas}/
├── validation/{animales,frutas,objetos,personas}/
└── test/{animales,frutas,objetos,personas}/
```

Para validar las carpetas y contar las imágenes:

```bash
python -m src.content_classification.dataset \
  --dataset-dir data/classification \
  --output metadata/classification_dataset.json
```

Las imágenes de referencia entregadas para evaluar el rompecabezas no deben
mezclarse con el dataset de entrenamiento.

## Dataset desde COCO

Descargar y extraer las anotaciones oficiales de COCO 2017:

```bash
mkdir -p data/coco
curl -L https://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  -o data/coco/annotations_trainval2017.zip
unzip data/coco/annotations_trainval2017.zip -d data/coco
```

Crear un subconjunto balanceado de 400 imágenes por clase:

```bash
python -m src.content_classification.coco_dataset \
  --annotations data/coco/annotations/instances_train2017.json \
  --output-dir data/classification \
  --images-per-class 400
```

El comando asigna cada imagen a su clase dominante y crea las particiones
`train`, `validation` y `test` con proporciones 70 %, 15 % y 15 %.

## Entrenamiento

```bash
python -m src.content_classification.trainer \
  --dataset-dir data/classification \
  --output-dir models/content_classification \
  --epochs 10
```

El entrenamiento guarda el modelo, las etiquetas y un JSON con las métricas en
la carpeta indicada por `--output-dir`.
