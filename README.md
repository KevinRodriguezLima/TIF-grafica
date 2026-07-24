# Reconstrucción de rompecabezas y clasificación de objetos

Proyecto que recibe piezas de una imagen, reconstruye el rompecabezas y
clasifica el contenido final como `animales`, `frutas`, `objetos` o `personas`.

El clasificador utiliza MobileNetV2, pesos iniciales de ImageNet y transfer
learning con un subconjunto balanceado de COCO.

## Flujo completo

```text
piezas PNG
→ segmentación
→ contornos y bordes
→ parejas compatibles
→ búsqueda global de cuadrícula
→ imagen reconstruida
→ MobileNetV2
→ clase y confianza
```

## Requisitos

- Python 3.13 recomendado.
- Aproximadamente 4 GB libres para entorno, modelo y datos.
- macOS, Linux o Windows.

`requirements.txt` instala las dependencias de ejecución:

- `numpy`: operaciones numéricas.
- `opencv-python`: procesamiento y renderizado de imágenes.
- `tensorflow`: MobileNetV2 y transfer learning.
- `torch` y `torchvision`: modelos alternativos ResNet18 y red siamesa.

`requirements-dev.txt` agrega `pytest` para ejecutar las pruebas.

## Instalación

### macOS o Linux

```bash
python3.13 -m venv .venv-tf
source .venv-tf/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para ejecutar las pruebas:

```bash
python -m pip install -r requirements-dev.txt
```

### Windows PowerShell

```powershell
py -3.13 -m venv .venv-tf
.\.venv-tf\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para ejecutar las pruebas:

```powershell
python -m pip install -r requirements-dev.txt
```

Verificar la instalación:

```bash
python -c "import cv2, numpy, tensorflow; print('Dependencias correctas')"
```

## Archivos necesarios

Los datasets, modelos y resultados están ignorados por Git debido a su tamaño.
Para usar el modelo entrenado deben existir juntos:

```text
models/content_classification_coco_500/
├── mobilenetv2_classifier.keras
└── labels.json
```

Si esos archivos no están disponibles, se deben generar siguiendo la sección
de entrenamiento.

Las piezas entregadas deben tener esta estructura:

```text
input/pieces/imagen1/
├── pieza_01.png
├── pieza_02.png
└── ...
```

La referencia, utilizada únicamente para evaluar, debe estar en:

```text
data/evaluation/imagen1/imagen_referencia_512.png
```

## Clasificar una imagen completa

Una imagen completa no necesita convertirse en piezas:

```bash
python -m src.content_classification.classifier \
  --image "/ruta/a/imagen.jpg" \
  --model models/content_classification_coco_500/mobilenetv2_classifier.keras \
  --output results/clasificacion_imagen.json
```

El comando imprime las cuatro probabilidades y guarda el resultado. Para
consultarlo nuevamente:

```bash
python -m json.tool results/clasificacion_imagen.json
```

## Ejecutar reconstrucción y clasificación

```bash
python -m src.main run-all \
  --pieces-dir input/pieces/imagen1 \
  --processed-dir processed/imagen1 \
  --metadata-dir metadata/imagen1 \
  --results-dir results/imagen1 \
  --puzzle-id imagen1 \
  --content-classifier-model models/content_classification_coco_500/mobilenetv2_classifier.keras
```

Archivos principales generados:

```text
results/imagen1/
├── assembly_solution.json
├── reconstructed.png
├── reconstructed_debug.png
├── reconstruction_report.json
└── classification_result.json
```

Ver la clasificación:

```bash
python -m json.tool results/imagen1/classification_result.json
```

Ver solamente la clase:

```bash
python -c "import json; print(json.load(open('results/imagen1/classification_result.json'))['accepted_prediction'])"
```

## Evaluar la reconstrucción

La referencia no interviene en el ensamblaje. Se utiliza después para medir el
resultado:

```bash
python -m src.rendering.reconstruction_evaluator \
  --reference data/evaluation/imagen1/imagen_referencia_512.png \
  --reconstruction results/imagen1/reconstructed.png \
  --output results/imagen1/reconstruction_evaluation.json
```

El reporte incluye tamaño, MAE, MSE, PSNR y similitud. Para `imagen1` se obtuvo
aproximadamente 97.97 % de similitud.

## Crear piezas desde una imagen

Cuadrícula regular de 3 filas y 5 columnas:

```bash
python -m src.data_generation.generate_dataset generate \
  --input "/ruta/a/imagen.png" \
  --rows 3 --cols 5 --sides 4 --seed 42 \
  --output data/generated/mi_rompecabezas
```

Piezas irregulares con Voronoi:

```bash
python -m src.data_generation.generate_dataset generate \
  --input "/ruta/a/imagen.png" \
  --rows 3 --cols 5 --sides 0 --seed 42 \
  --output data/generated/mi_rompecabezas_voronoi
```

Valores disponibles:

- `--sides 0`: Voronoi.
- `--sides 3`: triángulos.
- `--sides 4`: cuadriláteros.
- `--sides 6`: hexágonos.

## Dataset de clasificación

El dataset tiene cuatro carpetas dentro de cada partición:

```text
dataset/
├── train/{animales,frutas,objetos,personas}/
├── validation/{animales,frutas,objetos,personas}/
└── test/{animales,frutas,objetos,personas}/
```

Validarlo:

```bash
python -m src.content_classification.dataset \
  --dataset-dir data/sources/coco \
  --output metadata/coco_summary.json
```

El dataset final contiene:

```text
train:      350 imágenes por clase
validation:  75 imágenes por clase
test:        75 imágenes por clase
total:      500 imágenes por clase, 2000 imágenes
```

## Crear el dataset desde COCO

Descargar y extraer las anotaciones COCO 2017:

```bash
mkdir -p data/coco
curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  -o data/coco/annotations_trainval2017.zip
unzip data/coco/annotations_trainval2017.zip -d data/coco
```

Crear 500 imágenes por clase:

```bash
python -m src.content_classification.coco_dataset \
  --annotations data/coco/annotations/instances_train2017.json \
  --output-dir data/sources/coco \
  --images-per-class 500 \
  --min-dominance 0.80
```

El filtro del 80 % exige que una clase domine el área anotada.

## Recopilar desde Openverse

Openverse es una fuente adicional. Sus resultados deben revisarse porque las
etiquetas provienen de búsquedas de texto:

```bash
python -m src.content_classification.openverse_collector \
  --output-dir data/sources/openverse \
  --images-per-query 50 \
  --class-name animales
```

Clases permitidas:

```text
animales, frutas, objetos, personas
```

El recolector conserva autor, licencia y URL, y puede reanudar descargas. La
API puede responder `429 Too Many Requests` al superar la cuota anónima.

## Unir fuentes y eliminar duplicados

```bash
python -m src.content_classification.dataset_merger \
  --source data/sources/coco \
  --source data/sources/openverse \
  --output-dir data/classification_merged \
  --images-per-class 500
```

## Entrenar MobileNetV2

```bash
python -m src.content_classification.trainer \
  --dataset-dir data/sources/coco \
  --output-dir models/content_classification_coco_500 \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --patience 3 \
  --weights imagenet
```

Archivos generados:

```text
models/content_classification_coco_500/
├── mobilenetv2_classifier.keras
├── labels.json
└── training_results.json
```

Ver las métricas:

```bash
python -m json.tool \
  models/content_classification_coco_500/training_results.json
```

El entrenamiento final obtuvo aproximadamente 92.33 % de exactitud en test.

## Transfer learning utilizado

1. MobileNetV2 carga pesos aprendidos con ImageNet.
2. Se elimina la salida original de 1000 categorías.
3. Se congela la base convolucional.
4. Se agrega una salida `Dense(4, softmax)`.
5. Esa salida se entrena con las cuatro clases creadas desde COCO.

## Pruebas

```bash
python -m pytest -q
```

Actualmente deben aprobar 29 pruebas.

## Problemas frecuentes

### `No module named tensorflow`

```bash
source .venv-tf/bin/activate
python -m pip install -r requirements.txt
```

### No encuentra `labels.json`

`labels.json` debe estar en la misma carpeta que el archivo `.keras`.

### Openverse responde `429`

Se alcanzó la cuota temporal. El recolector conserva el avance y se puede
ejecutar nuevamente más tarde.

### La clasificación devuelve resultados extraños

Confirmar que se utiliza el modelo:

```text
models/content_classification_coco_500/mobilenetv2_classifier.keras
```

No normalizar manualmente la imagen: el modelo incluye internamente el
preprocesamiento de MobileNetV2.
