# TIF Gráfica: rompecabezas y reconocimiento

Proyecto académico que recibe una imagen, la divide en piezas, las desordena y, en fases posteriores, reconstruirá la imagen y reconocerá su contenido.

## Flujo

`imagen → normalización → piezas → mezcla → reconstrucción futura → clasificación futura`

La fase actual trabaja únicamente en cómo crear las piezas del rompecabezas. El primer prototipo usa una cuadrícula regular porque permite validar el recorte y la mezcla antes de experimentar con Voronoi.

## Instalación

TensorFlow requiere Python 3.10 a 3.13. Para este proyecto se recomienda Python
3.13:

```bash
python3.13 -m venv .venv-tf
source .venv-tf/bin/activate
pip install -r requirements-dev.txt
```

## Uso

```bash
python -m src.data_generation.generate_dataset generate \
  --input data/raw/dbz.jpg \
  --rows 3 --cols 3 --sides 4 --seed 42 \
  --output data/generated/test_001
```

La salida contiene la imagen procesada, las piezas PNG, dos versiones del rompecabezas desordenado y sus metadatos JSON.

Formas disponibles:

- `--sides 3`: dos triángulos por cada celda.
- `--sides 4`: un cuadrilátero por cada celda.
- `--sides 6`: dos hexágonos irregulares por cada celda.

Los triángulos y hexágonos usan transparencia para conservar sus lados rectos sin perder partes de la imagen.

## Estado breve

- Generador de piezas de 3, 4 y 6 lados: disponible para revisión.
- Voronoi: alternativa pendiente de evaluar, no implementada todavía.
- Solver automático, TensorFlow e interfaz: fuera del trabajo actual.

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
