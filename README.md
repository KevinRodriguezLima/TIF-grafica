# TIF Grafica: rompecabezas y reconocimiento

Proyecto academico que recibe una imagen, la divide en piezas, las desordena y, en fases posteriores, reconstruira la imagen y reconocera su contenido.

## Flujo

`imagen -> normalizacion -> piezas -> mezcla -> reconstruccion futura -> clasificacion futura`

La fase actual trabaja en crear piezas de rompecabezas desde imagenes locales o descargadas de internet. Soporta grillas regulares y poligonos irregulares con Voronoi.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

- `--sides 0`: poligonos irregulares con Voronoi.
- `--sides 3`: dos triangulos por cada celda.
- `--sides 4`: un cuadrilatero por cada celda.
- `--sides 6`: dos hexagonos irregulares por cada celda.

## Descargar imagenes abiertas

Busca imagenes en Openverse y guarda un manifiesto con fuente, autor y licencia.

```bash
python -m src.data_generation.generate_dataset download-images \
  --query "landscape photo" \
  --limit 3 \
  --output data/raw/openverse_landscape
```

## Generar data en lote

```bash
python -m src.data_generation.generate_dataset generate-batch \
  --input-dir data/raw/openverse_landscape \
  --rows 3 --cols 3 --sides 0 --seed 100 \
  --output data/generated/openverse_landscape
```

Tambien se puede descargar y generar en una sola corrida:

```bash
python -m src.data_generation.generate_dataset generate-web \
  --query "landscape photo" \
  --limit 3 \
  --raw-output data/raw/openverse_landscape \
  --rows 3 --cols 3 --sides 0 --seed 100 \
  --output data/generated/openverse_landscape
```

## Estado breve

- Generador de piezas de 3, 4 y 6 lados: disponible.
- Voronoi: implementado con piezas irregulares, preview mezclado y contornos debug.
- Descarga web: implementada con Openverse para imagenes con licencia abierta.
- Solver automatico, TensorFlow e interfaz: fuera del trabajo actual.
