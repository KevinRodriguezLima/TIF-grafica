# TIF Gráfica: rompecabezas y reconocimiento

Proyecto académico que recibe una imagen, la divide en piezas, las desordena y, en fases posteriores, reconstruirá la imagen y reconocerá su contenido.

## Flujo

`imagen → normalización → piezas → mezcla → reconstrucción futura → clasificación futura`

La fase actual trabaja únicamente en cómo crear las piezas del rompecabezas. El primer prototipo usa una cuadrícula regular porque permite validar el recorte y la mezcla antes de experimentar con Voronoi.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python -m src.main generate \
  --input data/input/dbz.jpg \
  --rows 3 --cols 3 --sides 4 --seed 42 \
  --output output/test_001
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
