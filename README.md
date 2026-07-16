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
  --rows 3 --cols 3 --seed 42 \
  --output output/test_001
```

La salida contiene la imagen procesada, las piezas PNG, dos versiones del rompecabezas desordenado y sus metadatos JSON.

## Estado breve

- Prototipo del generador de piezas rectangulares: disponible para revisión.
- Voronoi: alternativa pendiente de evaluar, no implementada todavía.
- Solver automático, TensorFlow e interfaz: fuera del trabajo actual.
