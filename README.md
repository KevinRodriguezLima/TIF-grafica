# TIF GrÃ¡fica: rompecabezas y reconocimiento

Proyecto acadÃ©mico que recibe una imagen, la divide en piezas, las desordena y, en fases posteriores, reconstruirÃ¡ la imagen y reconocerÃ¡ su contenido.

## Flujo

`imagen â†’ normalizaciÃ³n â†’ piezas â†’ mezcla â†’ reconstrucciÃ³n futura â†’ clasificaciÃ³n futura`

La fase actual trabaja Ãºnicamente en cÃ³mo crear las piezas del rompecabezas. El primer prototipo usa una cuadrÃ­cula regular porque permite validar el recorte y la mezcla antes de experimentar con Voronoi.

## InstalaciÃ³n

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

- `--sides 3`: dos triÃ¡ngulos por cada celda.
- `--sides 4`: un cuadrilÃ¡tero por cada celda.
- `--sides 6`: dos hexÃ¡gonos irregulares por cada celda.

Los triÃ¡ngulos y hexÃ¡gonos usan transparencia para conservar sus lados rectos sin perder partes de la imagen.

## Estado breve

- Generador de piezas de 3, 4 y 6 lados: disponible para revisiÃ³n.
- Voronoi: alternativa pendiente de evaluar, no implementada todavÃ­a.
- Solver automÃ¡tico, TensorFlow e interfaz: fuera del trabajo actual.

