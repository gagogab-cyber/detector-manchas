# ▣ Inspección de tela — web (Flask) y Streamlit Cloud

Página web donde subes la foto de una prenda y marca las zonas fuera de tolerancia
(manchas, cambios de tono, roturas). Usa CIELAB + ΔE CIEDE2000, umbralización adaptativa y
análisis espacial. Corre en CPU con ~200 MB de RAM. Sin base de datos: las fotos no se guardan en disco.

## Cómo correrla
**Windows:** doble clic en `correr.bat` (crea el venv, instala todo y levanta el servidor).
**Mac/Linux:** `./correr.sh`

A mano:
```
python -m venv venv
venv\Scripts\activate            (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Luego abre **http://localhost:5000** en el navegador.

**Interfaz Streamlit** (la misma que se publica en la nube):

```
streamlit run streamlit_app.py
```

Abre **http://localhost:8501**. Para esta interfaz no hacen falta Flask ni waitress.
La consola también muestra una dirección para abrirla **desde el celular** (misma Wi-Fi);
el botón "Tomar foto con la cámara" abre la cámara del teléfono.

Si el puerto 5000 está ocupado: `set PORT=8080` (Windows) o `PORT=8080 python app.py` (Mac/Linux).

## Novedades de esta versión
- **Fotos de catálogo (prenda sobre fondo liso):** detecta el fondo (blanco, gris, de color, con degradado
  de luz o PNG transparente), analiza solo la prenda y mide lo "normal" solo dentro de ella.
  Antes marcaba la polera entera como mancha (81% de área).
- **Cuello, puños, costuras y contorno** de la prenda ya no se marcan como defectos.
- **Colores apagados y tonos piel (beige, salmón, rojo ladrillo):** antes el filtro de piel borraba la
  prenda entera. Ahora la piel solo se ignora si es una parte menor de la foto (manos, brazos).
- **Manchas oscuras suaves o algo difusas** ya no se pierden: el filtro de sombras compara el borde y el contraste con la zona vecina, en vez de descartar todo lo que solo cambie de brillo. Las sombras anchas siguen descartándose.
- Contornos más ajustados: cada mancha crece solo hasta la mitad de su propia intensidad.
- El % de área dañada se calcula sobre la prenda (no sobre toda la foto) cuando hay fondo.

## Qué trae la página
- Arrastrar y soltar, elegir archivo o tomar foto con la cámara (se analiza sola al cargar).
- Veredicto (APROBADA / FUERA DE TOLERANCIA), nº de defectos, **% de área dañada** y tiempo.
- Imagen analizada, zonas marcadas (rojo = grave, naranja = leve, línea amarilla = contorno) y mapa de anomalía. Clic en una imagen para ampliarla.
- Tabla de defectos con ΔE medio/máx, tipo y nivel; descarga de imagen marcada y CSV.
- Ajustes: tolerancia ΔE, sensibilidad, área mínima, resolución, filtros y recorte de bordes.

## Carpetas
```
app.py               servidor Flask local (API + página)
templates/index.html página Flask
static/css/style.css estilo industrial
static/js/main.js    lógica del navegador (carga, API, resultados)
streamlit_app.py     archivo principal en Streamlit Cloud
app_streamlit.py     interfaz Streamlit (toda la lógica de la app en la nube)
.streamlit/          tema de Streamlit (sin secretos)
src/                 motor de detección (color, preprocess, detector, cnn)
cli.py               uso por consola
train_cnn.py         entrena la CNN opcional (requiere torch, solo local)
tests/               pruebas (motor y web)
ejemplos/            5 fotos reales usadas como pruebas
requirements.txt     dependencias de Flask y Streamlit
.gitignore           excluye venv, cachés y archivos locales
```

## API (por si la conectas a otro sistema)
`POST /api/analizar` (multipart): campo `imagen` + ajustes opcionales
(`tol`, `z_thr`, `min_area`, `max_side`, `boost`, `sombras`, `estampados`, `piel`, `paleta`, `cnn`,
`crop_izq`, `crop_der`, `crop_arr`, `crop_aba`). Responde JSON con `veredicto`, `total`,
`area_danada_pct`, `defectos[]` e imágenes en base64. Valores fuera de rango se limitan solos.
`GET /api/salud` → `{"estado":"ok"}`.

## Cómo funciona el motor
0. Si hay fondo liso conectado al borde, se separa la prenda (se ignora lo que toca su contorno: cuello, puños).
1. Preproceso: bilateral + blur proporcional al tamaño de la foto (apaga la textura del tejido).
2. CIELAB: separa luminosidad L\* del color a\*, b\*.
3. Color normal de la tela con un ajuste de superficie robusto (una mancha grande no lo contamina).
4. Puntaje de anomalía: luminosidad y color por separado, cada uno normalizado por el ruido de esa tela.
5. Histéresis por zona: la mancha crece hasta su borde real.
6. Filtros: sombras/arrugas (borde y contraste local), estampados repetidos, piel grande, fragmentos de borde.
7. ΔE CIEDE2000 por zona, tipo y nivel.

## Pruebas
```
python tests/test_regresion.py    motor: 5 fotos reales (telas y polera con fondo blanco) + sintética
python tests/test_colores.py      9 colores de prenda (beige, rojo ladrillo, negro...) sobre fondo blanco
python tests/test_sombras.py      manchas suaves/difusas frente a sombras y telas limpias
python tests/test_web.py          web: devuelve lo mismo que el motor, y maneja errores
```

## Subir a GitHub y Streamlit Cloud
El repo está listo para subir tal cual: `.gitignore` excluye `venv/`, `__pycache__/`,
`.vscode/`, modelos entrenados y salidas temporales. Si subes los archivos arrastrándolos
desde el explorador, no incluyas esas carpetas (son locales y no afectan el resultado).

1. Sube la carpeta a un repositorio de GitHub.
2. En [share.streamlit.io](https://share.streamlit.io) elige **New app** y el repo.
3. **Main file path:** `streamlit_app.py` (lo detecta automáticamente).
4. **Advanced settings → Python version:** 3.11 o 3.12. requirements.txt se instala solo.
5. **Deploy.** No hay secretos ni variables de entorno obligatorias.

Notas para la nube:
- `requirements.txt` ya trae `streamlit` y `pandas`. `torch` se dejó fuera a propósito: la
  CNN es opcional y solo se ofrece si existe `models/stain_cnn.pt` (no viene incluido).
- El límite es 20 MB por foto, igual que en la versión Flask. Las fotos grandes se reducen antes de analizarse.
- Si quieres publicar solo Streamlit, puedes borrar `app.py`, `templates/`, `static/`,
  `correr.bat`, `correr.sh` y `cli.py` de la copia subida: el motor en `src/` es el mismo.
- Para probar cambios antes de subirlos: `streamlit run streamlit_app.py`.

## Notas para ponerla en un servidor
- `python app.py` usa **waitress** (2 hilos, apto para Windows y para 4 GB de RAM). No uses el servidor de desarrollo de Flask en producción.
- Límite de 20 MB por foto. Las fotos grandes se reducen antes de analizar.
- Para internet pública, ponla detrás de HTTPS (nginx, Caddy o el proveedor). Con HTTPS la cámara del navegador funciona también en celulares.
- La página carga Bootstrap desde un CDN: necesita internet en el navegador para verse con todo el estilo.

## Límites conocidos
- Detecta **anomalías**; no distingue mancha de rotura o quemadura (para eso, entrenar la CNN con `train_cnn.py`).
- Mejor con la prenda plana, de frente y llenando el cuadro.
- Una mancha de más de ~40% de la foto se confunde con el color normal.
- Manchas **pegadas al contorno** de la prenda (dentro de ~2% del borde) se ignoran cuando hay fondo separado.
- Si la prenda es del mismo color que el fondo (blanco sobre blanco) o el fondo tiene viñeta fuerte, no se
  separa y se analiza toda la imagen: recorta bordes o usa una foto con mejor contraste.
- Sensibilidad por debajo de 3 hace que las manchas se unan con sombras y pliegues; por eso el mínimo es 3.
- Calibrado con 5 fotos reales y 9 colores sintéticos; una tela nueva puede pedir retocar la Sensibilidad.
