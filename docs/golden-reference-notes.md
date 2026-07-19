# Referencia dorada (Task 0.2): notas

`refs/golden/` es generado por `toolkit/run_reference.py` corriendo GMFSS_Fortuna variante PG/base
(ver `docs/vendored-sources.md`) en CPU puro sobre los tripletes de `refs/inputs/`. Los `.npy`
y PNGs de `refs/golden/` están gitignored (`refs/golden/` completo, ver `.gitignore`) — se
regeneran localmente corriendo el script, no se commitean (son grandes y 100% reproducibles a
partir de código + pesos vendoreados + `refs/inputs/`, que sí se commitean).

## Regenerar

```powershell
pwsh -File toolkit/setup-env.ps1        # una vez
.venv/Scripts/python.exe toolkit/run_reference.py
```

## Contenido real de `refs/inputs/` (concern documentado)

El brief de la tarea describía `vf.mp4`, `vs.mp4`, `vwarm.mp4`, `cancel.mp4` (en
`image-upscaler-amd/runtime/uploads/`) como "real 1920x1080, 24fps, 24-frame anime clips ...
usados como fixtures de prueba de Upflow". Al extraer e inspeccionar visualmente los frames
(`ffmpeg` + lectura directa de los PNG resultantes), **no son anime**: son patrones sintéticos de
calibración estilo SMPTE — barras de color planas, una línea diagonal con gradiente que se
desplaza cuadro a cuadro, un punto y una barra gris en movimiento, un timecode superpuesto
(`00:00:00.NNN` + número de frame), y un patrón de ruido/checkerboard en una esquina. Dados los
nombres de los archivos (`vwarm` = warm-cache, `cancel` = cancelación de job), son casi
seguramente fixtures de integración para la mecánica de pipeline de Upflow (extracción de frames,
encoding, cache, cancelación), no muestras curadas de contenido anime para evaluar calidad visual
de un modelo de IA.

**Impacto en esta tarea**: se usaron estos archivos igual, tal como indicó explícitamente el
brief ("usa estos — NO fetches nada de internet para frames de prueba"), porque la instrucción de
evitar fetch de internet es una restricción dura y estos archivos sí tienen movimiento real,
continuo y determinista (línea/punto/barra trasladándose linealmente cuadro a cuadro) — son un
test funcional válido de que el pipeline de optical flow + softsplat + fusion produce un frame
intermedio geométricamente correcto (SSIM alto es esperable y se verificó). Lo que **no** se puede
afirmar es que esto ejercite las características específicas de motion/estilo anime (bordes cel-
shaded, oclusión de personajes, paneles con motion blur estilizado) que es la razón de ser de
GMFSS_Fortuna frente a un interpolador genérico — la validación de *ese* aspecto queda pendiente
para cuando haya contenido anime real disponible (p. ej. un clip CC explícito, como sugiere el
brief como fallback).

**Recomendación para fases posteriores**: si en algún punto se necesita validar específicamente
comportamiento anime-especifico (no solo paridad numérica ONNX-vs-PyTorch, que es agnóstica al
contenido), sustituir `refs/inputs/` por 2-3 frames de un clip anime CC-licenciado explícito. No
afecta la validez de la referencia dorada para su propósito principal en este proyecto: ser el
ground truth numérico que Fase 1/2/3 usan para verificar que las exportaciones ONNX producen
bit-a-bit (dentro de tolerancia de punto flotante) los mismos tensores que esta corrida PyTorch
CPU — esa validación es independiente del contenido semántico de la imagen.
