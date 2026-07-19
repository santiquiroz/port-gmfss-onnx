# Fuentes vendoreadas

Este proyecto vendorea (copia) código fuente de dos repos MIT upstream. Ningún código de
`sniklaus/softmax-splatting` (licencia académica, no comercial) fue clonado, leído ni usado como
referencia en ningún momento — esa restricción es dura y se mantiene en todas las fases.

## 1. HolyWu/vs-gmfss_fortuna (MIT)

- Repo: https://github.com/HolyWu/vs-gmfss_fortuna
- Commit vendoreado: `f4f990a456678942beb7bcbca3fd5645d139ebe4` (2023-04-08, HEAD de `master` al momento
  de vendorear, 2026-07-18)
- Por qué este repo: "composición limpia" del pipeline GMFSS_Fortuna como paquete VapourSynth, con los
  pesos ya commiteados en `vsgmfss_fortuna/models/*.pkl` (no requiere descargar de Google Drive).
- Vendoreado en `toolkit/vendor/vs_gmfss_fortuna/`:
  - `FeatureNet.py`, `MetricNet.py`, `FusionNet_b.py`, `util.py` (contiene `MyPReLU`/`MyPixelShuffle`,
    reimplementaciones trace-friendly de `nn.PReLU`/`nn.PixelShuffle`, verificadas matemáticamente
    equivalentes y con el mismo nombre de parámetro `weight` — compatibles con los `state_dict`
    guardados en los `.pkl`)
  - `gmflow/` completo (`gmflow.py`, `backbone.py`, `geometry.py`, `matching.py`, `position.py`,
    `transformer.py`, `trident_conv.py`, `utils.py`, `__init__.py`) — GMFlow (optical flow con
    atención tipo swin)
  - `models/flownet.pkl` (GMFlow, 18.9MB), `models/feat_base.pkl` (FeatureNet, 3.3MB),
    `models/fusionnet_base.pkl` (FusionNet/GridNet, 31.4MB), `models/metric_base.pkl` (MetricNet, 0.5MB)
    — **variante "base" únicamente** (ver sección "Variante PG/pg104" abajo). Estos `.pkl` están
    gitignored (`*.pkl` en `.gitignore`) y no se commitean; se re-generan re-clonando el repo HolyWu
    en el commit de arriba.
- **Explícitamente NO vendoreado** de este repo (no aplica a la variante PG/base, o usa cupy):
  - `softsplat.py` (usa `cupy` — el pipeline de referencia usa `softsplat_torch.py` de 98mxr en su lugar)
  - `FusionNet_u.py`, `IFNet_HDv3.py`, `models/*_union.pkl`, `models/rife.pkl` (solo se usan en modo
    `union`, no en el modo `base`/PG que es el target de este port)
  - `GMFSS.py`, `__init__.py` (el wrapper VapourSynth original; su lógica de composición fue
    reimplementada standalone en `toolkit/run_reference.py`, propia de este proyecto, para no depender
    de VapourSynth/TensorRT/CUDA)

## 2. 98mxr/GMFSS_Fortuna (MIT)

- Repo: https://github.com/98mxr/GMFSS_Fortuna
- Commit vendoreado: `0fb7ac1dc292e2615217110dd9d82557845fb919` (2026-01-27, HEAD de `master` al momento
  de vendorear, 2026-07-18)
- Por qué este repo: es el único de los dos con una implementación pura-PyTorch de softmax-splatting
  (`model/softsplat_torch.py`) que no requiere `cupy`/CUDA. `HolyWu/vs-gmfss_fortuna` solo tiene la
  variante cupy (`softsplat.py`).
- Vendoreado en `toolkit/vendor/gmfss_fortuna_98mxr/`:
  - `softsplat_torch.py` — **único archivo tomado de este repo**, verbatim, sin modificar. Implementación
    forward-splatting pura PyTorch (`index_add_` + bilinear splatting a 4 vecinos), sin `cupy`, sin
    kernels CUDA custom. Contiene el control-flow no exportable a ONNX mencionado en el brief de la
    Fase 1 (`if not finite_mask.any(): return tenOut`) — para la referencia CPU no importa, corre eager.
- **Explícitamente NO vendoreado** de este repo:
  - `model/softsplat.py` (usa `cupy`, y aunque tiene licencia MIT propia, se evitó por completo para no
    correr riesgo de mezclar con derivaciones de `sniklaus/softmax-splatting`)
  - Todo lo demás (`train_*.py`, `model/GMFSS_infer_*.py`, `model/discriminator.py`, `model/lpips/*`,
    `model/dataset.py`, etc.) — código de entrenamiento/composición no necesario; la composición del
    pipeline de referencia es propia (`toolkit/run_reference.py`), no una copia de
    `model/GMFSS_infer_b.py`.

## Variante PG / pg104

El brief de la tarea pide explícitamente la variante **"PG"/pg104**. Ninguno de los dos repos usa
literalmente el string "pg104" en el código ni en pesos — es terminología de SVFI (Squirrel Video Frame
Interpolation, https://doc.svfi.group/en/pages/model-spec/), el software GUI que popularizó estos
checkpoints. Investigación (búsqueda web, doc.svfi.group/en/pages/model-spec/) da:

> **pg104**: "The newest GMFSS anime model, currently the most powerful anime frame interpolation model"
> (distinto de `union_v`/`Umss_v1`, que son variantes union)

Y en el propio repo de entrenamiento `98mxr/GMFSS_Fortuna`, el README documenta:

```
1. Train gmfss with gan optimization        →  train_pg.py    (modo base, con GAN)
2. Train gmfss_union with gan optimization   →  train_upg.py   (modo union, con GAN)
```

Es decir, **`train_pg.py` entrena específicamente el modo "base" (no-union) con optimización GAN** — el
nombre del script (`pg` = train script para el modo *base* con *GAN*) es la fuente directa del nombre
"PG" del checkpoint que circula en SVFI. `HolyWu/vs-gmfss_fortuna` empaqueta el resultado de ese
entrenamiento como `feat_base.pkl` / `fusionnet_base.pkl` / `metric_base.pkl` (sufijo `_base`, no
`_union`), y su propio `gmfss_fortuna()` (`vsgmfss_fortuna/__init__.py`) documenta `model=0` como
`"0 = base model"` y lo deja como default.

**Conclusión (juicio documentado, no 100% verificable byte-a-byte contra el checkpoint numerado
"104" específico de SVFI, que es una build interna cerrada de ese proyecto)**: la variante "PG"/pg104
pedida en el brief corresponde a los pesos **`*_base.pkl`** de `HolyWu/vs-gmfss_fortuna` — el modo no-union
de GMFSS_Fortuna, entrenado vía `train_pg.py`. Es la interpretación más defendible dada la evidencia
disponible (nomenclatura del script de entrenamiento + documentación pública de SVFI describiendo pg104
como "el modelo anime más potente", coincidente con la descripción que 98mxr da del modo base/gmfss
frente a union). Se reporta como **DONE_WITH_CONCERNS** por esta ambigüedad — si en una fase posterior
aparece evidencia de que "pg104" es en cambio la variante union, el cambio de pesos es mecánico (cambiar
sufijo `_base` → `_union` y usar `FusionNet_u.py` + `rife.pkl` + `IFNet_HDv3.py`, no afecta la arquitectura
GMFlow/FeatureNet/MetricNet/softsplat ya validada).

## Composición del pipeline (propia, `toolkit/gmfss_pg_pipeline.py`)

No existe un archivo único vendoreado que combine FeatureNet + GMFlow + MetricNet + softsplat_torch +
FusionNet en modo base sin cupy — `GMFSS.py` (HolyWu) usa `softsplat.py` (cupy) y `GMFSS_infer_b.py`
(98mxr) importa por paquete absoluto (`model.xxx`, no reusable standalone). `toolkit/gmfss_pg_pipeline.py`
reimplementa el `forward()`/`reuse()` de esos dos archivos (arquitectónicamente idénticos entre sí — ver
diffs verificados abajo) usando `softsplat_torch.softsplat` como única función de warp, y añade hooks de
captura de tensores intermedios. Es código propio de este proyecto, no una copia. `toolkit/run_reference.py`
es el driver/entrypoint que carga los tripletes de `refs/inputs/`, invoca la composición y escribe
`refs/golden/`.

### Archivos `__init__.py` añadidos (no upstream)

`toolkit/vendor/vs_gmfss_fortuna/__init__.py`, `toolkit/vendor/gmfss_fortuna_98mxr/__init__.py`, y
`toolkit/vendor/vs_gmfss_fortuna/gmflow/__init__.py` (este último sí existía upstream pero vacío) se
dejaron vacíos intencionalmente — solo existen para que `from .util import ...` / `from .gmflow.geometry
import ...` (imports relativos usados dentro del código vendoreado sin modificar) resuelvan como paquete
Python normal. El `__init__.py` real de HolyWu (`vsgmfss_fortuna/__init__.py`, que importa
`vapoursynth`/`tensorrt`/`torch_tensorrt`) NO se vendoreó — no es necesario para inferencia standalone y
esas dependencias no están en `toolkit/requirements.txt`.

### Verificación de equivalencia entre HolyWu y 98mxr

Antes de elegir HolyWu como fuente de código, se diffearon los archivos compartidos entre ambos repos:
`FeatureNet.py`, `MetricNet.py`, `FusionNet_b.py`, y todo `gmflow/*.py`. Las diferencias son 100%
cosméticas: reordenamiento de imports, `MyPReLU`/`MyPixelShuffle` (envoltorios trace-friendly
matemáticamente idénticos a `nn.PReLU`/`nn.PixelShuffle`, mismo nombre de parámetro), llamadas
`torch.fx.wrap(...)`, y diferencias de whitespace/trailing newline. Ningún cambio de arquitectura,
ningún cambio de forma de tensor, ningún cambio de fórmula matemática.
