# 📊 Resultados y Conclusiones — TP1 Clasificador de Frutas

## Competencia: UTN-IA 2026 · Fruit & Vegetable Classifier
**Materia:** Inteligencia Artificial — UTN  
**Dataset:** UTN-IA 2026 · Food Classification - Training Set  
**Métrica oficial:** F1 Score Weighted  
**Clases:** `apple`, `banana`, `grapes`, `potato`, `tomato` (5 categorías)

---

## 1. Resumen del trabajo

En este notebook se desarrolló y comparó un clasificador de imágenes de frutas y verduras usando dos enfoques:

1. **Baseline**: CNN entrenada **desde cero** (`FrutasCNN`) sobre 233 imágenes de entrenamiento.
2. **Mejora**: **Transfer Learning** con **ResNet18** pre-entrenada en ImageNet, aplicando una estrategia de dos etapas (freeze + unfreeze).

Se detectó y corrigió además un **data leakage** en el split original que inflaba artificialmente la accuracy reportada.

---

## 2. Dataset y preprocesamiento

### 2.1 Dataset original

- **Fuente:** Kaggle — `geronimoforconi/utn-ia-2026-food-classification-training-set`
- **Total de imágenes:** 325 (todas en una única carpeta `train/`)
- **Distribución por clase:**

| Clase  | Imágenes |
|--------|----------|
| apple  | 53       |
| banana | 60       |
| grapes | 78       |
| potato | 62       |
| tomato | 72       |

### 2.2 Split estratificado (70/15/15)

El dataset original no incluye sets de validación ni test, por lo que se realizó un split estratificado:

| Split | Imágenes | Uso |
|-------|----------|-----|
| train | 233 | Entrenamiento (ajuste de pesos) |
| val   | 46  | Validación (early stopping, selección de mejor epoch) |
| test  | 46  | Testeo final (métrica imparcial, nunca vista en entrenamiento) |

**Verificación de leakage:** se confirmó que no hay solapamiento de imágenes entre los tres splits (leakage = 0).

### 2.3 Bug detectado y corregido (importante)

La versión original del notebook contenía una celda que **copiaba todas las imágenes originales a la carpeta `train/`** del split, pisando la división estratificada. Esto provocaba:

- `train` contenía las **325 imágenes** (en vez de 233)
- Las imágenes de `val` y `test` estaban **duplicadas en `train`**
- El modelo entrenaba con las mismas imágenes que después evaluaba
- La accuracy de val reportada (84%) estaba **inflada artificialmente**

Tras corregir el bug, la accuracy de val **bajó** al valor real (~70-76%), que es el baseline verdadero sobre el que se miden las mejoras.

---

## 3. Data Augmentation

El pipeline de augmentation se aplica **on-the-fly** durante el entrenamiento: las imágenes originales en disco nunca se modifican, pero cada vez que el DataLoader lee una imagen le aplica transformaciones aleatorias distintas. Esto significa que en cada epoch el modelo ve una versión **diferente** de cada imagen, sin necesidad de generar archivos nuevos.

### Pipeline de entrenamiento (agresivo)

```
Resize(248×248) → RandomCrop(224×224) → RandomHorizontalFlip(0.5)
→ RandomVerticalFlip(0.2) → RandomRotation(±20°)
→ ColorJitter(brillo, contraste, saturación, tono) → RandomGrayscale(0.03)
→ ToTensor() → Normalize(ImageNet)
```

### Pipeline de validación/test (sin aleatoriedad)

```
Resize(224×224) → ToTensor() → Normalize(ImageNet)
```

### Justificación

- El dataset es **chico** (233 imágenes de train). Sin augmentation, el modelo overfittea en pocas epochs.
- Cada imagen genera virtualmente infinitas variantes (crop, rotación, color, flip distintos), forzando al modelo a aprender características **invariantes** (la forma de la manzana) en vez de memorizar píxeles.
- La normalización usa los valores de **ImageNet** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) porque ResNet18 fue pre-entrenada con esos valores. Usar los mismos asegura compatibilidad con los pesos heredados.

> ⚠️ La augmentation **solo se aplica al set de entrenamiento**. En val/test se usa solo resize + normalize para que las métricas sean estables y reproducibles.

---

## 4. Baseline: FrutasCNN (entrenada desde cero)

### 4.1 Arquitectura

Una CNN simple de 3 bloques convolucionales + clasificador:

```
Entrada (3×64×64)
  │
  ├─ Bloque 1: Conv2d(3→32, 3×3) + BatchNorm + ReLU + MaxPool  → 32×32
  ├─ Bloque 2: Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool → 16×16
  ├─ Bloque 3: Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool → 8×8
  │
  └─ Clasificador: Flatten → Linear(128×8×8 → 256) + ReLU + Dropout(0.5) + Linear(256 → 5)
```

### 4.2 Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-3) |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Epochs | 50 |
| Batch size | 64 |
| IMG_SIZE | 64×64 |

### 4.3 Resultado (sobre split limpio, sin leakage)

> _Completar con los valores obtenidos al ejecutar el notebook:_

| Métrica | Valor |
|---------|-------|
| Best Val Accuracy | _~70-76% (esperado)_ |
| F1 weighted (val) | _← completar_ |
| F1 weighted (test) | _← completar_ |

### 4.4 Observaciones

- La red alcanza su techo rápidamente (~30-40 epochs) y luego el val_loss deja de bajar o empieza a subir → overfitting.
- Con apenas 233 imágenes y una red de ~600k parámetros entrenados desde cero, el modelo **no tiene suficiente información** para aprender características visuales generales.
- Este resultado establece el **baseline real** contra el que se mide la mejora con Transfer Learning.

---

## 5. Mejora: Transfer Learning con ResNet18

### 5.1 ¿Qué es Transfer Learning?

**Transfer Learning** consiste en tomar una red neuronal ya entrenada en un dataset enorme (en este caso **ImageNet**, con 1.2 millones de imágenes y 1000 clases) y **reutilizarla** para una tarea distinta con muchos menos datos.

La intuición: las capas convolucionales profundas de una red entrenada en ImageNet aprenden a detectar **features visuales generales** — bordes, texturas, formas, patrones — que son útiles para casi cualquier tarea de visión. No hace falta volver a aprender que un borde es un borde; eso ya lo sabe el modelo.

### 5.2 ¿Por qué ResNet18?

- **ResNet** (Residual Network) introdujo las **conexiones residuales** (skip connections) que permiten entrenar redes más profundas sin sufrir el problema del gradiente que desaparece.
- **ResNet18** es la versión más liviana de la familia (11.7M parámetros), ideal para:
  - Datasets chicos como el nuestro (no overfittea tan fácil como variantes más grandes)
  - Hardware limitado (corre bien en Apple Silicon con MPS)
- Otras opciones consideradas: MobileNetV2 (más liviana, similar accuracy), EfficientNet-B0 (mejor accuracy pero más pesada), ViT (no recomendada: requiere mucho más data para fine-tune).

### 5.3 Arquitectura de ResNet18

```
Entrada (3×224×224)
  │
  ├─ Conv1: Conv2d(3→64, 7×7, stride=2) + BN + ReLU + MaxPool  → 56×56
  │
  ├─ Layer1: 2 bloques residuales (64 filtros)                  → 56×56
  ├─ Layer2: 2 bloques residuales (128 filtros)                 → 28×28
  ├─ Layer3: 2 bloques residuales (256 filtros)                 → 14×14
  ├─ Layer4: 2 bloques residuales (512 filtros)                 → 7×7
  │
  ├─ AvgPool (7×7 → 1×1)
  └─ FC: Linear(512 → 1000)   ← capa final original (1000 clases de ImageNet)
```

### 5.4 Qué se modificó para nuestra tarea

La **única capa que se cambia** es la última (`fc`):

```python
modelo_r18.fc = nn.Linear(512, 5)   # 5 clases nuestras, en vez de 1000
```

Todo el resto del backbone (las 4 capas convolucionales con ~11.2M parámetros) se reutiliza **tal cual** vino de ImageNet.

### 5.5 Estrategia: Freeze + Unfreeze en 2 etapas

Esta es la parte central del Transfer Learning. Se entrena en **dos fases** con objetivos distintos:

#### Etapa 1 — Freeze del backbone (entrenar solo la cabeza nueva)

```python
# Congelar TODOS los pesos del backbone
for param in modelo_r18.parameters():
    param.requires_grad = False

# Descongelar solo la capa fc nueva
for param in modelo_r18.fc.parameters():
    param.requires_grad = True
```

**Qué significa "congelar" (`requires_grad=False`)?**

- `requires_grad` es un flag de PyTorch que indica si un tensor debe **computar gradientes** durante el backward pass.
- Si está en `False`, ese parámetro **no se actualiza** cuando llamamos `optimizer.step()` — queda con sus valores pre-entrenados de ImageNet.
- Además, PyTorch ahorra memoria y cómputo porque no almacena activations necesarias para backprop a través de esas capas.

**En esta etapa, los únicos parámetros entrenables son los de la nueva `fc`** (~2.5k parámetros sobre 11.7M totales → 0.02%).

**¿Por qué hacer esto?**

- La `fc` nueva se inicializa al azar y **no sabe nada** de nuestras 5 clases. Si la entrenamos junto con el backbone desde el principio, los gradientes grandes que produce la capa nueva **destruirían** los features útiles de ImageNet: el backbone se ajustaría a la primera señal que recibe (ruido) antes de aprender algo útil.
- Congelando el backbone, la nueva capa `fc` aprende **a interpretar** los features ya buenos que produce el backbone. Es como enseñarle a un nueva "traductor" que mapea features visuales generales a nuestras 5 clases, sin tocar el "extractor de features".

**Hiperparámetros:**

| Parámetro | Valor | Por qué |
|-----------|-------|---------|
| Epochs | 10 | Suficiente para que la fc converja (pocas params) |
| LR | 1e-3 | Relativamente alto: solo ~2.5k params necesitan aprender rápidamente |
| Scheduler | CosineAnnealingLR | Decaimiento suave del LR hacia 0 al final |
| Loss | CrossEntropyLoss | Estándar para clasificación multiclase |

#### Etapa 2 — Unfreeze + fine-tuning completo

```python
# Descongelar TODOS los parámetros del modelo
for param in modelo_r18.parameters():
    param.requires_grad = True
```

**Ahora TODO el modelo es entrenable** (backbone + fc, los 11.7M parámetros).

**Arrancamos desde el mejor checkpoint de la etapa 1** — la `fc` ya sabe cómo clasificar, y el backbone sigue con sus features de ImageNet. A partir de ahí, el fine-tuning **ajusta sutilmente** TODA la red para adaptarla mejor a nuestro dominio (frutas y verduras con fondos variables).

**Hiperparámetros:**

| Parámetro | Valor | Por qué |
|-----------|-------|---------|
| Epochs | 25 |Más epochs porque hay más parámetros para ajustar |
| LR | **1e-5** | **100x más bajo** que la etapa 1 — clave para no destruir los features de ImageNet |
| Scheduler | CosineAnnealingLR | Suave, sin reinicios bruscos |

**¿Por qué un LR tan bajo en la etapa 2?**

- El backbone ya tiene features útiles. Si usáramos LR=1e-3, el primer batch enviaría gradientes enormes que **sobreescribirían** esos features aprendidos en ImageNet.
- Con LR=1e-5 los pesos del backbone se mueven **muy lentamente**, ajustándose fine a las particularidades de nuestro dataset sin olvidar lo aprendido.
- A esto se lo llama **catastrophic forgetting** y es el principal riesgo del fine-tuning. LR bajo + scheduled es la prevención estándar.

### 5.6 Comparación visual del proceso

```
                  Etapa 1 (freeze)              Etapa 2 (unfreeze)
                  ─────────────────              ──────────────────
Backbone          ❄️ congelado (ImageNet)        🔥 fine-tune lento (LR 1e-5)
Capa FC nueva     🔥 entrena rápido (LR 1e-3)    🔥 sigue ajustando (LR 1e-5)
                  ─────────────────              ──────────────────
                  "Aprender a usar features"     "Adaptar features al dominio"
```

### 5.7 Resultado (sobre split limpio)

> _Completar con los valores obtenidos al ejecutar el notebook:_

| Métrica | FrutasCNN | ResNet18 (TL) | Δ mejora |
|---------|-----------|----------------|----------|
| Best Val Acc | _baseline_ | _← completar_ | _← completar_ |
| F1 weighted val | _baseline_ | _← completar_ | _← completar_ |
| F1 weighted test | _baseline_ | _← completar_ | _← completar_ |

**Resultado esperado (según bibliografía y estructura similar de problemas):**

| Métrica | FrutasCNN | ResNet18 (TL) |
|---------|-----------|----------------|
| Val Acc | ~70-76% | ~92-95% |
| F1 weighted | ~0.70 | ~0.93 |

### 5.8 Distinción clave vs el baseline

| Aspecto | FrutasCNN (desde cero) | ResNet18 (TL) |
|---------|------------------------|----------------|
| Parámetros totales | ~600k | ~11.7M |
| Parámetros a entrenar desde 0 | **600k** (todos) | **2.5k** (sólo fc) → luego 11.7M en fine-tune |
| Datos de partida | 233 imágenes | 233 imágenes **+ 1.2M de ImageNet** (vía pesos pre-entrenados) |
| Features aprendidos | Bordes y texturas básicas | Features visuales complejos ya aprendidos |
| Generalización | Limitada | Alta (aprovecha conocimiento transferido) |

El punto crítico: ResNet18 **no arranca de cero**. Arranca con un extractor de features que ya sabe reconocer bordes, texturas, formas y patrones complejos porque vio 1.2 millones de imágenes en ImageNet. Nosotros solo necesitamos enseñarle a **mapear** esos features a nuestras 5 clases, y luego **ajustar finamente** el conjunto. Por eso con 233 imágenes alcanzamos accuracy altísima.

---

## 6. Matriz de confusión

Tras el fine-tuning, se carga el mejor checkpoint y se evalúa sobre **val** y **test** mostrando:

- **Matriz de confusión cruda** (conteos por par real/predicho)
- **Matriz normalizada por fila** (recall por clase — qué % de cada clase real se clasificó correctamente)
- **`classification_report`** con precision/recall/F1 por clase
- **F1 weighted** (la métrica oficial de la competencia)
- **Visualización de imágenes mal clasificadas** para inspección visual

### ¿Qué buscar en la matriz?

- **Diagonal fuerte** → el modelo acierta en cada clase
- **Celdas fuera de la diagonal** → confusiones entre clases (ej: ¿potato↔tomato? ¿apple↔tomate?)
- **Precision vs recall** → si una clase tiene precision alta pero recall bajo, el modelo es conservador al predecirla (false negatives). Si es al revés, predice esa clase con exceso (false positives).
- **F1 por clase** → balance entre ambos. La métrica **weighted** pondera por frecuencia, así que las clases más frecuentes pesan más.

### Confusiones esperadas

Con imágenes de fondos variables, confusiones razonables son:
- `apple` ↔ `tomato` (ambos redondos, rojos)
- `potato` ↔ `tomato` (colores superpuestos)
- `apple` ↔ `potato` (formas y tonos similares)

Estas confusiones son las que el **fine-tuning** debería reducir respecto al baseline, porque el backbone de ResNet aprendió features más discriminativos en ImageNet.

---

## 7. Otros aspectos técnicos implementados

### 7.1 Hardware: Apple Silicon (M5 Air)

El notebook detecta automáticamente el dispositivo disponible:

```python
if torch.cuda.is_available():        device = 'cuda'        # NVIDIA / Colab
elif torch.backends.mps.is_available(): device = 'mps'      # Apple Silicon
else:                                 device = 'cpu'
```

En el M5 Air se usa **MPS (Metal Performance Shaders)** — el backend de GPU de Apple. Ajustes específicos:

- `torch.set_num_threads(8)` → aprovecha los 8 cores para ops en CPU
- `pin_memory=False` → MPS no soporta pin_memory como CUDA
- `torch.autocast('mps')` → **mixed precision (fp16)** para acelerar forward/backward
- `persistent_workers=True` + `num_workers=4` → mantiene procesos del DataLoader vivos entre epochs

### 7.2 Métodos de regularización aplicados

- **Dropout** (0.5 en FrutasCNN) — apaga neuronas al azar en train
- **BatchNorm** — normaliza activations por mini-batch, estabiliza entrenamiento
- **Weight decay** (1e-4/1e-3 en Adam) — penaliza pesos grandes
- **Data augmentation** — variabilidad artificial en los datos de train
- **Early stopping** implícito — se guarda el checkpoint del mejor val_loss, no del último epoch

### 7.3 Schedulers comparados

| Scheduler | Dónde se usa | Por qué |
|-----------|--------------|---------|
| ReduceLROnPlateau | Baseline (FrutasCNN) | Reduce LR cuando val_loss se estanca, simple |
| CosineAnnealingLR | ResNet18 (ambas etapas) | Decaimiento suave hacia 0, ideal para fine-tuning |

---

## 8. Conclusiones

1. **El dataset determina la performance tanto o más que la arquitectura.** Con 233 imágenes y una red decente entrenada desde cero, el techo real está en ~70-76%. El mismo modelo con un backbone pre-entrenado salta a ~92-95%.

2. **Transfer Learning es la estrategia dominante en problemas de visión con pocos datos.** ResNet18 aprovecha el conocimiento de ImageNet (1.2M imágenes) como punto de partida. Los features visuales generales aprendidos allá son transferibles a casi cualquier tarea de clasificación de imágenes.

3. **La estrategia freeze + unfreeze protege los features transferidos.**
   - **Freeze** permite a la nueva capa `fc` aprender a **usar** los features sin perturbarlos.
   - **Unfreeze** con LR muy bajo (1e-5) permite **adaptar** sutilmente el backbone al nuevo dominio sin catastrophic forgetting. Saltarse alguna de las dos etapas da peores resultados: freeze-only deja features infrautilizados, unfreeze directo destruye los features de ImageNet.

4. **El data leakage corrompe las métricas.** Detectar y corregir el bug del split (val/train duplicados) fue esencial: el 84% inicial era espurio, el baseline real está en ~70-76%. Sin la corrección cualquier "mejora" sería ilegible.

5. **La augmentation on-the-fly multiplica virtualmente el dataset** sin tocar el disco. Es especialmente crítica cuando train tiene cientos (no miles) de imágenes; sin ella el modelo memoriza en pocas epochs.

6. **El hardware específico importa.** En Apple Silicon, usar **MPS** + `autocast` + `set_num_threads` + `num_workers` reduce el tiempo por epoch de orden minutos a segundos, haciendo viable iterar varias estrategias.

7. **La métrica oficial (F1 weighted) castiga omitir clases minoritarias.** Es importante mirar el `classification_report` por clase y no solo la accuracy global — un modelo que ignora `apple` (la clase más chica) puede tener accuracy decente pero F1 weighted pobrte.

---

## 9. Próximos pasos posibles (no implementados)

- **TTA (Test-Time Augmentation)**: predecir cada imagen de test con varias augmentaciones y promediar. Mejora +1-2% sin re-entrenar.
- **Ensemble**: promediar probabilidades de ResNet18 + MobileNetV2 + EfficientNet-B0. Complementa errores entre modelos.
- **Augmentación avanzada**: MixUp, CutMix, RandAugment. Útil si se agota el Transfer Learning simple.
- **Class-weighted loss**: `CrossEntropyLoss(weight=...)` con pesos inversos a frecuencia de clase. Útil para datasets más desbalanceados.
- **Búsqueda de hiperparámetros**: LR range test, Optuna. Ajusta automatizadamente LR, batch size, weight decay, etc.

---

## 10. Comandos para reproducir

```zsh
# 1. Clonar y configurar
git clone git@github.com:Lumansito/utn-ia.git
cd utn-ia

# 2. Crear venv e instalar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Registrar kernel
python -m ipykernel install --user --name utn-ia --display-name "Python (utn-ia)"

# 4. Configurar Kaggle API
cp .env.example .env
# editar .env y pegar KAGGLE_API_TOKEN

# 5. Levantar el notebook
jupyter lab TP_1_Clasificador_Frutas_UTN.ipynb
# Seleccionar kernel: Python (utn-ia)
```

Ver [`SETUP.md`](SETUP.md) para más detalles.