# 🛠️ Setup del entorno

Guía para instalar dependencias, configurar credenciales Kaggle y levantar el notebook en local.

---

## 1. Requisitos previos

- **Python 3.10+** (probado con 3.12)
- **Git**
- Una cuenta en [Kaggle](https://www.kaggle.com) con API token generado

> Verificá Python:
> ```zsh
> python3 --version
> ```

---

## 2. Clonar el repositorio

```zsh
git clone git@github.com:Lumansito/utn-ia.git
cd utn-ia
```

---

## 3. Crear el entorno virtual

Mac/Linux:
```zsh
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Verificá que el prompt muestre `(.venv)`.

---

## 4. Instalar dependencias

```zsh
pip install --upgrade pip
pip install -r requirements.txt
```

Esto instala: `torch`, `torchvision`, `kaggle`, `scikit-learn`, `seaborn`, `matplotlib`, `pandas`, `pillow`, `numpy`, `jupyter`, `ipykernel`, `python-dotenv`.

---

## 5. Registrar el kernel de Jupyter

Para que el notebook use el venv como kernel:

```zsh
python -m ipykernel install --user --name utn-ia --display-name "Python (utn-ia)"
```

Esto se hace **una sola vez**. El notebook ya tiene configurado el kernelspec `utn-ia`.

---

## 6. Configurar credenciales de Kaggle

1. Ir a [kaggle.com](https://www.kaggle.com) → tu perfil → **Settings** → **API** → **Create New Token**
2. Copiar el contenido del archivo `kaggle.json` (o el `API TOKEN`).
3. Crear un archivo `.env` en la raíz del proyecto (no se commitea, está en `.gitignore`):

   ```zsh
   cp .env.example .env
   ```

4. Editar `.env` y pegar tu token:

   ```env
   KAGGLE_API_TOKEN=tu_token_real_aqui
   ```

> ⚠️ **Nunca commitear el `.env` ni `kaggle.json`.** Ambos están ignorados por `.gitignore`.

Alternativa (credenciales clásicas de Kaggle): ubicar `kaggle.json` en `~/.kaggle/kaggle.json` con permisos `600`:
```zsh
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

---

## 7. Levantar el notebook

```zsh
source .venv/bin/activate       # activar venv (si no lo está)
jupyter notebook TP_1_Clasificador_Frutas_UTN.ipynb
# o:
jupyter lab TP_1_Clasificador_Frutas_UTN.ipynb
```

Al abrir, verificar que el kernel seleccionado sea **`Python (utn-ia)`** (Kernel → Change kernel → `Python (utn-ia)`).

---

## 8. Estructura del proyecto

```
utn-ia/
├── .env                      # credenciales (IGNORADO, no commitear)
├── .env.example              # template para .env (commiteable)
├── .gitignore
├── requirements.txt
├── SETUP.md                  # este archivo
├── TP_1_Clasificador_Frutas_UTN.ipynb
└── data/                     # datasets descargados (IGNORADO, se crea al correr el notebook)
    └── fruit_recognition/
```

---

## 9. Uso del notebook (solo Etapa 2)

El notebook está acotado a la **Etapa 2** (dataset realista *UTN-IA 2026 · Food Classification - Training Set*). Flujo:

1. **Celda de configuración**: carga `.env` con `load_dotenv()` → lee `KAGGLE_API_TOKEN`.
2. **Descarga del dataset**: usa `!kaggle datasets download -d ...` a `./data/fruit_recognition/`.
3. **Split train/val/test** (70/15/15) → `./data/fruit_recognition_splint/`.
4. **Entrenamiento** con `FrutasCNN` y augmentation agresivo.
5. **Curvas de aprendizaje** y evaluación.

> 💡 Las celdas con `!kaggle ...` usan magias de shell dentro de Jupyter. Funcionan tanto en local como en Colab.

---

## 10. Comandos rápidos (resumen)

```zsh
# Setup completo desde cero
python3 -m venv .venv &&
source .venv/bin/activate &&
pip install -r requirements.txt &&
python -m ipykernel install --user --name utn-ia --display-name "Python (utn-ia)" &&
cp .env.example .env && $EDITOR .env &&          # editar .env con tu token
jupyter lab TP_1_Clasificador_Frutas_UTN.ipynb
```

---

## 11. Problemas comunes

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'dotenv'` | `pip install python-dotenv` (ya está en `requirements.txt`) |
| `KAGGLE_API_TOKEN` vacío | Verificar que `.env` exista y tenga el token, y que la celda corra `load_dotenv()` |
| Kernel `Python (utn-ia)` no aparece | Repetir paso 5 (registrar kernel) |
| `kaggle: command not found` | `pip install kaggle` y reactivar el venv |
| Torch lento en Mac (sin GPU) | Normal: en Mac no hay CUDA. Para debuggear bajá `EPOCHS` a 2-3 |
| Lento de arranque en Mac Intel (sin GPU) | Considerá usar Colab con GPU T4 para el entrenamiento final |
| ¿Usa GPU en Mac? | Sí — Apple Silicon (M1/M2/M3/M4/M5) usa **MPS** (Metal), detectado automáticamente |

---

## 12. Colab (alternativa)

Si corrés el notebook en Google Colab con GPU:
1. Subir el notebook a Colab.
2. Cambiar las rutas `./data/...` de vuelta a `/content/data/...` (o usar `/content/data` que ya está).
3. Pegar el `KAGGLE_API_TOKEN` directamente en la celda (en Colab no hay `.env`).
4. `Entorno de ejecución → Cambiar tipo → GPU T4`.