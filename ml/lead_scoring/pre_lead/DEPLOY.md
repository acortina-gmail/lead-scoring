# Guía de despliegue — Lead Scoring (GCP)

Todo lo que tienes que hacer para **desplegar, actualizar y destruir**, paso a paso.
Proyecto destino: **`test-ml-flow-484314`**, región **`us-central1`** (donde está la tabla).

Esta guía cubre los **tres escenarios**:
- **[A) Desde 0](#a-desplegar-desde-0)** — no hay nada en GCP.
- **[B) Actualizar](#b-actualizar-cuando-ya-hay-algo-desplegado)** — ya hay algo desplegado.
- **[C) Destruir](#c-destruir-todo)** — borrar los recursos.

Todos los comandos se ejecutan **desde `ml/lead_scoring/pre_lead/`**.

| Pieza | Qué es | Se crea/actualiza con |
|---|---|---|
| Infra | bucket GCS + Artifact Registry + APIs + alerta de fallo | Terraform (`terraform/`) |
| Imágenes | `training-base` (pipeline) + `lead-scoring-serve` (API) | `deploy/01_build_images.sh` |
| Modelos | 2 joblibs (`landing`+`main`) en `gs://…/models/<env>/live/` | `deploy/02_run_pipeline.sh` (Vertex) |
| API | servicio Cloud Run `lead-scoring-<env>` | `deploy/03_deploy_serving.sh` |

> **`ENV`** (`dev` por defecto \| `prod`) separa modelos y servicio por entorno **sin
> necesitar un segundo proyecto GCP**. Para testear te basta `dev`; `prod` es para
> producción real. El paso 2 entrena un *candidate* y, si no empeora respecto al *live*
> actual, lo promociona a *live* (lo que sirve la API). Detalle en el README →
> **Environments & model promotion**.

## TL;DR (desde 0, con prerequisitos ya hechos)

```bash
cd terraform && terraform init && terraform apply && cd ..   # 0) infra
./deploy/01_build_images.sh                                   # 1) imágenes
ENV=dev ./deploy/02_run_pipeline.sh                           # 2) entrenar (Vertex)
ENV=dev ./deploy/03_deploy_serving.sh                         # 3) API (Cloud Run)
```
Orden **obligatorio** `0 → 1 → 2 → 3`. El paso 3 necesita que el 2 haya dejado los
modelos en `live/`, o la API arranca pero responde `503` (sin modelos).

---

## 0. Prerequisitos (una sola vez)

### Herramientas
```bash
gcloud --version        # Google Cloud CLI
bq version              # BigQuery CLI
terraform version       # >= 1.5
./.venv/bin/python -V   # Python 3.12 (el venv del repo)
```
Si falta Terraform: `brew install terraform`.

### Login y proyecto
```bash
gcloud auth login                                  # tu cuenta (acortina@knowmadmood.com)
gcloud auth application-default login              # ADC — lo usa el envío del pipeline a Vertex
gcloud config set project test-ml-flow-484314      # proyecto activo
```
> El `application-default login` es importante: el SDK de Vertex (paso 2) usa esas
> credenciales (ADC), no las de `gcloud auth login`.

### Comprobaciones rápidas
```bash
# ¿tienes permisos en el proyecto? (debe salir roles/editor)
gcloud projects get-iam-policy test-ml-flow-484314 \
  --flatten="bindings[].members" \
  --filter="bindings.members:$(gcloud config get-value account)" \
  --format="value(bindings.role)"

# ¿está la tabla de entrenamiento?
bq show test-ml-flow-484314:dataset.lead_scoring_train
```

### Configuración (revisa antes de empezar)
Ya está todo apuntando al proyecto correcto. Solo revísalo:

| Dónde | Qué |
|---|---|
| `terraform/terraform.tfvars` | project, region, **bucket**, ar_repo, `alert_emails` |
| `deploy/config.sh` | mismos valores (los scripts leen de aquí) |
| `src/leadscoring/config.py` | mismos valores + features por segmento |

⚠️ El **nombre del bucket** (`bq-pfu-ga4-leadscoring`) es global y único. Si quieres
otro, cámbialo **en los 3 sitios** a la vez (p.ej. `test-ml-flow-484314-leadscoring`).

---

## A) Desplegar desde 0

### Paso 0 — Infra con Terraform
Crea las APIs, el bucket de GCS, el repo de imágenes (Artifact Registry) y la alerta de
email si falla el pipeline.

```bash
cd terraform
terraform init        # descarga el provider de Google (1ª vez)
terraform plan        # opcional: revisa lo que va a crear
terraform apply       # escribe 'yes' para confirmar
cd ..
```
Es idempotente (puedes re-ejecutarlo sin romper nada).

> Alternativa sin Terraform: `./deploy/00_setup_gcp.sh` hace lo mismo con gcloud
> (pero **sin** la alerta de fallo de pipeline, que solo está en Terraform).

### Paso 1 — Construir las imágenes
Construye y sube a Artifact Registry las dos imágenes (entrenamiento + serving) con
Cloud Build. Tarda ~3-6 min la primera vez.

```bash
./deploy/01_build_images.sh
```
Sube `…/training-base:latest` (componentes del pipeline) y `…/lead-scoring-serve:latest`
(la API). Verifica:
```bash
gcloud artifacts docker images list us-central1-docker.pkg.dev/test-ml-flow-484314/lead-scoring
```

### Paso 2 — Entrenar (Vertex AI Pipelines)
Compila y lanza el pipeline; entrena los dos modelos y deja los artefactos en el bucket.

```bash
ENV=dev ./deploy/02_run_pipeline.sh
```
- Instala `kfp` + `google-cloud-aiplatform` en el venv (1ª vez).
- Imprime el ID del job. **Míralo en la consola**: Vertex AI → Pipelines (us-central1).
  Verás el grafo, las **métricas, la curva ROC y el informe HTML** por segmento (con el
  **PR-AUC** destacado y los **grados A/B/C** por lead).

Cuando acabe (verde), comprueba que el gate promocionó los modelos a `live/`:
```bash
gcloud storage ls gs://bq-pfu-ga4-leadscoring/models/dev/live/
# lead_scoring_landing.joblib
# lead_scoring_main.joblib
```
El paso `validate-and-promote-<segmento>` muestra `promoted=1/0` y un HTML con el motivo.
Si un retrain empeora, el gate **NO** promociona (deja el `live` anterior) y el pipeline
sigue **verde** (gate **SOFT**).

> Solo compilar sin lanzar (validar): `ENV=dev ./deploy/02_run_pipeline.sh --compile-only`

### Paso 3 — Desplegar la API (Cloud Run)
```bash
ENV=dev ./deploy/03_deploy_serving.sh
```
Despliega `lead-scoring-dev` (scale-to-zero, auth privada, sirve el modelo `live` del
entorno). Imprime la **URL**. Verifica:
```bash
URL=$(gcloud run services describe lead-scoring-dev --region us-central1 --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

curl -s "$URL/health" -H "Authorization: Bearer $TOKEN"          # debe listar landing + main

curl -s -X POST "$URL/score" -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"form_name":"unbounce_master","platform":"WEB","page_name":"unbounce/mba","page_location":"https://x/landing/mba?utm_campaign=brand","product_id":123,"user_province":"Barcelona","user_studies":"es-2","ga_session_number":2}'
```
Devuelve algo como:
```json
{"segmento":"landing","score":0.07,"grade":"A","base_rate":0.023,"lift_vs_base":3.0,"features_used":[...]}
```

> **Campos del payload — enrutado y features:** hoy el segmento (landing/main) se enruta
> por `segmento` explícito o, si falta, por `form_name` (empieza por `unbounce` → landing).
> En el **deploy real** el enrutado pasará a usar **`platform`** (lo pediste al cliente),
> por eso el ejemplo ya lo incluye — habrá que adaptar `route_segment` (pendiente). Además,
> manda siempre los campos que son **feature** del modelo o se verán como `MISSING`:
> `main` usa `form_name`, `page_name`, `product_id`, `user_country`, `user_province`,
> `user_studies`, `ga_session_number`; `landing` usa `page_name`, `language_site`,
> `utm_campaign` (de `page_location`), `user_studies`, `ga_session_number`.

---

## B) Actualizar (cuando ya hay algo desplegado)

Qué reconstruir depende de **qué cambiaste**. Regla: si tocaste el código de un proceso,
hay que **reconstruir su imagen** antes de relanzarlo.

| Cambiaste… | 01 build | 02 train | 03 deploy |
|---|:--:|:--:|:--:|
| Solo datos (BigQuery) | — | ✅ | — (auto-reload / `/reload`) |
| `serving/app.py` o `leadscoring` (serving) | ✅ | — | ✅ |
| `pipelines/` o `leadscoring` (entreno) | ✅ | ✅ | — |
| Infra (`terraform/`) | — | — | — → `terraform apply` |

### B1. Reentrenar el modelo (datos nuevos, mismo código)
```bash
ENV=dev ./deploy/02_run_pipeline.sh        # entrena candidate → gate → live
```
- El **caching está OFF** por defecto, así que un re-lanzamiento **siempre entrena** (el
  caching de Vertex se basa en la *referencia* de la tabla, no en su contenido).
- **No hace falta redeploy:** la API re-chequea GCS cada ~5 min (`MODEL_RELOAD_CHECK_SECONDS`)
  y hace *hot-swap* del nuevo `live` sola. Para que lo coja **al instante**:
  ```bash
  curl -s -X POST "$URL/reload" -H "Authorization: Bearer $TOKEN"
  ```

### B2. Cambiaste código de **serving** (`serving/app.py`, `src/leadscoring/`)
La imagen de serving lleva el código dentro → reconstruir y redeployar:
```bash
./deploy/01_build_images.sh
ENV=dev ./deploy/03_deploy_serving.sh
```

### B3. Cambiaste código del **pipeline/entreno** (`pipelines/`, `src/leadscoring/`)
La imagen `training-base` lleva el código → reconstruir y reentrenar:
```bash
./deploy/01_build_images.sh
ENV=dev ./deploy/02_run_pipeline.sh
```

### B4. Promocionar dev → prod
Repite **2 y 3** con `ENV=prod` (entrena/promociona el modelo prod y levanta el servicio
`lead-scoring-prod`):
```bash
ENV=prod ./deploy/02_run_pipeline.sh
ENV=prod ./deploy/03_deploy_serving.sh
```

### B5. Vía CI/CD (GitHub Actions) — automático
- **PR**: corre `ruff` + `pytest` (`.github/workflows/ci.yml`).
- **Merge a `main`** (cambios en `ml/lead_scoring/pre_lead/**`): build + deploy a **dev**
  automático; **prod** queda detrás de una **aprobación manual** (GitHub Environment `prod`).
- **Reentrenar**: workflow manual `train.yml` (`workflow_dispatch`, elige `dev`/`prod`).
- Requiere el secret `GCP_SA_KEY` en el repo. Detalle en [`deploy/CICD.md`](deploy/CICD.md).

---

## C) Destruir todo

Un script borra en el orden correcto (Cloud Run → bucket + Artifact Registry →
opcionalmente la clave de CI):

```bash
./deploy/99_teardown.sh            # pide confirmación (teclea el project id)
./deploy/99_teardown.sh --yes      # sin preguntar
```

**Borra:** servicios Cloud Run (`lead-scoring-dev` *y* `lead-scoring-prod`), bucket
`gs://bq-pfu-ga4-leadscoring` (con modelos + artefactos), repo Artifact Registry (con
imágenes). Usa `terraform destroy` si hay estado; si no, borra con `gcloud`.

**NO toca:** la tabla de BigQuery `dataset.lead_scoring_train` (tus datos), la service
account por defecto, ni las APIs.

**Clave de CI (opcional):** por defecto la deja (la usa GitHub Actions). Para borrarla:
```bash
./deploy/99_teardown.sh --yes --delete-sa-keys   # ⚠️ rompe el deploy por GitHub Actions
```
Si la borras, quita también el secret `GCP_SA_KEY` del repo de GitHub.

> Solo bajar la API (sin borrar datos/imágenes):
> `gcloud run services delete lead-scoring-dev --region us-central1`.

---

## Problemas comunes

| Síntoma | Causa / arreglo |
|---|---|
| `/health` o `/score` da **503** | No hay modelos en `models/<env>/live/` → corre el paso 2 (y que el gate promocione) antes del 3. |
| El modelo nuevo no se sirve | Espera ~5 min o haz `POST /reload`; revisa que `validate-and-promote` hizo `promoted=1` (mira el HTML en Vertex). |
| Cambié serving y no se refleja | Rebuild (`01`) + redeploy (`03`): el código va dentro de la imagen, no se recarga solo. |
| `PermissionDenied` al lanzar pipeline | No hiciste `gcloud auth application-default login`. |
| Build falla: repo no existe | No corriste Terraform / paso 0 (falta el repo de Artifact Registry). |
| Pipeline falla leyendo BigQuery | La tabla o el bucket no están en `us-central1` (deben coincidir con la región). |
| `bucket already exists` | El nombre es global; elige otro en los 3 sitios de config. |
| `curl` da 403 | Falta el token: añade `-H "Authorization: Bearer $(gcloud auth print-identity-token)"`. |

---

## Resumen de qué hace cada cosa

| Paso | Herramienta | Crea / hace |
|---|---|---|
| 0 | Terraform | bucket + Artifact Registry + APIs + alerta de fallo (infra) |
| 1 | Cloud Build | imágenes Docker (entrenamiento + serving) |
| 2 | Vertex Pipelines | entrena los 2 modelos → joblibs en GCS + métricas/HTML en la UI |
| 3 | Cloud Run | despliega la API de scoring (tiempo real) |

CI/CD (GitHub Actions) automatiza **1 y 3** en cada merge a `main` (dev automático, prod
con aprobación) y el reentreno con un workflow manual — ver `deploy/CICD.md`.
