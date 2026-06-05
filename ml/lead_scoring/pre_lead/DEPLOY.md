# Guía de despliegue — Lead Scoring (GCP)

Todo lo que tienes que hacer para desplegar, paso a paso. Proyecto destino:
**`test-ml-flow-484314`**, región **`us-central1`** (donde está la tabla copiada).

## TL;DR (si ya tienes los prerequisitos)

```bash
cd terraform && terraform init && terraform apply && cd ..   # 0) infra
./deploy/01_build_images.sh                                   # 1) imágenes
ENV=dev ./deploy/02_run_pipeline.sh                           # 2) entrenar (Vertex)
ENV=dev ./deploy/03_deploy_serving.sh                         # 3) API (Cloud Run)
```

Orden **obligatorio**: 0 → 1 → 2 → 3. El paso 3 necesita que el 2 haya dejado los
modelos en el bucket, o la API arranca pero responde 503 (sin modelos).

> **`ENV`** (`dev` por defecto \| `prod`) separa modelos y servicio por entorno **sin
> necesitar un segundo proyecto GCP**. El paso 2 entrena un *candidate* y, si no empeora
> respecto al *live* actual, lo promociona a *live* (lo que sirve la API). Detalle completo
> en el README → **Environments & model promotion**.

---

## 0. Prerequisitos (una sola vez)

### Herramientas
Comprueba que tienes todo:
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

---

## 1. Configuración (revisa antes de empezar)

Ya está todo apuntando al proyecto correcto. Solo revísalo:

| Dónde | Qué |
|---|---|
| `terraform/terraform.tfvars` | project, region, **bucket**, ar_repo |
| `deploy/config.sh` | mismos valores (los scripts leen de aquí) |
| `src/leadscoring/config.py` | mismos valores + features por segmento |

⚠️ El **nombre del bucket** (`bq-pfu-ga4-leadscoring`) es global y único. Si quieres
otro, cámbialo **en los 3 sitios** a la vez (p.ej. `test-ml-flow-484314-leadscoring`).

---

## 2. Paso 0 — Infra con Terraform

Crea las APIs, el bucket de GCS y el repositorio de imágenes (Artifact Registry).

```bash
cd terraform
terraform init        # descarga el provider de Google (1ª vez)
terraform plan        # opcional: revisa lo que va a crear
terraform apply       # escribe 'yes' para confirmar
cd ..
```
Crea: bucket `gs://bq-pfu-ga4-leadscoring`, repo Docker `lead-scoring`, y habilita
las APIs. Es idempotente (puedes re-ejecutarlo sin romper nada).

> Alternativa sin Terraform: `./deploy/00_setup_gcp.sh` hace lo mismo con gcloud.

---

## 3. Paso 1 — Construir las imágenes

Construye y sube a Artifact Registry dos imágenes (entrenamiento + serving) con
Cloud Build. Tarda ~3-6 min la primera vez.

```bash
./deploy/01_build_images.sh
```
Sube:
- `…/lead-scoring/training-base:latest` (la usan los componentes del pipeline)
- `…/lead-scoring/lead-scoring-serve:latest` (la API)

Verifica:
```bash
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/test-ml-flow-484314/lead-scoring
```

---

## 4. Paso 2 — Entrenar (Vertex AI Pipelines)

Compila y lanza el pipeline. Entrena los dos modelos (landing + main) y deja los
artefactos en el bucket.

```bash
./deploy/02_run_pipeline.sh
```
- Instala `kfp` + `google-cloud-aiplatform` en el venv (1ª vez).
- Imprime el ID del job. **Míralo en la consola**: Vertex AI → Pipelines (us-central1).
  Ahí verás el grafo, las **métricas, la curva ROC y el informe HTML de lift** por segmento.

Cuando acabe (verde), comprueba que el gate promocionó los modelos a `live/`:
```bash
gcloud storage ls gs://bq-pfu-ga4-leadscoring/models/dev/live/
# lead_scoring_landing.joblib
# lead_scoring_main.joblib
```
El paso `validate-and-promote-<segmento>` en la UI de Vertex muestra `promoted=1/0`
y un HTML con el motivo. Si un retrain empeora, el gate **NO** promociona (deja el
`live` anterior) y el pipeline sigue **verde** (gate SOFT).

> Solo compilar sin lanzar (para validar):
> `ENV=dev ./deploy/02_run_pipeline.sh --compile-only`

---

## 5. Paso 3 — Desplegar la API (Cloud Run)

```bash
ENV=dev ./deploy/03_deploy_serving.sh
```
Despliega el servicio `lead-scoring-dev` (scale-to-zero, auth privada, sirve el modelo
`live` del entorno). Al final imprime la **URL**.

Verifica:
```bash
URL=$(gcloud run services describe lead-scoring-dev --region us-central1 \
      --format='value(status.url)')

# health (debe listar landing + main)
curl -s "$URL/health" -H "Authorization: Bearer $(gcloud auth print-identity-token)"

# scorear un lead de ejemplo
curl -s -X POST "$URL/score" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H 'Content-Type: application/json' \
  -d '{"form_name":"unbounce_master","page_location":"https://x/landing/mba?utm_campaign=brand","user_studies":"es-2","language_site":"es","ga_session_number":2}'
```
Devuelve algo como:
```json
{"segmento":"landing","score":0.07,"lift_vs_base":3.0,"features_used":[...]}
```

---

## Reentrenar / actualizar el modelo

1. (Si cambian datos/variables) re-lanza el entrenamiento: `ENV=dev ./deploy/02_run_pipeline.sh`
   (entrena *candidate* → gate → promociona a *live* si no empeora)
2. ⚠️ La API **carga los modelos al arrancar**, así que NO ve los modelos nuevos
   hasta que se reinicia. Fuerza una revisión nueva:
   ```bash
   ENV=dev ./deploy/03_deploy_serving.sh   # vuelve a desplegar (recarga el modelo live)
   ```

---

## Problemas comunes

| Síntoma | Causa / arreglo |
|---|---|
| `/health` da **503** | No hay modelos en `models/<env>/live/` → corre el paso 2 (y que el gate promocione) antes del 3. |
| Modelo nuevo no se sirve | `validate-and-promote` no promocionó (mira `promoted` y el HTML en Vertex), o no redesplegaste (la API recarga al arrancar). |
| `PermissionDenied` al lanzar pipeline | No hiciste `gcloud auth application-default login`. |
| Build falla: repo no existe | No corriste Terraform / paso 0 (falta el repo de Artifact Registry). |
| Pipeline falla leyendo BigQuery | La tabla o el bucket no están en `us-central1` (deben coincidir con la región). |
| `bucket already exists` | El nombre es global; elige otro en los 3 sitios de config. |
| `curl` da 403 | Falta el token: añade `-H "Authorization: Bearer $(gcloud auth print-identity-token)"`. |

---

## Resumen de qué hace cada cosa

| Paso | Herramienta | Crea / hace |
|---|---|---|
| 0 | Terraform | bucket + Artifact Registry + APIs (infra fija) |
| 1 | Cloud Build | imágenes Docker (entrenamiento + serving) |
| 2 | Vertex Pipelines | entrena los 2 modelos → joblibs en GCS + métricas en la UI |
| 3 | Cloud Run | despliega la API de scoring (tiempo real) |

CI/CD (más adelante, con Cloud Build) automatizaría los pasos 1 y 3 en cada push.
