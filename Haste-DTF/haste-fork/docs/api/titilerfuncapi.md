# HASTE TiTiler Function API

A dynamic tile server deployed as an Azure Function, used by HASTE to serve Cloud Optimized GeoTIFF (COG) imagery and model inference results to the visualizer.

This deployment is adapted from the official TiTiler Azure deployment by [Development Seed](https://developmentseed.org/):
**[github.com/developmentseed/titiler/tree/main/deployment/azure](https://github.com/developmentseed/titiler/tree/main/deployment/azure)**

---

## What is TiTiler?

[TiTiler](https://developmentseed.org/titiler/) is an open-source dynamic tile server built on [FastAPI](https://fastapi.tiangolo.com/). It reads geospatial raster files (COGs, STAC items, MosaicJSONs) directly from blob storage and serves XYZ map tiles on demand — no pre-tiling required.

HASTE uses it to render:
- Pre- and post-event satellite imagery layers
- Model inference output (predicted damage GeoTIFF) with colormap overlays

---

## HASTE Modifications

The upstream deployment provides a minimal Azure Function wrapper around the TiTiler FastAPI app. The `app/__init__.py` in this directory extends that with the following changes:

### Feature flags
The upstream unconditionally registers all routers. Here, each router is conditionally included based on `ApiSettings`:

```python
if not api_settings.disable_cog:
    app.include_router(cog.router, prefix="/cog", ...)
if not api_settings.disable_stac:
    app.include_router(stac.router, prefix="/stac", ...)
if not api_settings.disable_mosaic:
    app.include_router(mosaic.router, prefix="/mosaicjson", ...)
```

### Added middleware stack
The upstream has no middleware. This deployment adds:

| Middleware | Purpose |
|-----------|---------|
| `CORSMiddleware` | Configurable allowed origins via `api_settings.cors_origins` |
| `CompressionMiddleware` | Response compression (excludes JPEG, PNG, JP2, WebP — already compressed) |
| `CacheControlMiddleware` | Cache headers via `api_settings.cachecontrol`; excludes `/healthz` |
| `LoggerMiddleware` + `TotalTimeMiddleware` | Request logging and timing (debug mode only) |
| `LowerCaseQueryStringMiddleware` | Normalizes query parameter casing (optional) |

### Added endpoints
Two endpoints not present in the upstream minimal example:

- `GET /healthz` — health check, returns `{"ping": "pong!"}`
- `GET /` — HTML landing page via TiTiler's built-in template

### Exception handlers
Explicit registration of TiTiler's COG and Mosaic error codes:

```python
add_exception_handlers(app, DEFAULT_STATUS_CODES)
add_exception_handlers(app, MOSAIC_STATUS_CODES)
```

### Route prefix removed
`host.json` sets `routePrefix: ""` to strip the default `/api` prefix that Azure Functions adds to all routes.

---

## Routes

| Prefix | Description |
|--------|-------------|
| `/cog` | Cloud Optimized GeoTIFF endpoints (tile, info, statistics, preview) |
| `/stac` | SpatioTemporal Asset Catalog endpoints |
| `/mosaicjson` | MosaicJSON mosaic endpoints |
| `/tms` | TileMatrixSets |
| `/healthz` | Health check |
| `/` | Landing page |

Full interactive API docs are available at `/docs` when running.

---

## Configuration

TiTiler is configured via environment variables through `ApiSettings` (from `titiler.application.settings`):

| Variable | Description |
|----------|-------------|
| `TITILER_API_DISABLE_COG` | Set to `true` to disable COG endpoints |
| `TITILER_API_DISABLE_STAC` | Set to `true` to disable STAC endpoints |
| `TITILER_API_DISABLE_MOSAIC` | Set to `true` to disable MosaicJSON endpoints |
| `TITILER_API_CORS_ORIGINS` | Comma-separated list of allowed CORS origins |
| `TITILER_API_CACHECONTROL` | Cache-Control header value for tile responses |
| `TITILER_API_DEBUG` | Set to `true` to enable request logging middleware |
| `TITILER_API_LOWER_CASE_QUERY_PARAMETERS` | Set to `true` to normalize query param casing |

---

## Development Setup

### Prerequisites

- Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
- Azure Functions Core Tools:

```bash
npm install -g azure-functions-core-tools@4 --unsafe-perm true
```

### Running Locally

```bash
func start
```

### Deployment

```bash
func azure functionapp publish {your-function-app-name} --python
```

For full deployment instructions see the upstream docs:
https://docs.microsoft.com/en-us/azure/azure-functions/create-first-function-cli-python
