// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// In dev compose, hastefuncapi returns blob URLs with the docker-internal
// hostname `azurite` (e.g. http://azurite:10000/devstoreaccount1/...). That's
// what the function-app code path needs server-side, but the *browser* can't
// resolve `azurite` — it's a docker network DNS name. Azurite is bound to
// host port 10000 in docker-compose, so the same blob is reachable from the
// browser at http://localhost:10000/... (or 127.0.0.1).
//
// Most blob fetches in the UI go through api-proxy/titiler server-side and
// never hit this. The PMTiles archive used by the Interactive Labeler is the
// exception: pmtiles.js issues HTTP range requests directly from the browser,
// so the URL we hand it must be browser-resolvable.
//
// This helper is a no-op in production (URL has no `azurite` hostname) and in
// non-local dev (UI not running on localhost). It only rewrites the host when
// the UI is on localhost/127.0.0.1 AND the URL points at the docker name.
export function toBrowserBlobUrl(url) {
  if (typeof url !== "string" || !url) return url;
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  if (parsed.hostname !== "azurite") return url;
  if (typeof window === "undefined") return url;
  const localHosts = new Set(["localhost", "127.0.0.1"]);
  if (!localHosts.has(window.location.hostname)) return url;
  parsed.hostname = window.location.hostname;
  return parsed.toString();
}

// Interactive Labeler imagery tiles are served by titiler. Tile-URL templates
// are persisted with the deployment-relative `/api/titiler/<path>` prefix,
// which in the cloud is routed by APIM (and in docker by the nginx api-proxy).
// Running the UI locally via `swa start` there is no such route, so imagery
// 404s. When VITE_TITILER_URL is set AND the UI is on localhost, rewrite that
// prefix to a directly-reachable titiler base. No-op in production (non-local
// host) and when the env var is unset, so it's safe to ship.
export function toBrowserTitilerUrl(url) {
  if (typeof url !== "string" || !url) return url;
  if (typeof window === "undefined") return url;
  const localHosts = new Set(["localhost", "127.0.0.1"]);
  if (!localHosts.has(window.location.hostname)) return url;
  const base = import.meta.env.VITE_TITILER_URL;
  if (!base) return url;
  const marker = "/api/titiler/";
  const idx = url.indexOf(marker);
  if (idx === -1) return url;
  return base.replace(/\/+$/, "") + "/" + url.slice(idx + marker.length);
}
