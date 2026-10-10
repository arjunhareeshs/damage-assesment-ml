# Build from the REPO ROOT (haste-fork/), same as every other image in this
# docker/ directory -- COPY paths below are relative to that context, not to
# this Dockerfile's own directory (Docker always resolves COPY against the
# build context root; `../../` paths reaching outside the context are
# rejected outright, which is what this file did before this fix). The
# source folder is also lowercase `ui/`, not `UI/` -- the previous casing
# only worked by accident on Windows' case-insensitive filesystem and would
# have failed on ACR's Linux build agents:
#   az acr build --registry <name> --image hasteui:latest --file docker/ui/Dockerfile .
#
# Stage 1: Build Vite React App
#
# Node 24, not 20.14: vite@8 requires Node 20.19+/22.12+ and errors out on
# the 20.14.0 this image previously pinned ("Vite requires Node.js version
# 20.19+ or 22.12+").
FROM mcr.microsoft.com/azurelinux/base/nodejs:24 AS builder

WORKDIR /app

# Copy dependency manifests first for layer caching. package-lock.json is
# NOT copied/used here: it was generated on a non-Linux dev machine, so it
# doesn't record the `linux-x64-gnu` optional native binding for
# rolldown/vite. Both `npm ci` and a plain `npm install` still consult an
# on-disk lockfile's recorded optional-dependency platforms and silently
# produce a build missing that binding either way (only failing later at
# `vite build` runtime: "Cannot find module '@rolldown/binding-linux-x64-gnu'"
# / npm/cli#4828) -- npm's own documented fix for that bug is to remove
# package-lock.json (and node_modules) and reinstall, which is what not
# copying the lockfile into this stage achieves.
COPY ui/package.json ./
RUN npm install

# Copy source code and build production assets
COPY ui/ .
# npm ci's install of node_modules/.bin/* loses its executable bit on this
# base image/build-agent combo (observed on ACR's Linux build agents),
# so `npm run build` fails with "vite: Permission denied" without this.
RUN chmod -R +x node_modules/.bin \
    && npm run build

# Stage 2: Ultra-slim production web server (~25 MB)
FROM nginx:1.25-alpine AS production

# Copy compiled static distribution from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Add Single Page Application (SPA) fallback routing configuration
RUN printf 'server { listen 5000; location / { root /usr/share/nginx/html; try_files $uri $uri/ /index.html; } }\n' > /etc/nginx/conf.d/default.conf

EXPOSE 5000

CMD ["nginx", "-g", "daemon off;"]