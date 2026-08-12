# Stage 1: Build Vite React App
FROM mcr.microsoft.com/azurelinux/base/nodejs:20.14 AS builder

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY ../../UI/package*.json ./
RUN npm ci || npm install

# Copy source code and build production assets
COPY ../../UI .
RUN npm run build

# Stage 2: Ultra-slim production web server (~25 MB)
FROM nginx:1.25-alpine AS production

# Copy compiled static distribution from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Add Single Page Application (SPA) fallback routing configuration
RUN printf 'server { listen 5000; location / { root /usr/share/nginx/html; try_files $uri $uri/ /index.html; } }\n' > /etc/nginx/conf.d/default.conf

EXPOSE 5000

CMD ["nginx", "-g", "daemon off;"]