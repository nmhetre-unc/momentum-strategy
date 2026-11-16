# The dashboard image: the built static site, served by nginx.
#
#   docker compose up -d --build web     (from the repo root)
#
# build:   npm ci from the lock file, then `vite build` into dist/. Type
#          checking, lint and tests are CI's job (and `npm run build`
#          locally); `tsc -b` would also check test files that read
#          ../contracts and ../openapi.json, which are outside this context.
# runtime: nginx-unprivileged, which runs as uid 101 and listens on 8080,
#          with our nginx.conf and the site. No Node at run time.
#
# Bump each tag and digest together.
ARG NODE_IMAGE=node:26.13.1-alpine3.23@sha256:39d8c143e60e99ae98f014bebea1e72fc96d3c9bdf9504bc3a27af5c05bfa39f
ARG NGINX_IMAGE=nginxinc/nginx-unprivileged:1.30.6-alpine@sha256:f5116ed4606f3eb015dd4a8e28cb9b351476dc0ad4bc6e9f93b19cec130cdf03

# ---------------------------------------------------------------------------
FROM ${NODE_IMAGE} AS build

WORKDIR /app
# Dependencies first, so editing the app doesn't reinstall them.
COPY package.json package-lock.json .npmrc ./
RUN npm ci --no-audit --no-fund
COPY index.html vite.config.ts tsconfig*.json ./
COPY src ./src
RUN npx vite build

# ---------------------------------------------------------------------------
FROM ${NGINX_IMAGE} AS runtime

COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/security-headers.conf /etc/nginx/security-headers.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
# nginx directly: the base image's entrypoint scripts try to rewrite files
# under /etc/nginx, which the read-only root filesystem doesn't allow, and
# this config needs none of them.
ENTRYPOINT []
CMD ["nginx", "-g", "daemon off;"]
