FROM node:24-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts eslint.config.js index.html ./
COPY src/ ./src/

ARG VITE_API_BASE_URL=/service
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build


FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist/ /usr/share/nginx/html/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["wget", "--quiet", "--output-document=-", "http://127.0.0.1:8080/"]
