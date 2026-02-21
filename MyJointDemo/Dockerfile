# ── build stage ──
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# ── serve stage ──
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
RUN mkdir -p /etc/nginx/templates/
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 80
CMD ["/start.sh"]
