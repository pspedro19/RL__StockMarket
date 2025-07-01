# 🚀 Deployment

Archivos para despliegue y containerización del sistema de trading.

## 📁 Contenido

### 🐳 Docker
- `Dockerfile` - Imagen Docker del sistema
- `docker-compose.yml` - Orquestación de servicios

## 🔧 Uso

### Docker Simple
```bash
# Construir imagen
docker build -t rl-trading-system .

# Ejecutar contenedor
docker run -p 8000:8000 rl-trading-system
```

### Docker Compose
```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

### Servicios Incluidos
- **trading-api**: API principal del sistema
- **postgres**: Base de datos
- **redis**: Cache y colas
- **grafana**: Monitoreo
- **prometheus**: Métricas

## 🌐 Endpoints

Una vez desplegado:
- API: http://localhost:8000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## 📋 Variables de Entorno

Crear archivo `.env`:
```env
POSTGRES_DB=trading_db
POSTGRES_USER=trader
POSTGRES_PASSWORD=secure_password
REDIS_URL=redis://redis:6379
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

## 🔒 Producción

Para producción, modificar:
- Passwords y secretos
- Volúmenes persistentes
- Configuración de red
- SSL/TLS certificates 