# Smart Market Analyzer: Production Setup

This guide provides instructions for deploying and operating the Smart Market Analyzer in a production environment.

## System Architecture

The production version of Smart Market Analyzer consists of the following components:

1. **API Service**: FastAPI-based REST API that provides endpoints for data processing, model training, predictions, signals, and backtesting.
2. **Worker Service**: Background worker that handles real-time data processing, model inference, and trade execution.
3. **Redis**: Used for caching, rate limiting, and as a message broker.
4. **MongoDB**: Persistent storage for market data, model states, and trading results.
5. **Prometheus & Grafana**: Monitoring and visualization of system metrics.

## Production Enhancements

The production version includes the following enhancements:

### 1. Data Management
- Real-time market data integration via NinjaTrader API
- Efficient data caching and persistence
- Data validation and error handling

### 2. Portfolio and Risk Management
- Multi-instrument portfolio management
- Position sizing based on volatility and risk parameters
- Correlation-aware risk management
- Drawdown controls and risk limits

### 3. Deployment and Scaling
- Containerized deployment with Docker and Docker Compose
- Horizontally scalable architecture
- Resource allocation and monitoring

### 4. Monitoring and Observability
- Comprehensive logging
- Prometheus metrics for performance monitoring
- Grafana dashboards for visualization
- Error tracking and alerting

### 5. Reliability and Fault Tolerance
- Automatic recovery from failures
- State persistence and recovery
- Circuit breakers for external dependencies

## Deployment Instructions

### Prerequisites
- Docker and Docker Compose installed
- NinjaTrader account with API access (for live trading)
- Minimum 8GB RAM and 4 CPU cores recommended

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/smart-market-analyzer.git
cd smart-market-analyzer
```

2. Configure environment:
```bash
# Copy sample config
cp config.sample.json config.json

# Edit config with your settings
nano config.json
```

3. Build and start the services:
```bash
docker-compose up -d
```

4. Access the API documentation:
```
http://localhost:8000/docs
```

5. Access Grafana dashboards:
```
http://localhost:3000
Username: admin
Password: smartmarket
```

### Production Configuration

Key configuration parameters in `config.json`:

```json
{
  "api": {
    "rate_limiting": {
      "enabled": true,
      "limits": {
        "default": 60,
        "/predict": 120,
        "/backtest": 10
      }
    },
    "cors": {
      "origins": ["https://yourdomain.com"]
    }
  },
  "data": {
    "cache_ttl": 300,
    "data_sources": {
      "ninjatrader": {
        "enabled": true,
        "api_key": "YOUR_API_KEY",
        "base_url": "https://api.ninjatrader.com"
      }
    }
  },
  "portfolio": {
    "initial_capital": 100000,
    "risk_per_trade": 0.02,
    "max_correlated_risk": 0.06,
    "max_drawdown": 0.15
  },
  "trading": {
    "min_confidence": 0.75,
    "min_time_between_trades": 300,
    "use_trailing_stops": true,
    "trading_hours": {
      "monday": [{"start": "09:30", "end": "16:00"}],
      "tuesday": [{"start": "09:30", "end": "16:00"}],
      "wednesday": [{"start": "09:30", "end": "16:00"}],
      "thursday": [{"start": "09:30", "end": "16:00"}],
      "friday": [{"start": "09:30", "end": "16:00"}]
    }
  },
  "monitoring": {
    "prometheus": {
      "enabled": true,
      "port": 8001
    },
    "log_dir": "logs",
    "health_check_interval": 60,
    "services_to_check": {
      "ninjatrader_api": "https://api.ninjatrader.com/health"
    }
  }
}
```

## Operations Guide

### Monitoring

1. **System Health**: 
   - Check the health endpoint: `http://localhost:8000/health`
   - View system metrics in Grafana: `http://localhost:3000/d/smart-market-health`

2. **Performance Metrics**:
   - Model accuracy and latency
   - Trading performance (win rate, Sharpe ratio, drawdown)
   - API response times

3. **Logs**:
   - Application logs are stored in `./logs`
   - View logs with: `docker-compose logs -f api worker`

### Backup and Restore

1. **Backup data**:
```bash
# Backup MongoDB
docker-compose exec mongo mongodump --out /data/db/backup
# Copy backup files
docker cp smart-market-analyzer_mongo_1:/data/db/backup ./backup/mongo

# Backup Redis
docker-compose exec redis redis-cli SAVE
# Copy backup file
docker cp smart-market-analyzer_redis_1:/data/dump.rdb ./backup/redis/
```

2. **Restore data**:
```bash
# Restore MongoDB
docker cp ./backup/mongo smart-market-analyzer_mongo_1:/data/db/backup
docker-compose exec mongo mongorestore /data/db/backup

# Restore Redis
docker cp ./backup/redis/dump.rdb smart-market-analyzer_redis_1:/data/
docker-compose restart redis
```

### Scaling

1. **Horizontal scaling**:
```bash
# Scale worker service
docker-compose up -d --scale worker=3
```

2. **Vertical scaling**:
   - Update resource limits in `docker-compose.yml`
   - Apply changes: `docker-compose up -d`

### Troubleshooting

1. **API Issues**:
   - Check logs: `docker-compose logs api`
   - Verify health endpoint: `curl http://localhost:8000/health`
   - Restart service: `docker-compose restart api`

2. **Worker Issues**:
   - Check logs: `docker-compose logs worker`
   - Check worker status in MongoDB: `db.worker_status.find()`
   - Restart service: `docker-compose restart worker`

3. **Database Issues**:
   - Check MongoDB logs: `docker-compose logs mongo`
   - Check Redis logs: `docker-compose logs redis`
   - Verify connections: `docker-compose exec mongo mongo --eval "db.stats()"`
   - Check Redis status: `docker-compose exec redis redis-cli INFO`

## Security Considerations

### API Security

1. **Authentication**:
   - JWT-based authentication is configured by default
   - API keys for machine-to-machine communication
   - Regular key rotation recommended (every 90 days)

2. **Network Security**:
   - Configure firewall rules to restrict access to API endpoints
   - Use HTTPS in production with valid SSL certificates
   - Add to `docker-compose.yml`:
     ```yaml
     api:
       environment:
         - SSL_CERT_PATH=/app/certs/cert.pem
         - SSL_KEY_PATH=/app/certs/key.pem
     volumes:
       - ./certs:/app/certs
     ```

3. **Data Protection**:
   - All sensitive data (API keys, credentials) encrypted at rest
   - Database encryption configured by default
   - Logs sanitized to remove sensitive information

### System Security

1. **Container Security**:
   - Regular security updates: `docker-compose pull && docker-compose up -d`
   - Non-root users in containers
   - Resource limits to prevent DoS

2. **Dependencies**:
   - Regular dependency updates
   - Vulnerability scanning: `docker-compose exec api pip-audit`
   - Minimal image size with multi-stage builds

## Performance Tuning

### Model Optimization

1. **Inference Performance**:
   - Optimize batch size for predictions (default: 32)
   - Enable TensorFlow optimizations: `TF_XLA_FLAGS=--tf_xla_auto_jit=2`
   - GPU acceleration if available

2. **Memory Management**:
   - Configure model cache size in `config.json`:
     ```json
     "model": {
       "cache_size_mb": 512,
       "precision": "float16"
     }
     ```

### Database Optimization

1. **MongoDB Indexes**:
   - Create indexes for frequently queried fields:
     ```bash
     docker-compose exec mongo mongo --eval '
       db.market_data.createIndex({"instrument": 1, "timestamp": 1});
       db.predictions.createIndex({"instrument": 1, "timestamp": 1});
       db.signals.createIndex({"instrument": 1, "timestamp": 1});
     '
     ```

2. **Redis Configuration**:
   - Tune memory settings in `redis.conf`:
     ```
     maxmemory 1gb
     maxmemory-policy allkeys-lru
     ```

## High Availability Setup

For mission-critical deployments, a high-availability setup is recommended:

1. **Database Replication**:
   - MongoDB replica set
   - Redis sentinel or cluster

2. **Load Balancing**:
   - Use a reverse proxy (NGINX, Traefik) for API load balancing
   - Configure health checks for automatic failover

3. **Multi-Region Deployment**:
   - Deploy in multiple data centers or cloud regions
   - Geo-distributed database replicas

Example NGINX configuration for load balancing:
```nginx
upstream smart_market_api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    server_name api.smartmarket.com;

    location / {
        proxy_pass http://smart_market_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Production Checklist

Before going live, ensure:

1. **Security**:
   - All default credentials changed
   - Firewall rules in place
   - SSL/TLS configured
   - API authentication enabled

2. **Monitoring**:
   - Alerting rules configured
   - Logging retention policy set
   - Uptime monitoring enabled

3. **Backup**:
   - Automated backup schedule established
   - Backup verification procedure in place
   - Disaster recovery plan documented

4. **Documentation**:
   - System architecture documented
   - Runbooks for common operations
   - Incident response procedures

5. **Testing**:
   - Load testing completed
   - Failover testing performed
   - Security testing/audit completed

## Continuous Improvement

1. **Model Performance**:
   - Monitor prediction accuracy metrics
   - Implement A/B testing framework for model updates
   - Automate retraining on performance degradation

2. **System Performance**:
   - Regular performance reviews
   - Capacity planning based on usage patterns
   - Periodic optimization of queries and code

3. **Feedback Loop**:
   - Track trading performance metrics
   - Implement automated performance reporting
   - Review and tune risk parameters based on results

---

For additional support, contact the development team at support@smartmarketanalyzer.com or open an issue on the GitHub repository.