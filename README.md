version: '3.7'
services:
  training_collector:
    restart: always
    container_name: training_collector
    build: application
    networks:
      - default
    ports:
      - "8000:8000"
      - "8080:8080"
    volumes:
      - type: bind
        source: cifar-10-batches-py
        target: /cifar-10-batches-py

networks:
  default:
    driver: bridge
    name: custom_fl
