#!/bin/bash
set -e

echo "Waiting for database..."
sleep 5

echo "Initializing Airflow database..."
airflow db init || echo "Database already initialized"

echo "Creating/updating admin user..."
# Delete existing admin user if it exists, then create with fixed password
airflow users delete admin 2>&1 || echo "Admin user does not exist, will create new one"

# Create admin user with password 'admin'
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>&1 || echo "Admin user created/updated"

echo "========================================="
echo "Airflow Admin Credentials:"
echo "Username: admin"
echo "Password: admin"
echo "========================================="

echo "Starting Airflow standalone..."
exec airflow standalone
