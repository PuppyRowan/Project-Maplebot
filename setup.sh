#!/bin/bash

# Exit on error
set -e

echo "Setting up Maplebot..."

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    read -p "Enter your Discord bot token: " token
    echo "DISCORD_TOKEN=$token" > .env
    echo "BOT_PREFIX=!" >> .env
    echo "LOG_LEVEL=INFO" >> .env
fi

# Load environment variables into systemd service
echo "Configuring systemd service with environment variables..."
sudo mkdir -p /etc/systemd/system/maplebot.service.d
echo "[Service]" | sudo tee /etc/systemd/system/maplebot.service.d/override.conf
while IFS= read -r line; do
    echo "Environment=\"$line\"" | sudo tee -a /etc/systemd/system/maplebot.service.d/override.conf
done < .env

# Setup systemd service
echo "Setting up systemd service..."
sudo cp maplebot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable maplebot
sudo systemctl start maplebot

echo "Setup complete! Bot should now be running."
echo "Check status with: sudo systemctl status maplebot"