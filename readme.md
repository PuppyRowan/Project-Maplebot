# Maplebot - Discord Puppy Bot

A fun Discord bot that transforms a user's messages into playful puppy-like interactions. The bot includes features like mood changes, scheduled activities, and customizable response patterns.

## Quick Setup

1. **Clone the repository**
```bash
git clone [your-repo-url]
cd maplebot
```

2. **Create Discord Bot**
- Go to [Discord Developer Portal](https://discord.com/developers/applications)
- Create New Application
- Go to Bot section and create a bot
- Copy the bot token
- Enable Message Content Intent under Privileged Gateway Intents

3. **Run Setup Script**
```bash
chmod +x setup.sh
./setup.sh
```
The setup script will:
- Install required system dependencies
- Create Python virtual environment
- Install Python packages
- Create .env file with your bot token
- Set up systemd service

4. **Configuration**
Edit `bot/config.py` to set:
- TARGET_USER_ID = Your target user's Discord ID
- Other settings as needed

5. **Start the Bot**
```bash
sudo systemctl start maplebot
```

## Commands
- `!set_bark <chance>` - Set probability of bark responses (0-1)
- `!set_uwu <chance>` - Set probability of uwu speech (0-1)
- `!puppytime` - Check current puppy schedule activity
- `!gag` - Toggle muzzle/gag mode
- `!off` - Toggle extra features on/off

## Requirements
- Python 3.8+
- discord.py 2.0.0+
- python-dotenv

## System Requirements
- Linux-based system with systemd
- Python 3.8 or higher
- Internet connection

## Support
For issues or questions, please open an issue in the repository.