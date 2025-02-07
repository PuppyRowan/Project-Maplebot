# Standard library imports
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class GuildSettings:
    bark_chance: float = 0.01
    uwu_chance: float = 0.02
    puppy_time_chance: float = 0.05
    gag_active: bool = False
    features_disabled: bool = True
    current_mood: str = 'happy'

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def update_setting(self, setting: str, value: any):
        """Update a setting and return True if value changed"""
        if hasattr(self, setting) and getattr(self, setting) != value:
            setattr(self, setting, value)
            return True
        return False

class GuildSettingsManager:
    def __init__(self, save_path: str = "guild_settings.json"):
        self.settings: Dict[str, GuildSettings] = {}
        self.save_path = save_path
        self.load_settings()

    def get_settings(self, guild_id: int) -> GuildSettings:
        guild_key = str(guild_id)  # Convert to string for JSON compatibility
        if guild_key not in self.settings:
            self.settings[guild_key] = GuildSettings()
            self.save_settings()  # Auto-save when creating new settings
        return self.settings[guild_key]

    def save_settings(self):
        """Save settings to JSON file"""
        data = {
            guild_id: settings.to_dict() 
            for guild_id, settings in self.settings.items()
        }
        try:
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        """Load settings from JSON file"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.settings = {
                        guild_id: GuildSettings.from_dict(settings_data)
                        for guild_id, settings_data in data.items()
                    }
            else:
                # Initialize empty settings if file doesn't exist
                self.settings = {}
                print(f"No settings file found at {self.save_path}. Starting with empty settings.")
        except Exception as e:
            print(f"Error loading settings: {e}")
            self.settings = {}  # Fallback to empty settings on error