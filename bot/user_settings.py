from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, Optional

@dataclass
class UserSettings:
    user_id: int
    current_mood: str = 'happy'
    bark_chance: float = 0.01
    uwu_chance: float = 0.02
    puppy_time_chance: float = 0.05
    gag_active: bool = False
    features_disabled: bool = False
    is_puppy: bool = False  # Flag to mark if user is a "puppy"

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class UserSettingsManager:
    def __init__(self, save_path: str = "bot/users.json"):
        self.settings: Dict[str, UserSettings] = {}
        self.save_path = save_path
        self.load_settings()

    def get_settings(self, user_id: int) -> Optional[UserSettings]:
        """Get settings for user, returns None if user isn't a puppy"""
        user_key = str(user_id)
        return self.settings.get(user_key)

    def add_puppy(self, user_id: int) -> UserSettings:
        """Add a new puppy user"""
        user_key = str(user_id)
        if user_key not in self.settings:
            self.settings[user_key] = UserSettings(user_id=user_id, is_puppy=True)
            self.save_settings()
        return self.settings[user_key]

    def remove_puppy(self, user_id: int) -> bool:
        """Remove a puppy user, returns True if user was removed"""
        user_key = str(user_id)
        if user_key in self.settings:
            del self.settings[user_key]
            self.save_settings()
            return True
        return False

    def save_settings(self):
        """Save settings to JSON file"""
        data = {
            user_id: settings.to_dict() 
            for user_id, settings in self.settings.items()
        }
        try:
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving user settings: {e}")

    def load_settings(self):
        """Load settings from JSON file"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.settings = {
                        user_id: UserSettings.from_dict(settings_data)
                        for user_id, settings_data in data.items()
                    }
            else:
                self.settings = {}
        except Exception as e:
            print(f"Error loading user settings: {e}")
            self.settings = {}