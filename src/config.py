import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Travel Bot"""
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_WEBHOOK_URL: Optional[str] = os.getenv("TELEGRAM_WEBHOOK_URL")
    
    # Hugging Face Configuration
    HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    HUGGINGFACE_MODEL_NAME: str = os.getenv("HUGGINGFACE_MODEL_NAME", "facebook/blenderbot-400M-distill")
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./travel_bot.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # External APIs
    FLIGHT_API_KEY: Optional[str] = os.getenv("FLIGHT_API_KEY")
    HOTEL_API_KEY: Optional[str] = os.getenv("HOTEL_API_KEY")
    
    # Notification Configuration
    NOTIFICATION_INTERVAL: int = int(os.getenv("NOTIFICATION_INTERVAL", "300"))
    MAX_RETRY_ATTEMPTS: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        required_fields = [
            "TELEGRAM_BOT_TOKEN",
            "HUGGINGFACE_API_TOKEN"
        ]
        
        for field in required_fields:
            if not getattr(cls, field):
                raise ValueError(f"Missing required configuration: {field}")
        
        return True

# Global config instance
config = Config() 