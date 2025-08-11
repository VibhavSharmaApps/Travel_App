#!/usr/bin/env python3
"""
Setup script for Travel Bot
Helps users configure and start the bot quickly
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Travel Bot Setup                          ║
║              AI-Powered Telegram Travel Assistant            ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies")
        sys.exit(1)

def create_env_file():
    """Create .env file from template"""
    env_template = "config.env.example"
    env_file = ".env"
    
    if os.path.exists(env_file):
        print(f"⚠️  {env_file} already exists, skipping...")
        return
    
    if not os.path.exists(env_template):
        print(f"❌ Error: {env_template} not found")
        sys.exit(1)
    
    print(f"\n🔧 Creating {env_file} from template...")
    shutil.copy(env_template, env_file)
    print(f"✅ {env_file} created successfully")
    print(f"⚠️  Please edit {env_file} with your configuration")

def setup_database():
    """Initialize database"""
    print("\n🗄️  Setting up database...")
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        
        from src.database import db_manager
        db_manager.create_tables()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        print("⚠️  You can run this manually later")

def check_environment():
    """Check if required environment variables are set"""
    print("\n🔍 Checking environment configuration...")
    
    required_vars = [
        "TELEGRAM_BOT_TOKEN",
        "HUGGINGFACE_API_TOKEN"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file")
        return False
    
    print("✅ All required environment variables are set")
    return True

def run_tests():
    """Run basic tests"""
    print("\n🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
        print("✅ Tests passed successfully")
    except subprocess.CalledProcessError:
        print("❌ Some tests failed")
        print("⚠️  You can still run the bot, but some features might not work correctly")

def print_next_steps():
    """Print next steps for the user"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                        Next Steps                            ║
╚══════════════════════════════════════════════════════════════╝

1. 📝 Edit .env file with your configuration:
   - TELEGRAM_BOT_TOKEN (get from @BotFather)
   - HUGGINGFACE_API_TOKEN (get from huggingface.co)

2. 🚀 Start the bot:
   python main.py

3. 📱 Test the bot on Telegram:
   - Find your bot by username
   - Send /start to begin

4. 📚 Read the documentation:
   - README.md for detailed instructions
   - Check logs/travel_bot.log for debugging

5. 🐳 For production deployment:
   - Use Docker: docker-compose up -d
   - Set up proper SSL certificates
   - Configure webhook URL

Need help? Check the README.md file or create an issue on GitHub.
    """)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create environment file
    create_env_file()
    
    # Setup database
    setup_database()
    
    # Check environment
    env_ok = check_environment()
    
    # Run tests if environment is configured
    if env_ok:
        run_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 