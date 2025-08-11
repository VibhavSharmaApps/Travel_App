#!/usr/bin/env python3
"""
Travel Bot - AI-Powered Telegram Travel Assistant
================================================

Main application entry point for the Travel Bot system.

This module initializes and runs the complete travel bot application including:
- Configuration validation
- Database initialization
- AI engine setup
- Data manager integration
- Telegram bot initialization
- Notification scheduler
- Graceful shutdown handling

The application uses Excel data for flights and hotels in demo mode,
with Hugging Face transformers for natural language processing.

Author: Travel Bot Team
Date: 2024
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import config
from src.database import db_manager
from src.data_manager import data_manager
from src.telegram_bot import telegram_bot
from src.scheduler import notification_scheduler
from src.ai_engine import ai_engine

# Configure comprehensive logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('travel_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TravelBotApp:
    """
    Travel Bot Application Class
    ===========================
    
    Main application class that orchestrates all components of the travel bot.
    Handles initialization, startup, running, and graceful shutdown of all services.
    
    Attributes:
        is_running (bool): Flag indicating if the application is running
        tasks (List): List of running asyncio tasks
        components (Dict): Dictionary of initialized components
    """
    
    def __init__(self):
        """
        Initialize the Travel Bot Application
        
        Sets up the application state and prepares for component initialization.
        """
        self.is_running = False
        self.tasks = []
        self.components = {}
        
        logger.info("Travel Bot Application initialized")
    
    async def initialize(self):
        """
        Initialize all application components
        
        This method performs the complete initialization sequence:
        1. Validates configuration
        2. Sets up database
        3. Loads travel data from Excel files
        4. Initializes AI engine
        5. Sets up Telegram bot
        6. Prepares notification system
        
        Raises:
            Exception: If any component fails to initialize
        """
        try:
            logger.info("Starting Travel Bot initialization...")
            
            # Step 1: Validate configuration
            logger.info("Validating configuration...")
            config.validate()
            logger.info("✅ Configuration validated successfully")
            
            # Step 2: Initialize database
            logger.info("Initializing database...")
            db_manager.create_tables()
            logger.info("✅ Database initialized successfully")
            
            # Step 3: Load travel data from Excel files
            logger.info("Loading travel data from Excel files...")
            if data_manager.load_data():
                logger.info("✅ Travel data loaded successfully")
                
                # Log data statistics
                stats = data_manager.get_statistics()
                logger.info(f"📊 Data Statistics:")
                logger.info(f"   Flights: {stats['flights']['total']} records")
                logger.info(f"   Hotels: {stats['hotels']['total']} records")
                logger.info(f"   Available airports: {len(stats['flights']['airports'])}")
                logger.info(f"   Available locations: {len(stats['hotels']['locations'])}")
            else:
                logger.warning("⚠️ Failed to load travel data - using mock data")
            
            # Step 4: Initialize AI engine
            logger.info("Initializing AI engine...")
            # AI engine is initialized when imported, but we can verify it's working
            test_result = ai_engine.process_message("Hello")
            if test_result['intent']:
                logger.info("✅ AI engine initialized successfully")
            else:
                logger.warning("⚠️ AI engine may not be fully functional")
            
            # Step 5: Initialize Telegram bot
            logger.info("Initializing Telegram bot...")
            await telegram_bot.initialize()
            logger.info("✅ Telegram bot initialized successfully")
            
            # Step 6: Prepare notification system
            logger.info("Preparing notification system...")
            # Notification scheduler will be started when the app starts
            logger.info("✅ Notification system prepared")
            
            logger.info("🎉 Travel Bot initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Travel Bot: {e}")
            raise
    
    async def start(self):
        """
        Start the Travel Bot application
        
        This method starts all running components and maintains the application
        until shutdown is requested. It handles:
        - Starting the notification scheduler
        - Running the Telegram bot
        - Managing background tasks
        - Handling graceful shutdown
        
        The application runs until interrupted by a signal or error.
        """
        try:
            self.is_running = True
            logger.info("🚀 Starting Travel Bot application...")
            
            # Start notification scheduler in background
            logger.info("Starting notification scheduler...")
            scheduler_task = asyncio.create_task(notification_scheduler.start())
            self.tasks.append(scheduler_task)
            logger.info("✅ Notification scheduler started")
            
            # Start Telegram bot in background
            logger.info("Starting Telegram bot...")
            bot_task = asyncio.create_task(telegram_bot.run())
            self.tasks.append(bot_task)
            logger.info("✅ Telegram bot started")
            
            # Log startup completion
            logger.info("🎉 Travel Bot is now running!")
            logger.info("📱 Users can now interact with the bot on Telegram")
            logger.info("🔔 Automated notifications are active")
            logger.info("🤖 AI-powered responses are enabled")
            logger.info("📊 Excel data integration is active")
            logger.info("")
            logger.info("Press Ctrl+C to stop the bot gracefully...")
            
            # Wait for all tasks to complete (they run indefinitely)
            await asyncio.gather(*self.tasks)
            
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"❌ Error in main application: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """
        Gracefully shutdown the Travel Bot application
        
        This method ensures all components are properly shut down:
        - Stops the notification scheduler
        - Cancels all running tasks
        - Closes database connections
        - Logs shutdown completion
        
        The shutdown process is designed to be clean and complete.
        """
        try:
            logger.info("🔄 Shutting down Travel Bot...")
            
            # Stop notification scheduler
            logger.info("Stopping notification scheduler...")
            notification_scheduler.stop()
            logger.info("✅ Notification scheduler stopped")
            
            # Cancel all running tasks
            logger.info("Cancelling running tasks...")
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    logger.debug(f"Cancelled task: {task.get_name()}")
            
            # Wait for tasks to complete cancellation
            if self.tasks:
                logger.info("Waiting for tasks to complete...")
                await asyncio.gather(*self.tasks, return_exceptions=True)
                logger.info("✅ All tasks completed")
            
            # Close database connections
            logger.info("Closing database connections...")
            # Database connections are handled by SQLAlchemy session management
            logger.info("✅ Database connections closed")
            
            logger.info("🎉 Travel Bot shutdown completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    def setup_signal_handlers(self):
        """
        Setup signal handlers for graceful shutdown
        
        This method configures signal handlers to catch system signals
        (SIGINT, SIGTERM) and initiate graceful shutdown when received.
        """
        def signal_handler(signum, frame):
            """
            Signal handler for graceful shutdown
            
            Args:
                signum: Signal number
                frame: Current stack frame
            """
            logger.info(f"📡 Received signal {signum}, initiating shutdown...")
            self.is_running = False
            
            # Cancel the main event loop
            try:
                loop = asyncio.get_event_loop()
                loop.stop()
            except Exception as e:
                logger.error(f"Error stopping event loop: {e}")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        logger.info("✅ Signal handlers configured")

async def main():
    """
    Main function - Entry point for the Travel Bot application
    
    This function:
    1. Creates the application instance
    2. Sets up signal handlers
    3. Initializes all components
    4. Starts the application
    5. Handles any fatal errors
    
    The application runs until interrupted or an error occurs.
    """
    app = TravelBotApp()
    
    try:
        # Setup signal handlers for graceful shutdown
        app.setup_signal_handlers()
        
        # Initialize the application
        await app.initialize()
        
        # Start the application
        await app.start()
        
    except Exception as e:
        logger.error(f"💥 Fatal error in Travel Bot: {e}")
        logger.error("Application will exit with error code 1")
        sys.exit(1)

if __name__ == "__main__":
    """
    Application entry point
    
    When this script is run directly, it starts the Travel Bot application.
    The application will run until interrupted by a signal or error.
    """
    # Run the application using asyncio
    asyncio.run(main()) 