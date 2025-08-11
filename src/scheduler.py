import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import List

from .notification_service import notification_service
from .database import db_manager
from .models import Notification

logger = logging.getLogger(__name__)

class NotificationScheduler:
    """Scheduler for automated travel notifications"""
    
    def __init__(self):
        self.is_running = False
        self.tasks = []
    
    async def start(self):
        """Start the notification scheduler"""
        try:
            self.is_running = True
            
            # Schedule tasks
            schedule.every(5).minutes.do(self._generate_notifications)
            schedule.every(1).minutes.do(self._send_pending_notifications)
            schedule.every().hour.do(self._cleanup_old_notifications)
            
            logger.info("Notification scheduler started")
            
            # Run the scheduler
            while self.is_running:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Error in notification scheduler: {e}")
            self.is_running = False
    
    def stop(self):
        """Stop the notification scheduler"""
        self.is_running = False
        logger.info("Notification scheduler stopped")
    
    async def _generate_notifications(self):
        """Generate automated notifications for upcoming travel"""
        try:
            logger.info("Generating automated notifications...")
            await notification_service.generate_automated_notifications()
            
        except Exception as e:
            logger.error(f"Error generating notifications: {e}")
    
    async def _send_pending_notifications(self):
        """Send pending notifications to users"""
        try:
            with db_manager.get_session() as session:
                # Get unsent notifications
                pending_notifications = session.query(Notification).filter(
                    Notification.is_sent == False
                ).limit(10).all()  # Process 10 at a time
                
                for notification in pending_notifications:
                    try:
                        success = await notification_service.send_notification(notification.id)
                        if success:
                            logger.info(f"Sent notification {notification.id}")
                        else:
                            logger.warning(f"Failed to send notification {notification.id}")
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error sending notification {notification.id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing pending notifications: {e}")
    
    async def _cleanup_old_notifications(self):
        """Clean up old notifications (older than 30 days)"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            with db_manager.get_session() as session:
                # Delete old notifications
                deleted_count = session.query(Notification).filter(
                    Notification.created_at < cutoff_date,
                    Notification.is_read == True
                ).delete()
                
                session.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old notifications")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old notifications: {e}")
    
    async def send_immediate_notification(self, user_id: int, notification_type: str, 
                                        title: str, message: str, priority: str = "normal"):
        """Send an immediate notification"""
        try:
            notification = await notification_service.create_notification(
                user_id=user_id,
                notification_type=notification_type,
                title=title,
                message=message,
                priority=priority
            )
            
            # Send immediately
            success = await notification_service.send_notification(notification.id)
            
            if success:
                logger.info(f"Sent immediate notification {notification.id} to user {user_id}")
            else:
                logger.warning(f"Failed to send immediate notification {notification.id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending immediate notification: {e}")
            return False

# Global scheduler instance
notification_scheduler = NotificationScheduler() 