import logging
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from .models import Notification, User, Booking
from .database import db_manager

logger = logging.getLogger(__name__)

class NotificationService:
    """Service for handling automated travel notifications"""
    
    def __init__(self):
        self.notification_types = {
            'flight_delay': self._create_flight_delay_notification,
            'gate_change': self._create_gate_change_notification,
            'boarding': self._create_boarding_notification,
            'weather': self._create_weather_notification,
            'baggage': self._create_baggage_notification,
            'customs': self._create_customs_notification
        }
    
    async def create_notification(self, user_id: int, notification_type: str, 
                                title: str, message: str, booking_id: Optional[int] = None,
                                priority: str = "normal", metadata: Optional[Dict] = None) -> Notification:
        """Create a new notification"""
        try:
            with db_manager.get_session() as session:
                notification = Notification(
                    user_id=user_id,
                    booking_id=booking_id,
                    notification_type=notification_type,
                    title=title,
                    message=message,
                    priority=priority,
                    metadata=metadata or {}
                )
                
                session.add(notification)
                session.commit()
                
                logger.info(f"Created notification {notification_type} for user {user_id}")
                return notification
                
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
            raise
    
    async def send_notification(self, notification_id: int) -> bool:
        """Send notification to user via Telegram"""
        try:
            with db_manager.get_session() as session:
                notification = session.query(Notification).filter(
                    Notification.id == notification_id
                ).first()
                
                if not notification:
                    return False
                
                user = session.query(User).filter(User.id == notification.user_id).first()
                if not user:
                    return False
                
                # Format message
                message = self._format_notification_message(notification)
                
                # Send via Telegram
                from .telegram_bot import telegram_bot
                success = await telegram_bot.send_message(
                    chat_id=user.telegram_id,
                    text=message,
                    parse_mode='HTML'
                )
                
                if success:
                    notification.is_sent = True
                    notification.sent_at = datetime.utcnow()
                    session.commit()
                    logger.info(f"Sent notification {notification_id} to user {user.telegram_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
    
    def get_user_notifications(self, user_id: int, unread_only: bool = False) -> List[Notification]:
        """Get notifications for a user"""
        try:
            with db_manager.get_session() as session:
                query = session.query(Notification).filter(Notification.user_id == user_id)
                
                if unread_only:
                    query = query.filter(Notification.is_read == False)
                
                notifications = query.order_by(Notification.created_at.desc()).all()
                return notifications
                
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    def mark_notification_read(self, notification_id: int) -> bool:
        """Mark notification as read"""
        try:
            with db_manager.get_session() as session:
                notification = session.query(Notification).filter(
                    Notification.id == notification_id
                ).first()
                
                if not notification:
                    return False
                
                notification.is_read = True
                session.commit()
                
                logger.info(f"Marked notification {notification_id} as read")
                return True
                
        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")
            return False
    
    async def generate_automated_notifications(self):
        """Generate automated notifications for upcoming travel"""
        try:
            with db_manager.get_session() as session:
                # Get upcoming bookings (within next 24 hours)
                tomorrow = datetime.utcnow() + timedelta(days=1)
                upcoming_bookings = session.query(Booking).filter(
                    Booking.departure_date <= tomorrow,
                    Booking.status == "confirmed"
                ).all()
                
                for booking in upcoming_bookings:
                    await self._generate_booking_notifications(booking)
                    
        except Exception as e:
            logger.error(f"Error generating automated notifications: {e}")
    
    async def _generate_booking_notifications(self, booking: Booking):
        """Generate notifications for a specific booking"""
        try:
            # Flight delay simulation (10% chance)
            if booking.booking_type == "flight" and random.random() < 0.1:
                await self._create_flight_delay_notification(booking)
            
            # Gate change simulation (5% chance)
            if booking.booking_type == "flight" and random.random() < 0.05:
                await self._create_gate_change_notification(booking)
            
            # Boarding notification (for flights departing within 2 hours)
            if booking.booking_type == "flight":
                time_until_departure = booking.departure_date - datetime.utcnow()
                if timedelta(hours=1) <= time_until_departure <= timedelta(hours=2):
                    await self._create_boarding_notification(booking)
            
            # Weather notification (for destination)
            if random.random() < 0.3:  # 30% chance
                await self._create_weather_notification(booking)
                
        except Exception as e:
            logger.error(f"Error generating booking notifications: {e}")
    
    async def _create_flight_delay_notification(self, booking: Booking):
        """Create flight delay notification"""
        delay_minutes = random.randint(15, 120)
        title = "Flight Delay Alert"
        message = f"Your flight {booking.details.get('flight_number', 'N/A')} has been delayed by {delay_minutes} minutes."
        
        await self.create_notification(
            user_id=booking.user_id,
            notification_type="flight_delay",
            title=title,
            message=message,
            booking_id=booking.id,
            priority="high",
            metadata={"delay_minutes": delay_minutes}
        )
    
    async def _create_gate_change_notification(self, booking: Booking):
        """Create gate change notification"""
        old_gate = booking.details.get('gate', 'A1')
        new_gate = random.choice(['B2', 'C3', 'D4', 'E5'])
        title = "Gate Change Alert"
        message = f"Your flight {booking.details.get('flight_number', 'N/A')} gate has changed from {old_gate} to {new_gate}."
        
        await self.create_notification(
            user_id=booking.user_id,
            notification_type="gate_change",
            title=title,
            message=message,
            booking_id=booking.id,
            priority="high",
            metadata={"old_gate": old_gate, "new_gate": new_gate}
        )
    
    async def _create_boarding_notification(self, booking: Booking):
        """Create boarding notification"""
        title = "Boarding Alert"
        message = f"Boarding for flight {booking.details.get('flight_number', 'N/A')} will begin in 30 minutes."
        
        await self.create_notification(
            user_id=booking.user_id,
            notification_type="boarding",
            title=title,
            message=message,
            booking_id=booking.id,
            priority="normal"
        )
    
    async def _create_weather_notification(self, booking: Booking):
        """Create weather notification"""
        weather_conditions = ["sunny", "rainy", "cloudy", "stormy"]
        condition = random.choice(weather_conditions)
        title = "Weather Update"
        message = f"Weather at {booking.destination}: {condition} with temperature around {random.randint(15, 30)}°C."
        
        await self.create_notification(
            user_id=booking.user_id,
            notification_type="weather",
            title=title,
            message=message,
            booking_id=booking.id,
            priority="low"
        )
    
    async def _create_baggage_notification(self, booking: Booking):
        """Create baggage notification"""
        title = "Baggage Information"
        message = f"Your baggage for flight {booking.details.get('flight_number', 'N/A')} is ready for pickup at carousel 3."
        
        await self.create_notification(
            user_id=booking.user_id,
            notification_type="baggage",
            title=title,
            message=message,
            booking_id=booking.id,
            priority="normal"
        )
    
    async def _create_customs_notification(self, booking: Booking):
        """Create customs notification"""
        title = "Customs Information"
        message = f"Please have your passport and customs declaration form ready for flight {booking.details.get('flight_number', 'N/A')}."
        
        await self.create_notification(
            user_id=booking.user_id,
            notification_type="customs",
            title=title,
            message=message,
            booking_id=booking.id,
            priority="normal"
        )
    
    def _format_notification_message(self, notification: Notification) -> str:
        """Format notification for Telegram message"""
        priority_emoji = {
            "low": "🔵",
            "normal": "🟡", 
            "high": "🟠",
            "urgent": "🔴"
        }
        
        emoji = priority_emoji.get(notification.priority, "🟡")
        
        message = f"{emoji} <b>{notification.title}</b>\n\n"
        message += f"{notification.message}\n\n"
        message += f"📅 {notification.created_at.strftime('%Y-%m-%d %H:%M')}"
        
        if notification.booking_id:
            message += f"\n🔗 Booking ID: {notification.booking_id}"
        
        return message

# Global notification service instance
notification_service = NotificationService() 