from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()

class User(Base):
    """User model for storing Telegram user information"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, nullable=False)
    username = Column(String(100), nullable=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    language_code = Column(String(10), default="en")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bookings = relationship("Booking", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    
    def __repr__(self):
        return f"<User(telegram_id={self.telegram_id}, username={self.username})>"

class Booking(Base):
    """Booking model for storing travel reservations"""
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    booking_type = Column(String(50), nullable=False)  # 'flight', 'hotel', 'package'
    booking_reference = Column(String(100), unique=True, nullable=False)
    status = Column(String(50), default="confirmed")  # 'confirmed', 'cancelled', 'pending'
    
    # Booking details stored as JSON
    details = Column(JSON, nullable=False)
    
    # Travel dates
    departure_date = Column(DateTime, nullable=True)
    return_date = Column(DateTime, nullable=True)
    
    # Location information
    origin = Column(String(200), nullable=True)
    destination = Column(String(200), nullable=True)
    
    # Price information
    total_price = Column(String(50), nullable=True)
    currency = Column(String(10), default="USD")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="bookings")
    notifications = relationship("Notification", back_populates="booking")
    
    def __repr__(self):
        return f"<Booking(reference={self.booking_reference}, type={self.booking_type})>"

class Notification(Base):
    """Notification model for storing travel updates and alerts"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    booking_id = Column(Integer, ForeignKey("bookings.id"), nullable=True)
    
    notification_type = Column(String(50), nullable=False)  # 'flight_delay', 'gate_change', 'boarding', 'weather'
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Notification status
    is_sent = Column(Boolean, default=False)
    is_read = Column(Boolean, default=False)
    sent_at = Column(DateTime, nullable=True)
    
    # Priority level
    priority = Column(String(20), default="normal")  # 'low', 'normal', 'high', 'urgent'
    
    # Additional data
    notification_data = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    booking = relationship("Booking", back_populates="notifications")
    
    def __repr__(self):
        return f"<Notification(type={self.notification_type}, title={self.title})>"

class Conversation(Base):
    """Conversation model for storing chat history with AI"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String(100), nullable=False)
    
    # Message content
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    
    # Intent and entities extracted
    intent = Column(String(100), nullable=True)
    entities = Column(JSON, nullable=True)
    
    # Confidence scores
    confidence = Column(String(10), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Conversation(session_id={self.session_id}, intent={self.intent})>" 