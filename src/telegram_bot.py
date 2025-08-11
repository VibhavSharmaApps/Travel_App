import logging
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

from .config import config
from .ai_engine import ai_engine
from .booking_service import booking_service
from .notification_service import notification_service
from .models import User, Conversation
from .database import db_manager

logger = logging.getLogger(__name__)

class TravelBot:
    """Main Telegram bot class for travel assistance"""
    
    def __init__(self):
        self.application = None
        self.user_sessions = {}  # Store user conversation context
    
    async def initialize(self):
        """Initialize the bot"""
        try:
            self.application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("book", self.book_command))
            self.application.add_handler(CommandHandler("mybookings", self.my_bookings_command))
            self.application.add_handler(CommandHandler("notifications", self.notifications_command))
            self.application.add_handler(CommandHandler("search", self.search_command))
            
            # Add message handler for general conversation
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            # Add callback query handler for inline keyboards
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            
            logger.info("Telegram bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            raise
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        try:
            user = update.effective_user
            
            # Register user in database
            await self._register_user(user)
            
            welcome_message = f"""
🎉 Welcome to TravelBot, {user.first_name}!

I'm your AI travel assistant powered by Hugging Face. I can help you with:

✈️ **Flight Bookings** - Search and book flights
🏨 **Hotel Reservations** - Find and reserve hotels
📋 **Booking Management** - Check, modify, or cancel bookings
🔔 **Travel Updates** - Real-time notifications about your trips
🌤️ **Weather Information** - Get weather updates for your destination

Try these commands:
• /book - Start booking process
• /mybookings - View your bookings
• /notifications - Check travel updates
• /search - Search flights or hotels
• /help - Get help

Or simply chat with me naturally! 🗣️
            """
            
            keyboard = [
                [InlineKeyboardButton("✈️ Book Flight", callback_data="book_flight")],
                [InlineKeyboardButton("🏨 Book Hotel", callback_data="book_hotel")],
                [InlineKeyboardButton("📋 My Bookings", callback_data="my_bookings")],
                [InlineKeyboardButton("🔔 Notifications", callback_data="notifications")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(welcome_message, reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await update.message.reply_text("Sorry, I encountered an error. Please try again.")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🤖 **TravelBot Help**

**Commands:**
• /start - Start the bot
• /book - Book flights or hotels
• /mybookings - View your bookings
• /notifications - Check travel updates
• /search - Search flights or hotels
• /help - Show this help

**Features:**
• AI-powered conversation using Hugging Face
• Voice command support (coming soon)
• Real-time travel notifications
• Automated booking management
• Weather updates for destinations

**How to use:**
1. Start with /start to register
2. Use /book to begin booking process
3. Chat naturally with me for assistance
4. Get automated updates about your trips

Need more help? Just chat with me! 😊
        """
        await update.message.reply_text(help_text)
    
    async def book_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /book command"""
        keyboard = [
            [InlineKeyboardButton("✈️ Book Flight", callback_data="book_flight")],
            [InlineKeyboardButton("🏨 Book Hotel", callback_data="book_hotel")],
            [InlineKeyboardButton("📦 Book Package", callback_data="book_package")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "What would you like to book?",
            reply_markup=reply_markup
        )
    
    async def my_bookings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mybookings command"""
        try:
            user = update.effective_user
            user_record = await self._get_user(user.id)
            
            if not user_record:
                await update.message.reply_text("Please start the bot first with /start")
                return
            
            bookings = booking_service.get_user_bookings(user_record.id)
            
            if not bookings:
                await update.message.reply_text("You don't have any bookings yet. Use /book to make your first booking!")
                return
            
            message = "📋 **Your Bookings:**\n\n"
            for booking in bookings[:5]:  # Show last 5 bookings
                status_emoji = "✅" if booking.status == "confirmed" else "❌" if booking.status == "cancelled" else "⏳"
                message += f"{status_emoji} **{booking.booking_type.title()}** - {booking.booking_reference}\n"
                message += f"📍 {booking.origin} → {booking.destination}\n"
                message += f"📅 {booking.departure_date.strftime('%Y-%m-%d') if booking.departure_date else 'N/A'}\n"
                message += f"💰 {booking.total_price} {booking.currency}\n\n"
            
            if len(bookings) > 5:
                message += f"... and {len(bookings) - 5} more bookings"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in my_bookings command: {e}")
            await update.message.reply_text("Sorry, I couldn't retrieve your bookings. Please try again.")
    
    async def notifications_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /notifications command"""
        try:
            user = update.effective_user
            user_record = await self._get_user(user.id)
            
            if not user_record:
                await update.message.reply_text("Please start the bot first with /start")
                return
            
            notifications = notification_service.get_user_notifications(user_record.id, unread_only=True)
            
            if not notifications:
                await update.message.reply_text("🔔 You have no new notifications!")
                return
            
            message = "🔔 **Recent Notifications:**\n\n"
            for notification in notifications[:3]:  # Show last 3 notifications
                priority_emoji = {"low": "🔵", "normal": "🟡", "high": "🟠", "urgent": "🔴"}
                emoji = priority_emoji.get(notification.priority, "🟡")
                message += f"{emoji} **{notification.title}**\n"
                message += f"{notification.message[:100]}...\n"
                message += f"📅 {notification.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
            
            if len(notifications) > 3:
                message += f"... and {len(notifications) - 3} more notifications"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in notifications command: {e}")
            await update.message.reply_text("Sorry, I couldn't retrieve your notifications. Please try again.")
    
    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /search command"""
        keyboard = [
            [InlineKeyboardButton("✈️ Search Flights", callback_data="search_flights")],
            [InlineKeyboardButton("🏨 Search Hotels", callback_data="search_hotels")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "What would you like to search for?",
            reply_markup=reply_markup
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle general text messages"""
        try:
            user = update.effective_user
            message_text = update.message.text
            
            # Register user if not already registered
            user_record = await self._register_user(user)
            
            # Get user context
            user_context = self.user_sessions.get(user.id, {})
            
            # Process message with AI
            ai_response = ai_engine.process_message(message_text, user_context)
            
            # Store conversation
            await self._store_conversation(user_record.id, message_text, ai_response['response'], ai_response)
            
            # Update user context
            self.user_sessions[user.id] = ai_response['context']
            
            # Handle specific intents
            if ai_response['intent'] == 'book_flight':
                await self._handle_flight_booking(update, ai_response)
            elif ai_response['intent'] == 'book_hotel':
                await self._handle_hotel_booking(update, ai_response)
            elif ai_response['intent'] == 'check_booking':
                await self._handle_check_booking(update, ai_response)
            else:
                # Send AI response
                await update.message.reply_text(ai_response['response'])
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await update.message.reply_text("I'm sorry, I encountered an error processing your message. Please try again.")
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == "book_flight":
                await self._show_flight_booking_form(query)
            elif query.data == "book_hotel":
                await self._show_hotel_booking_form(query)
            elif query.data == "my_bookings":
                await self._show_user_bookings(query)
            elif query.data == "notifications":
                await self._show_user_notifications(query)
            elif query.data == "search_flights":
                await self._show_flight_search_form(query)
            elif query.data == "search_hotels":
                await self._show_hotel_search_form(query)
            else:
                await query.edit_message_text("Invalid option selected.")
                
        except Exception as e:
            logger.error(f"Error handling callback: {e}")
            await query.edit_message_text("Sorry, I encountered an error. Please try again.")
    
    async def _register_user(self, user) -> User:
        """Register or get user from database"""
        try:
            with db_manager.get_session() as session:
                existing_user = session.query(User).filter(User.telegram_id == user.id).first()
                
                if existing_user:
                    return existing_user
                
                new_user = User(
                    telegram_id=user.id,
                    username=user.username,
                    first_name=user.first_name,
                    last_name=user.last_name,
                    language_code=user.language_code or "en"
                )
                
                session.add(new_user)
                session.commit()
                
                logger.info(f"Registered new user: {user.id}")
                return new_user
                
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            raise
    
    async def _get_user(self, telegram_id: int) -> Optional[User]:
        """Get user from database"""
        try:
            with db_manager.get_session() as session:
                return session.query(User).filter(User.telegram_id == telegram_id).first()
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    async def _store_conversation(self, user_id: int, user_message: str, bot_response: str, ai_data: Dict):
        """Store conversation in database"""
        try:
            with db_manager.get_session() as session:
                conversation = Conversation(
                    user_id=user_id,
                    session_id=f"session_{user_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                    user_message=user_message,
                    bot_response=bot_response,
                    intent=ai_data.get('intent'),
                    entities=ai_data.get('entities'),
                    confidence=str(ai_data.get('confidence', 0.0))
                )
                
                session.add(conversation)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
    
    async def _handle_flight_booking(self, update: Update, ai_response: Dict):
        """Handle flight booking intent"""
        message = "✈️ I can help you book a flight! Please provide:\n\n"
        message += "• Origin city/airport\n"
        message += "• Destination city/airport\n"
        message += "• Travel date\n"
        message += "• Number of passengers\n\n"
        message += "Example: 'I need a flight from New York to London on March 15th for 2 passengers'"
        
        await update.message.reply_text(message)
    
    async def _handle_hotel_booking(self, update: Update, ai_response: Dict):
        """Handle hotel booking intent"""
        message = "🏨 I can help you book a hotel! Please provide:\n\n"
        message += "• Destination city\n"
        message += "• Check-in date\n"
        message += "• Check-out date\n"
        message += "• Number of guests\n\n"
        message += "Example: 'I need a hotel in Paris from March 15th to March 20th for 2 guests'"
        
        await update.message.reply_text(message)
    
    async def _handle_check_booking(self, update: Update, ai_response: Dict):
        """Handle booking check intent"""
        entities = ai_response.get('entities', {})
        booking_ref = entities.get('booking_reference')
        
        if booking_ref:
            booking = booking_service.get_booking_by_reference(booking_ref)
            if booking:
                message = f"📋 **Booking Found:**\n\n"
                message += f"Reference: {booking.booking_reference}\n"
                message += f"Type: {booking.booking_type.title()}\n"
                message += f"Status: {booking.status.title()}\n"
                message += f"Origin: {booking.origin}\n"
                message += f"Destination: {booking.destination}\n"
                message += f"Price: {booking.total_price} {booking.currency}"
                
                await update.message.reply_text(message, parse_mode='Markdown')
            else:
                await update.message.reply_text(f"❌ No booking found with reference {booking_ref}")
        else:
            await update.message.reply_text("Please provide your booking reference number to check your booking.")
    
    async def _show_flight_booking_form(self, query):
        """Show flight booking form"""
        message = "✈️ **Flight Booking**\n\n"
        message += "Please provide your flight details in this format:\n\n"
        message += "From: [Origin]\n"
        message += "To: [Destination]\n"
        message += "Date: [YYYY-MM-DD]\n"
        message += "Passengers: [Number]\n\n"
        message += "Example: 'I need a flight from New York to London on 2024-03-15 for 2 passengers'"
        
        await query.edit_message_text(message, parse_mode='Markdown')
    
    async def _show_hotel_booking_form(self, query):
        """Show hotel booking form"""
        message = "🏨 **Hotel Booking**\n\n"
        message += "Please provide your hotel details in this format:\n\n"
        message += "Location: [City]\n"
        message += "Check-in: [YYYY-MM-DD]\n"
        message += "Check-out: [YYYY-MM-DD]\n"
        message += "Guests: [Number]\n\n"
        message += "Example: 'I need a hotel in Paris from 2024-03-15 to 2024-03-20 for 2 guests'"
        
        await query.edit_message_text(message, parse_mode='Markdown')
    
    async def _show_user_bookings(self, query):
        """Show user bookings"""
        try:
            user_record = await self._get_user(query.from_user.id)
            if not user_record:
                await query.edit_message_text("Please start the bot first with /start")
                return
            
            bookings = booking_service.get_user_bookings(user_record.id)
            
            if not bookings:
                await query.edit_message_text("You don't have any bookings yet. Use /book to make your first booking!")
                return
            
            message = "📋 **Your Bookings:**\n\n"
            for booking in bookings[:3]:
                status_emoji = "✅" if booking.status == "confirmed" else "❌" if booking.status == "cancelled" else "⏳"
                message += f"{status_emoji} **{booking.booking_type.title()}** - {booking.booking_reference}\n"
                message += f"📍 {booking.origin} → {booking.destination}\n"
                message += f"📅 {booking.departure_date.strftime('%Y-%m-%d') if booking.departure_date else 'N/A'}\n\n"
            
            await query.edit_message_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing user bookings: {e}")
            await query.edit_message_text("Sorry, I couldn't retrieve your bookings.")
    
    async def _show_user_notifications(self, query):
        """Show user notifications"""
        try:
            user_record = await self._get_user(query.from_user.id)
            if not user_record:
                await query.edit_message_text("Please start the bot first with /start")
                return
            
            notifications = notification_service.get_user_notifications(user_record.id, unread_only=True)
            
            if not notifications:
                await query.edit_message_text("🔔 You have no new notifications!")
                return
            
            message = "🔔 **Recent Notifications:**\n\n"
            for notification in notifications[:3]:
                priority_emoji = {"low": "🔵", "normal": "🟡", "high": "🟠", "urgent": "🔴"}
                emoji = priority_emoji.get(notification.priority, "🟡")
                message += f"{emoji} **{notification.title}**\n"
                message += f"{notification.message[:80]}...\n\n"
            
            await query.edit_message_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing user notifications: {e}")
            await query.edit_message_text("Sorry, I couldn't retrieve your notifications.")
    
    async def _show_flight_search_form(self, query):
        """Show flight search form"""
        message = "✈️ **Flight Search**\n\n"
        message += "Please provide your search criteria:\n\n"
        message += "From: [Origin]\n"
        message += "To: [Destination]\n"
        message += "Date: [YYYY-MM-DD]\n"
        message += "Passengers: [Number]\n\n"
        message += "Example: 'Search flights from New York to London on 2024-03-15 for 2 passengers'"
        
        await query.edit_message_text(message, parse_mode='Markdown')
    
    async def _show_hotel_search_form(self, query):
        """Show hotel search form"""
        message = "🏨 **Hotel Search**\n\n"
        message += "Please provide your search criteria:\n\n"
        message += "Location: [City]\n"
        message += "Check-in: [YYYY-MM-DD]\n"
        message += "Check-out: [YYYY-MM-DD]\n"
        message += "Guests: [Number]\n\n"
        message += "Example: 'Search hotels in Paris from 2024-03-15 to 2024-03-20 for 2 guests'"
        
        await query.edit_message_text(message, parse_mode='Markdown')
    
    async def send_message(self, chat_id: int, text: str, parse_mode: str = None) -> bool:
        """Send message to user"""
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode
            )
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def run(self):
        """Run the bot"""
        try:
            await self.application.run_polling()
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise

# Global bot instance
telegram_bot = TravelBot() 