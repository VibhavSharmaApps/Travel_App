# Travel Bot - AI-Powered Telegram Travel Assistant

A sophisticated Telegram chatbot powered by Hugging Face AI that provides smart booking and reservation management, along with automated travel updates and notifications.

## 🌟 Features

### ✈️ Smart Booking and Reservation Management
- **AI Voice Agents**: Natural language processing for booking flights and hotels
- **Voice Commands**: Effortless reservation using voice commands (coming soon)
- **Availability Checking**: Real-time flight and hotel availability
- **Plan Modifications**: Easy booking modifications and cancellations
- **Multi-language Support**: Support for multiple languages

### 🔔 Automated Travel Updates and Notifications
- **Real-time Updates**: Automatic alerts about itinerary changes
- **Flight Delays**: Instant notifications about flight delays and gate changes
- **Boarding Alerts**: Timely boarding notifications
- **Weather Updates**: Destination weather information
- **Baggage Tracking**: Baggage pickup notifications
- **Customs Information**: Pre-arrival customs alerts

### 🤖 AI-Powered Features
- **Hugging Face Integration**: Advanced natural language processing
- **Intent Classification**: Smart understanding of user requests
- **Entity Extraction**: Automatic extraction of dates, locations, and numbers
- **Context Awareness**: Maintains conversation context
- **Personalized Responses**: Tailored responses based on user preferences

## 🏗️ Architecture

### Backend Components
- **Hugging Face Models**: For NLP and conversation generation
- **SQLAlchemy ORM**: Database management and models
- **PostgreSQL/SQLite**: Data persistence
- **Redis**: Caching and session management
- **Celery**: Background task processing
- **FastAPI**: REST API endpoints (optional)

### Core Services
- **AI Engine**: Hugging Face integration for NLP
- **Booking Service**: Flight and hotel booking management
- **Notification Service**: Automated travel updates
- **Scheduler**: Background task scheduling
- **Telegram Bot**: Main bot interface

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Telegram Bot Token
- Hugging Face API Token
- PostgreSQL (optional, SQLite for development)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd travel-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp config.env.example .env
```

Edit `.env` file with your configuration:
```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_WEBHOOK_URL=https://your-domain.com/webhook

# Hugging Face Configuration
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
HUGGINGFACE_MODEL_NAME=facebook/blenderbot-400M-distill

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/travel_bot_db
REDIS_URL=redis://localhost:6379/0

# External APIs (for demo purposes)
FLIGHT_API_KEY=your_flight_api_key
HOTEL_API_KEY=your_hotel_api_key

# Notification Configuration
NOTIFICATION_INTERVAL=300
MAX_RETRY_ATTEMPTS=3

# Logging
LOG_LEVEL=INFO
```

4. **Initialize the database**
```bash
python -c "from src.database import db_manager; db_manager.create_tables()"
```

5. **Run the bot**
```bash
python main.py
```

## 📱 Usage

### Getting Started
1. Start a conversation with your bot on Telegram
2. Send `/start` to initialize the bot
3. Use natural language to book flights or hotels
4. Receive automated updates about your travel

### Available Commands
- `/start` - Initialize the bot
- `/help` - Show help information
- `/book` - Start booking process
- `/mybookings` - View your bookings
- `/notifications` - Check travel updates
- `/search` - Search flights or hotels

### Example Conversations

**Booking a Flight:**
```
User: "I need a flight from New York to London on March 15th for 2 passengers"
Bot: "I'll help you find flights from New York to London on March 15th for 2 passengers. Here are some options..."
```

**Checking Booking:**
```
User: "What's the status of my booking FL123456?"
Bot: "Your flight booking FL123456 is confirmed. Departure: March 15th, 10:30 AM from JFK to LHR..."
```

**Weather Update:**
```
User: "What's the weather like in London?"
Bot: "Current weather in London: 15°C, partly cloudy. Perfect weather for sightseeing!"
```

## 🔧 Configuration

### Hugging Face Models
The bot uses several Hugging Face models:
- **Conversation Model**: `facebook/blenderbot-400M-distill` for natural responses
- **Sentence Encoder**: `all-MiniLM-L6-v2` for intent classification
- **Custom Models**: Can be configured for specific use cases

### Database Options
- **Development**: SQLite (default)
- **Production**: PostgreSQL with connection pooling
- **Caching**: Redis for session management

### Notification Settings
- **Interval**: Configurable notification check intervals
- **Priority Levels**: Low, Normal, High, Urgent
- **Retry Logic**: Automatic retry for failed notifications

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Test Coverage
- Unit tests for all services
- Integration tests for bot functionality
- Mock API testing for external services

## 📊 Monitoring and Logging

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General application information
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

### Log Files
- `travel_bot.log`: Main application log
- Database logs: Configured in database settings
- Telegram API logs: Available through python-telegram-bot

## 🔒 Security

### Best Practices
- Store sensitive tokens in environment variables
- Use HTTPS for webhook endpoints
- Implement rate limiting for API calls
- Regular security updates for dependencies
- Database connection encryption

### Privacy
- User data is stored securely
- GDPR compliant data handling
- Optional data retention policies
- User consent for notifications

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### Cloud Deployment
- **Heroku**: Easy deployment with Procfile
- **AWS**: EC2 with RDS and ElastiCache
- **Google Cloud**: App Engine with Cloud SQL
- **Azure**: App Service with Azure Database

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the docs folder
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join our community discussions
- **Email**: support@travelbot.com

## 🔮 Roadmap

### Upcoming Features
- **Voice Commands**: Full voice interaction support
- **Multi-language**: Extended language support
- **Advanced AI**: More sophisticated conversation models
- **Mobile App**: Native mobile application
- **Group Bookings**: Support for group travel
- **Loyalty Integration**: Airline and hotel loyalty programs

### Performance Improvements
- **Caching**: Enhanced caching strategies
- **Scalability**: Horizontal scaling support
- **Real-time Updates**: WebSocket integration
- **Analytics**: User behavior analytics

---

**Built with ❤️ using Hugging Face, Python, and Telegram Bot API** 