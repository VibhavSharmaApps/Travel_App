from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from .config import config
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for handling connections and sessions"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Setup database engine based on configuration"""
        try:
            if config.DATABASE_URL.startswith("sqlite"):
                # SQLite configuration for development
                self.engine = create_engine(
                    config.DATABASE_URL,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                    echo=False
                )
            else:
                # PostgreSQL configuration for production
                self.engine = create_engine(
                    config.DATABASE_URL,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    echo=False
                )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database engine setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup database engine: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_direct(self) -> Session:
        """Get database session directly (for use in async contexts)"""
        return self.SessionLocal()

# Global database manager instance
db_manager = DatabaseManager() 