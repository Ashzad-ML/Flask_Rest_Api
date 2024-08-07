from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Text,
    Integer,
    Boolean,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime


DATABASE_URL = "sqlite:///api_logs_new.db"


engine = create_engine(DATABASE_URL, echo=True)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


class APILog(Base):
    __tablename__ = "api_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    endpoint = Column(String)
    api_key_id = Column(Integer, ForeignKey("users.id"))
    request_method = Column(String)
    status_code = Column(Integer)
    prediction_result = Column(Text)
    error_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, nullable=False)
    user = relationship("User", back_populates="api_logs")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    # username = Column(String, unique=True, nullable=False)
    api_key = Column(String, unique=True, nullable=False)
    api_logs = relationship("APILog", back_populates="user")


Base.metadata.create_all(engine)
