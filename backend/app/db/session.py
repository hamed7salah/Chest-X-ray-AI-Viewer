import os
import time
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./dev.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

if not DATABASE_URL.startswith("sqlite"):
    retries = 0
    while retries < 10:
        try:
            with engine.connect() as conn:
                break
        except OperationalError:
            retries += 1
            time.sleep(2)
    else:
        raise RuntimeError("Unable to connect to the database after multiple retries.")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
