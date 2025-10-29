import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from src.config import settings

# ---------------------------------------------------
# 🧾 Configure logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# ⚙️ Create database engine with error handling
# ---------------------------------------------------
try:
    engine = create_engine(settings.DATABASE_URL, future=True, pool_pre_ping=True)
    # Test connection
    with engine.connect() as conn:
        conn.execute("SELECT 1")
    logger.info("✅ Database connection established successfully.")
except OperationalError as e:
    logger.error("❌ Failed to connect to the database.")
    logger.error(str(e))
    raise SystemExit(1)  # Stop the app if DB connection fails

# ---------------------------------------------------
# 🧱 Session factory setup
# ---------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,  # avoids lazy-loading errors after commit
)

# ---------------------------------------------------
# 🧩 Base model for ORM classes
# ---------------------------------------------------
Base = declarative_base()

# ---------------------------------------------------
# 🧰 Dependency for FastAPI routes
# ---------------------------------------------------
def get_db():
    """
    Creates a new SQLAlchemy session for each request.
    Ensures rollback on error and closes session cleanly.
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("❌ Database transaction failed:")
        logger.exception(e)
        raise
    finally:
        db.close()
