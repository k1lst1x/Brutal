from sqlalchemy import Column, Float, Integer, String, DateTime

from fortehackathon_brutal_backend.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    transdatetime = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    direction = Column(String, nullable=False)
