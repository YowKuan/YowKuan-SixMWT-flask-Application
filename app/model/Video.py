from app import db
from flask_sqlalchemy import SQLAlchemy

class Video(db.Model):
    __tablename__='video'
    id = db.Column(db.Integer, autoincrement=True)
    patient_id = db.Column(db.String(80), primary_key=True, nullable=False)
