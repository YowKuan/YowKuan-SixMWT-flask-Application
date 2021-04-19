from app import db
from flask_sqlalchemy import SQLAlchemy

class Distance_speed(db.Model):
    __tablename__ = 'distance_speed'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String, nullable=False)
    group = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False)
    distance = db.Column(db.Float, nullable=False)
    speed = db.Column(db.Float, nullable=False)
    moving_avg = db.Column(db.Float)
    time = db.Column(db.Float, nullable=False)