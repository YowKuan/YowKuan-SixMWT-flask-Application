from app import db
from flask_sqlalchemy import SQLAlchemy

class Distance_bar(db.Model):
    __tablename__ = 'distance_bar'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String, nullable=False)
    group = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False)
    distance = db.Column(db.Float, nullable=False)

