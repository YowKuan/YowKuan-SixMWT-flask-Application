from app import db
from flask_sqlalchemy import SQLAlchemy

class Rounds(db.Model):
    __tablename__ = 'rounds'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String, nullable=False)
    group = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False)
    round_count = db.Column(db.Integer, nullable=False)
    time = db.Column(db.Float, nullable=False)