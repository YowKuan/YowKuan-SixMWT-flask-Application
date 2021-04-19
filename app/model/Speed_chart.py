from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class Speed_chart(db.Model):
    __tablename__ = 'speed_chart'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(80), nullable = False)
    group = db.Column(db.Integer, nullable = False)
    date = db.Column(db.Date, nullable = False)
    time = db.Column(db.Integer, nullable = False)
    speed = db.Column(db.Float, nullable = False)



