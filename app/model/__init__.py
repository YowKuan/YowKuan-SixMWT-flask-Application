from app.model.Distance_bar import Distance_bar
from app.model.Speed_chart import Speed_chart
from app.model.Rounds import Rounds
from app.model.Distance_speed import Distance_speed
from app.model.Video import Video

class Model:
    def __init__(self, app=None):
        if app is not None:
            self.init_app(app)
    def init_app(self, app):
        app.model = self

        self.Distance_bar = Distance_bar
        self.Distance_speed = Distance_speed
        self.Speed_chart = Speed_chart
        self.Rounds = Rounds
        self.Video = Video