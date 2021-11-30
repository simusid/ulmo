from  sqlalchemy import create_engine

class UlmoDB():
    def __init__(self, dbname):
        self.engine = create_engine(f"sqlite:////{dbname}", echo=True)


