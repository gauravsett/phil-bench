class BaseModel:
    
    def __init__(self, config):
        self.config = config
    
    def respond(self, question, options):
        response = [1/len(options) for _ in options]
        return response

