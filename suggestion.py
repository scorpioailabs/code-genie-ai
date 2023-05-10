class Suggestion:
    def __init__(self, file, suggestion):
        self.file = file
        self.suggestion = suggestion

    def __hash__(self):
        return hash((self.file, self.suggestion))

    def __eq__(self, other):
        if isinstance(other, Suggestion):
            return self.file == other.file and self.suggestion == other.suggestion
        return False
