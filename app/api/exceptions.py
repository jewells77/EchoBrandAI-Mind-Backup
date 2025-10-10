class APIError(Exception):
    def __init__(
        self, message: str, status_code: int = 400, public_message: str = None
    ):
        self.message = message
        self.status_code = status_code
        self.public_message = public_message or message
        super().__init__(message)
