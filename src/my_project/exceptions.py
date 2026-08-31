import sys

def error_message_details(error_message, error_details: sys):

    _, _, exc_tb = error_details.exc_info()

    line_number = exc_tb.tb_lineno

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"error message: [{error_message}]"
    )

    return error_message


class CustomException(Exception):

    def __init__(self, error_message, error_details: sys):

        self.error_message = error_message_details( error_message, error_details )

        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message