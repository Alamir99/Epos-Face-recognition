import smtplib
from email.message import EmailMessage

class Verify:

    def __init__(self):
        pass
    def send_verification_email(self,sender_email, sender_password, recipient_email, subject, body):
        try:
            # Set up the email message
            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = sender_email
            message["To"] = recipient_email
            message.set_content(body)

            # Connect to the SMTP server
            with smtplib.SMTP("smtp-mail.outlook.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)

                # Send the email
                server.send_message(message)

            print("Verification email sent successfully!")

        except Exception as e:
            print(f"An error occurred while sending the email: {e}")


