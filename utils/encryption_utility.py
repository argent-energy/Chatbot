from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64
import time
import os
from dotenv import load_dotenv
import datetime

load_dotenv(".env.gpt4", override=True)

def encrypt_string(password, plaintext):
    # Convert password and plaintext to bytes
    password_bytes = password.encode('utf-8')
    plaintext_bytes = plaintext.encode('utf-8')

    # Generate a random IV
    iv = get_random_bytes(AES.block_size)

    # Pad plaintext to a multiple of 16 bytes
    padded_plaintext = pad(plaintext_bytes, AES.block_size)

    # Create an AES cipher object with the provided password and IV
    cipher = AES.new(password_bytes, AES.MODE_CBC, iv)

    # Encrypt the padded plaintext
    ciphertext = cipher.encrypt(padded_plaintext)

    # Return the encrypted ciphertext and IV as bytes
    return base64.b64encode(iv + ciphertext).decode('ascii')

def decrypt_string(password, ciphertext):
    # Convert password and ciphertext to bytes
    password_bytes = password.encode('utf-8')

    # Decode the ciphertext from Base64 back into bytes
    ciphertext = base64.b64decode(ciphertext.encode('ascii'))

    # Extract the IV and ciphertext from the input
    iv = ciphertext[:AES.block_size]
    ciphertext = ciphertext[AES.block_size:]

    # Create an AES cipher object with the provided password and IV
    cipher = AES.new(password_bytes, AES.MODE_CBC, iv)

    # Decrypt the ciphertext
    padded_plaintext = cipher.decrypt(ciphertext)

    # Unpad the plaintext and convert it to a string
    plaintext = unpad(padded_plaintext, AES.block_size).decode('utf-8')

    # Return the decrypted plaintext
    return plaintext

def main():
    # Get a password from environment variables
    password = os.environ["ENCRYPTION_KEY"]

    # Get email from the user
    email = input('Enter an email address: ')

    # Get config to load from the user
    config = input('Enter the config to load (ams/canvas/tares/test): ')

    # Get the number of days and hours to expire from the user
    days = int(input('Enter the number of days to expire: '))
    hours = int(input('Enter the number of hours to expire: '))
    seconds = (days * 24 * 60 * 60) + (hours * 60 * 60)
    
    # Get the current time in seconds since the epoch
    now = int(time.time())

    # Calculate the expiration time
    expires = now + seconds

    # Create a string in the format: email:expires
    plaintext = f'{email}:{config}:{expires}'

    # Encrypt the plaintext
    ciphertext = encrypt_string(password, plaintext)

    # Print the encrypted ciphertext
    print('Encrypted:', ciphertext)

    # Decrypt the ciphertext
    decrypted = decrypt_string(password, ciphertext)

    # Print the decrypted plaintext
    print('Decrypted:', decrypted)
    
    print(datetime.datetime.fromtimestamp(expires).strftime('Expires on %d %B %Y at %H:%M:%S IST'))

if __name__ == '__main__':
    main()