# Import necessary libraries and modules
import os
from openai import OpenAI
import openai
import base64
import requests
import cv2
import sounddevice as sd
import soundfile as sf
import io
import keyboard
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI()

# Function to get description for a single image using OpenAI GPT-4 Vision model
def image_description_1_image(image):
    """
    Generate a textual description for a single image using OpenAI GPT-4 Vision model.

    Args:
        image (str): Base64-encoded image.

    Returns:
        str: Generated textual description.
    """
    # Set headers for API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    # Define payload for API request
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Describe the image. Keep it brief. Don't start with 'The image shows'. Just give the description."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    },
                ]
            }
        ],
        "max_tokens": 300
    }

    # Make API request and return the generated description
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

# Function to get description for two consecutive images with contextual analysis
def image_description_2_images(image1, image2, image_description_prev):
    """
    Generate a textual description for two consecutive images with contextual analysis using OpenAI GPT-4 Vision model.

    Args:
        image1 (str): Base64-encoded previous frame image.
        image2 (str): Base64-encoded current frame image.
        image_description_prev (str): Description of the previous frame image.

    Returns:
        str: Generated textual description for the current frame.
    """
    # Set headers for API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    # Define payload for API request
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""The images are two consecutive frames of a continuous live video. 
                                    The first image is of the previous frame, and the second image is of the current frame.
                                    The description of the previous frame is {image_description_prev}.
                                    Compare the two images and also compare the description of both the frames.
                                    Then describe the current frame or anything new comes in the description. 
                                    Don't repeat anything that is already there in the previous frame or {image_description_prev}.
                                    Always make a connection between the two frames and between the current description and {image_description_prev} 
                                    as these are the images from a continuous live video feed. Don't mention anything about comparison in your final answer.
                                    Just describe the current frame after doing the above analysis. Don't start with 'in this frame'.  Keep it brief."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image1}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image2}"}
                    },
                ]
            }
        ],
        "max_tokens": 300
    }

    # Make API request and return the generated description
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

# Function to convert text to speech using OpenAI TTS-1-HD model
def text_to_speech(text):
    """
    Convert input text to speech using OpenAI TTS-1-HD model and play the generated audio.

    Args:
        text (str): Input text to be converted to speech.
    """
    # Generate audio from text using OpenAI TTS-1-HD model
    spoken_response = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        response_format="opus",
        input=text
    )

    # Convert audio response to playable format and play the audio
    buffer = io.BytesIO()
    for chunk in spoken_response.iter_bytes(chunk_size=4096):
        buffer.write(chunk)
    buffer.seek(0)

    with sf.SoundFile(buffer, 'r') as sound_file:
        data = sound_file.read(dtype='int16')
        sd.play(data, sound_file.samplerate)
        sd.wait()

# Set up video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Function to continuously describe live video frames and display the video
def live_video_description(cycle, cap=cap, window_name='frame'):
    """
    Continuously describe live video frames and display the video.

    Args:
        cycle (int): Number of frames to wait before describing the next frame.
        cap (cv2.VideoCapture): Video capture object.
        window_name (str): Name of the window to display the video.
    """
    if not cap.isOpened():
        return

    image_description_prev = ""
    n = 0
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if keyboard.is_pressed("q"):
            break

        if n == cycle:
            n = 0
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Call appropriate function based on whether it's the first frame or not
            if i == 0:
                image_description_crnt = image_description_1_image(base64_frame)
            else:
                image_description_crnt = image_description_2_images(base64_frame_prev, base64_frame, image_description_prev)

            # Print and speak the generated description
            print("\n Description==========>", image_description_crnt, "\n")
            text_to_speech(text=image_description_crnt)

            # Update the previous description and frame for the next iteration
            image_description_prev = " ".join([image_description_prev, image_description_crnt])
            base64_frame_prev = base64_frame
            i += 1
        n += 1

# Function to display the live video feed
def display_video(cap=cap):
    """
    Display the live video feed.

    Args:
        cap (cv2.VideoCapture): Video capture object.
    """
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

# Main function to start the video display and description threads
def main():
    threading.Thread(target=display_video).start()
    threading.Thread(target=live_video_description, args=(5,)).start()

# Entry point of the script
if __name__ == "__main__":
    main()
