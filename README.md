# Instructions

Optional (but recommended): Install ffmpeg

Linux:

```sh
sudo apt install ffmpeg
```

MacOS:

```sh
brew install ffmpeg
```

Windows:

https://www.ffmpeg.org/download.html

<hr>

1. Create a Python virtual environment

```sh
python -m venv .venv
source .venv/bin/activate
```

2. Install the requirements

```sh
python -m pip install -r requirements.txt
```

3. Run the Flask app

```sh
python -m flask run --host=0.0.0.0 --port=5180
```

4. Open the app in your browser at http://localhost:5180

5. Choose a video file and click the "Submit" button

6. Wait for the new video to be generated
