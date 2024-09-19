from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from deep_translator import GoogleTranslator
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages

# Load the pre-trained BLIP model and processor
model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)


# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if file is allowed (based on extension)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Generate caption with options for 'brief' or 'detailed'
def generate_caption(image_path, mode="brief"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    if mode == "brief":
        # Generate a short caption
        out = model.generate(**inputs, max_length=40, num_beams=3)
    else:
        # Generate a detailed caption
        out = model.generate(**inputs, max_length=300, num_beams=5, repetition_penalty=2.0)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def translate_caption(caption, target_language):
    translator = GoogleTranslator(source='auto', target=target_language)
    return translator.translate(caption)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Retrieve caption type (brief or detailed)
        caption_type = request.form.get('caption_type', 'brief')
        target_language = request.form.get('language', 'en') # Get the target language from the form

        try:
            # Generate caption based on user selection
            caption = generate_caption(file_path, mode=caption_type)
            translated_caption = translate_caption(caption, target_language)
        except Exception as e:
            flash(f'Error generating caption: {e}')
            return redirect(request.url)

        flash('Image successfully uploaded and caption generated')
        return render_template('result.html', caption=translated_caption, image=filename)
    
    flash('Invalid file format')
    return redirect(request.url)


@app.route('/gallery')
def gallery():
    images = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]
    return render_template('gallery.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
