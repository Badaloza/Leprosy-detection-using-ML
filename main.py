import numpy as np
import tensorflow as tf
import cv2
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from flask_ngrok import run_with_ngrok


def run(x, m):
  #frame = cv2.imread(x)
  #im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #resized_image = cv2.resize(x, (h,h), interpolation = cv2.INTER_CUBIC)
  resized_image = x.astype(np.float32)
  img = np.expand_dims(resized_image, 0) 

  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=m)
  interpreter.allocate_tensors()
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  # Test the model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], img)
  interpreter.invoke()
  output = interpreter.get_tensor(output_details[0]['index'])

  return (output)
  
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)
  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 10) #default 25
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image

def show(li, path):
  li = li.tolist()
  a = np.array(li, dtype=np.float32)
  b = np.array([b'1', b'2',b'3', b'4',b'5', b'6',b'7', b'8',b'9', b'10'], dtype=np.object)
  c = np.array([0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9], dtype=np.float32)
  return (draw_boxes(path,a,b,c))


allowed_exts = {'jpg', 'jpeg','png','JPG','JPEG','PNG'}
app = Flask(__name__)
run_with_ngrok(app)
app.secret_key = 'the random string'
def check_allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

@app.route("/",methods=['GET', 'POST'])
def index():
 if request.method == 'POST':
  if 'file' not in request.files:
   print('No file attached in request')
   return redirect(request.url)
  file = request.files['file']
  if file.filename == '':
   print('No file selected')
   return redirect(request.url)
  if file and check_allowed_file(file.filename):
   filename = secure_filename(file.filename)
   print(filename)
   img = Image.open(file.stream).convert("RGB") 
   img = np.asarray(img)
   rimg = cv2.resize(img, (300,300), interpolation = cv2.INTER_CUBIC)
   cc = 'blp_detect_float.tflite'
   img = show(*run( rimg, cc), rimg)
   flash(str(run( rimg, cc)))
   img = Image.fromarray(img)
   with BytesIO() as buf:
    img.save(buf, 'jpeg')
    image_bytes = buf.getvalue()
   encoded_string = base64.b64encode(image_bytes).decode()         
  return render_template('index.html', img_data=encoded_string), 200
 else:
  return render_template('index.html', img_data=""), 200

if __name__ == "__main__":
 app.debug=True
 app.run()
