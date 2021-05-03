from django.shortcuts import render
from .forms import ImageUploadForm
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np




# def handle_uploaded_file(f):
#     with open('img.jpg', 'wb+') as destination:
#         for chunk in f.chunks():
#             destination.write(chunk)

def handle_uploaded_file(f):
    with open('static/media/image.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def home(request):
   return render(request, 'home.html')
def ImageUpload(request):
    return render(request, 'uploadImage.html')

def imageprocess(request): # with torchvision
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])

        print(f"Running image through new model")
        import tensorflow as tf
        from distutils.version import StrictVersion
        # if StrictVersion(tf.__version__) != StrictVersion('1.13.1'):
        #     raise ImportError('Please use tensorflow 1.13.1 for this notebook.')
        print(f"Tensorflow Version: {tf.__version__}")

        # Configuring plotting
        import matplotlib.pyplot as plt
        #%matplotlib inline # from jupyter-notebook
        import os
        # Importing some common libraries
        import urllib
        import tarfile
        import numpy as np

        # Getting the model file
        #print(f"Getting the model file")
        MODEL_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
        MODEL_FILE = MODEL_NAME + '.tar.gz'

        if os.path.exists(MODEL_FILE) is False:
            opener = urllib.request.URLopener()
            opener.retrieve(MODEL_DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

        # Getting the box description file
        #print(f"Getting the box description file ... ")        
        BOX_DESCRIPTIONS_FILE = 'class-descriptions-boxable.csv'
        OID_DOWNLOAD_BASE = 'https://storage.googleapis.com/openimages/2018_04/'

        if os.path.exists(BOX_DESCRIPTIONS_FILE) is False:
            opener = urllib.request.URLopener()
            opener.retrieve(OID_DOWNLOAD_BASE + BOX_DESCRIPTIONS_FILE, BOX_DESCRIPTIONS_FILE)

        # # Getting test images
        # TEST_IMAGES = {
        #     #'cat.jpg': 'https://c2.staticflickr.com/7/6118/6370710013_cb6b0270d3_o.jpg',
        #     #'dog.jpg': 'https://c3.staticflickr.com/1/92/246323809_f8a8ab71fe_o.jpg',
        #     'pets.jpg': 'https://live.staticflickr.com/3273/2982384735_eeecaf03f2_b.jpg'
        # }

        # for filename, url in TEST_IMAGES.items():
        #     if os.path.exists(filename) is False:
        #         opener = urllib.request.URLopener()
        #         opener.retrieve(url, filename)

        # Extracting Model files
        #print(f"Extracting Model files")
        FROZEN_GRAPH_FILE = 'frozen_inference_graph.pb'
        PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME, FROZEN_GRAPH_FILE)

        if os.path.exists(MODEL_NAME) is False:
            tar_file = tarfile.open(MODEL_FILE)
            for file in tar_file.getmembers():
                filename = os.path.basename(file.name)
                if FROZEN_GRAPH_FILE in filename:
                    tar_file.extract(file, os.getcwd())

        # Loading box descriptions into dictionary     
        import pandas as pd

        ID_KEY = 'id'
        CLASS_KEY = 'class'
        NAME_KEY = 'name'

        df = pd.read_csv(BOX_DESCRIPTIONS_FILE, names=[ID_KEY, CLASS_KEY])
        category_index = {}
        for idx, row in df.iterrows():
            category_index[idx+1] = {ID_KEY: row[ID_KEY], NAME_KEY: row[CLASS_KEY]}

        # Load model from frozen file
        print(f"Load model from frozen file")
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef() 
            with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
            
                    
        # Checking test images 
        #print("Checking test images ...")          
        # from PIL import Image
        # IMAGE_SIZE = (12, 8)
        # def show_image(filename, image):
        #     """
        #     Shows the given image with its filename as title.
        #     :param filename: image filename
        #     :param image: image to show
        #     """
        #     plt.figure(figsize=IMAGE_SIZE)
        #     plt.title(filename)
        #     plt.axis('off')
        #     plt.imshow(image)

        # for filename, _ in TEST_IMAGES.items():
        #     original_image = Image.open(filename)
        #     show_image(filename, original_image)                

        # Fonts and Colors
        FONT_NAME = 'Ubuntu-R.ttf'
        # Bounding box colors
        COLORS = ['Green',
                  'Red', 'Pink',
                  'Olive', 'Brown', 'Gray',
                  'Cyan', 'Orange']

        class ObjectResult:
            """
            Represents a detection result, containing the object label,
            score confidence, and bounding box coordinates.
            """
            def __init__(self, label, score, box):
                self.label = label
                self.score = score
                self.box = box
            
            def __repr__(self):
                return '{0} ({1}%)'.format(self.label, int(100 * self.score))

        # Convert PIL to numpy array
        #print(f"Convert PIL to numpy array ... ")
        N_CHANNELS = 3

        def load_image_into_numpy_array(image):
            """
            Converts a PIL image into a numpy array (height x width x channels).
            :param image: PIL image
            :return: numpy array
            """
            (width, height) = image.size
            return np.array(image.getdata()) \
                .reshape((height, width, N_CHANNELS)).astype(np.uint8)

        # Processes classes, scores, and boxes, gathering in a list of ObjectResult
        #print(f"Processes classes, scores, and boxes, gathering in a list of ObjectResult ")
        def process_output(classes, scores, boxes, category_index):
            """
            Processes classes, scores, and boxes, gathering in a list of ObjectResult.
            :param classes: list of class id
            :param scores: list of scores
            :param boxes: list of boxes
            :param category_index: label dictionary
            :return: list of ObjectResult
            """
            results = []

            for clazz, score, box in zip(classes, scores, boxes):
                if score > 0.0:
                    label = category_index[clazz][NAME_KEY]
                    obj_result = ObjectResult(label, score, box)
                    results.append(obj_result)
            
            return results

        # Draws labeled boxes according to results on the given image
        #print(f"Draws labeled boxes according to results on the given image")
        import random
        import PIL.Image as Image
        def draw_labeled_boxes(image_np, results, min_score=.6):
            """
            Draws labeled boxes according to results on the given image.
            :param image_np: numpy array image
            :param results: list of ObjectResult
            :param min_score: optional min score threshold, default is 40%
            :return: numpy array image with labeled boxes drawn
            """
            results.sort(key=lambda x: x.score, reverse=False)
            image_np_copy = image_np.copy()
            for r in results:
                if r.score >= min_score:
                    color_idx = random.randint(0, len(COLORS) - 1)
                    color = COLORS[color_idx]

                    image_pil = Image.fromarray(np.uint8(image_np_copy)).convert('RGB')
                    draw_bounding_box_on_image(image_pil, r.box, color, str(r))
                    np.copyto(image_np_copy, np.array(image_pil))

            return image_np_copy

        # Calculates a suitable font for the image given the text and fraction
        #print(f"Calculates a suitable font for the image given the text and fraction")
        import PIL.ImageFont as ImageFont

        def get_suitable_font_for_text(text, img_width, font_name, img_fraction=0.12):
            """
            Calculates a suitable font for the image given the text and fraction.
            :param text: text that will be drawn
            :param img_width: width of the image
            :param font_name: name of the font
            :param img_fraction: optional desired image fraction allowed for the text 
            :return: suitable font
            """
            font = ImageFont.truetype("static/arial.ttf", 35)
        #    font = ImageFont.truetype("arial.ttf", 18)
        #     fontsize = 1
        #     font = ImageFont.truetype(FONT_NAME, fontsize)
        #     while font.getsize(text)[0] < img_fraction*img_width:
        #         fontsize += 1
        #         font = ImageFont.truetype(font_name, fontsize)
            return font

        # Draws the box and label on the given image.
        #print(f"Draws the box and label on the given image  ")
        import PIL.ImageDraw as ImageDraw
        TEXT_COLOR = 'Black'

        def draw_bounding_box_on_image(image, box, color, box_label):
            """
            Draws the box and label on the given image.
            :param image: PIL image
            :param box: numpy array containing the bounding box information
                        [top, left, bottom, right]
            :param color: bounding box color
            :param box_label: bounding box label
            """
            im_width, im_height = image.size
            top, left, bottom, right = box

            # Normalize coordinates
            left = left * im_width
            right = right * im_width
            top = top * im_height
            bottom = bottom * im_height

            # Draw the detected bounding box
            line_width = int(max(im_width, im_height) * 0.005)
            draw = ImageDraw.Draw(image)
            draw.rectangle(((left, top), (right, bottom)),
                           width=line_width,
                           outline=color)

            # Get a suitable font (in terms of size with respect to the image)
            font = get_suitable_font_for_text(box_label, im_width, FONT_NAME)
            text_width, text_height = font.getsize(box_label)

            # Draw the box label rectangle
            text_bottom = top + text_height
            text_rect = ((left, top),
                         (left + text_width + 2 * line_width,
                          text_bottom + 2 * line_width))
            draw.rectangle(text_rect, fill=color)

            # Draw the box label text 
            # right below the upper-left horizontal line of the bounding box
            text_position = (left + line_width, top + line_width)
            draw.text(text_position, box_label, fill=TEXT_COLOR, font=font)

        # Input tensor
        IMAGE_TENSOR_KEY = 'image_tensor'

        # Output tensors
        DETECTION_BOXES_KEY = 'detection_boxes'
        DETECTION_SCORES_KEY = 'detection_scores'
        DETECTION_CLASSES_KEY = 'detection_classes'

        TENSOR_SUFFIX = ':0'

        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        print(f"Run Inference ")
        def run_inference(graph, image_np):
            """
            Runs the inference on the given image.
            :param graph: tensorflow graph
            :param image_np: numpy image
            :return: dictionary with detected classes 
                     and their corresponding scores and boxes
            """
            output_tensor_dict = {
                DETECTION_BOXES_KEY: DETECTION_BOXES_KEY + TENSOR_SUFFIX,
                DETECTION_SCORES_KEY: DETECTION_SCORES_KEY + TENSOR_SUFFIX,
                DETECTION_CLASSES_KEY: DETECTION_CLASSES_KEY + TENSOR_SUFFIX
            }

            with graph.as_default():
                with tf.Session() as sess:
                    input_tensor = tf.get_default_graph()\
                        .get_tensor_by_name(IMAGE_TENSOR_KEY + TENSOR_SUFFIX)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    input_tensor_dict = {input_tensor: image_np_expanded}
                    output_dict = sess.run(output_tensor_dict,
                                           feed_dict=input_tensor_dict)

                    return {
                        DETECTION_BOXES_KEY: 
                            output_dict[DETECTION_BOXES_KEY][0],
                        DETECTION_SCORES_KEY: 
                            output_dict[DETECTION_SCORES_KEY][0],
                        DETECTION_CLASSES_KEY: 
                            output_dict[DETECTION_CLASSES_KEY][0].astype(np.int64)
                    }

        # Running Inference
        IMAGE_NP_KEY = 'image_np'
        RESULTS_KEY = 'results'

        file_result_dict = {}
        files = ["static/media/image.jpg"]
        for filename in files:
            image_np = load_image_into_numpy_array(Image.open(filename))
            
            output_dict = run_inference(graph, image_np)
         
            results = process_output(output_dict[DETECTION_CLASSES_KEY],
                                     output_dict[DETECTION_SCORES_KEY],
                                     output_dict[DETECTION_BOXES_KEY],
                                     category_index)

            file_result_dict[filename] = { IMAGE_NP_KEY: image_np, RESULTS_KEY: results }
        
        # Save Image
        print(f"Saving Image Result")
        for filename, img_results_dict in file_result_dict.items():
            processed_image = draw_labeled_boxes(img_results_dict[IMAGE_NP_KEY], 
                                                 img_results_dict[RESULTS_KEY])
            results = img_results_dict[RESULTS_KEY]
            #print(f"Results: {img_results_dict[RESULTS_KEY]}")            
            #show_image(filename, processed_image)
            
            im = Image.fromarray(processed_image)
            im.save("static/media/ImageResult.jpg")

        print(f"RESULT: {results}")

        res = []
        for e in results:
            res.append(e)
        return render(request, 'result.html',{'res':res})        

    return render(request, 'result.html')


def resnet50_imageprocess(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])

        model = ResNet50(weights='imagenet')
        img_path = 'img.jpg'

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)

        print('Predicted:', decode_predictions(preds, top = 3)[0])

        html = decode_predictions(preds, top = 3)[0]
        res = []
        for e in html:
            res.append((e[1], np.round(e[2]*100,2)))
        return render(request, 'result.html',{'res':res})

    return render(request, 'uploadImage.html')    