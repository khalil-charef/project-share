import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
import numpy as np
import glob
from .models import Doctor
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Custom Layer
class SqueezeExcitationLayer(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(SqueezeExcitationLayer, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.dense1 = tf.keras.layers.Dense(
            channel // self.reduction_ratio, activation="relu", use_bias=False
        )
        self.dense2 = tf.keras.layers.Dense(
            channel, activation="sigmoid", use_bias=False
        )

    def call(self, inputs):
        squeeze = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        excitation = self.dense2(self.dense1(squeeze))
        excitation = tf.reshape(excitation, [-1, 1, 1, inputs.shape[-1]])
        scale = tf.keras.layers.Multiply()([inputs, excitation])
        return scale


# Load model with the custom layer
model = tf.keras.models.load_model(
    os.path.join(settings.BASE_DIR, "CNN_SAE.h5"),
    custom_objects={"SqueezeExcitationLayer": SqueezeExcitationLayer},
)

# Class labels
class_labels = {0: "حميد", 1: "خبيث"}


# Function to preprocess the image
def preprocess_image(file_path, target_size=(128, 128)):
    img = load_img(file_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


# Views
import os
import glob
from django.conf import settings

def get_started(request):
    # Directory for uploaded images
    upload_dir = os.path.join(settings.BASE_DIR, "static/uploads")
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return render(request, "skin_cancer_detection_app/get_started.html", {"error": "No file uploaded."})

        # Delete all previous images
        for file_path in glob.glob(os.path.join(upload_dir, "*")):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        # Save the uploaded file
        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            # Preprocess the image
            img, img_array = preprocess_image(file_path)

            # Predict
            predictions = model.predict(img_array)
            malignant_confidence = predictions[0][0]
            benign_confidence = 1 - malignant_confidence

            predicted_class_name = (
                class_labels[1] if malignant_confidence >= 0.5 else class_labels[0]
            )

            result = {
                "image_path": f"/static/uploads/{filename}",
                "predicted_class": predicted_class_name,
                "benign_confidence": f"{benign_confidence * 100:.2f}",
                "malignant_confidence": f"{malignant_confidence * 100:.2f}",
            }

            return render(request, "skin_cancer_detection_app/get_started.html", {"result": result})

        except Exception as e:
            return render(request, "skin_cancer_detection_app/get_started.html", {"error": f"An error occurred: {str(e)}"})

    # If the request is not POST (e.g., when clicking "Upload Another Image")
    # Delete all files in the upload directory
    for file_path in glob.glob(os.path.join(upload_dir, "*")):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    return render(request, "skin_cancer_detection_app/get_started.html")



from django.core.paginator import Paginator
def doctors_page(request):
    # Get query parameters from the request
    name_query = request.GET.get('name', '')
    city_query = request.GET.get('city', '')

    # Filter doctors based on the query
    doctors = Doctor.objects.all()
    if name_query:
        doctors = doctors.filter(name__icontains=name_query)
    if city_query:
        doctors = doctors.filter(city__icontains=city_query)

        # Paginate results: 6 doctors per page
    paginator = Paginator(doctors, 6)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'request': request,
        }

    return render(request, 'skin_cancer_detection_app/doctors.html', context)

def home(request):
    doctors = Doctor.objects.all()
    return render(request, 'skin_cancer_detection_app/home.html', {'doctors': doctors })

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
