from django import forms
class ImageUploadForm(forms.Form):
    image = forms.ImageField()

from django import forms
from .models import Image


class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = Image
        fields = ('title', 'image')    