from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
from .serializers import FileSerializer
import scipy.io.wavfile as wav
from deepspeechapi import ds
import json, os, time

class FileView(APIView):

  parser_classes = (MultiPartParser, FormParser)

  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data, required=False)
    first = time.time()
    if file_serializer.is_valid():
      file_serializer.save()
      aud = file_serializer.data["file"][1:]
      os.system('ffmpeg -i '+aud+' -acodec pcm_s16le -ac 1 -ar 16000 '+ aud.split('.')[0]+'.wav -y')
      fs, audio = wav.read(aud.split('.')[0]+'.wav')
      processed_data = ds.stt(audio, fs)
      res = dict()
      res["transcript"] = processed_data
      print(time.time() - first)
      return Response(res, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# @api_view(["POST"])
# def Inference(data):
#     try:
#         x = json.loads(data.body)
#         y = str(x*10)
#         return JsonResponse('API response is: '+y, safe=False)
#     except ValueError as e:
#         return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
