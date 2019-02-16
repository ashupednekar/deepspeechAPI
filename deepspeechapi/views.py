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
#from deepspeechapi import ds
import json, os, time
from deepspeechapi.transcription import decoder, parser, model, device, decode_results

def transcribe(audio_path, parser, model, decoder, device):
    spect = parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets

class FileView(APIView):

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = FileSerializer(data=request.data, required=False)
        first = time.time()
        if file_serializer.is_valid():
            file_serializer.save()
            aud = file_serializer.data["file"][1:]
            os.system('ffmpeg -i '+aud+' -acodec pcm_s16le -ac 1 -ar 16000 '+ aud.split('.')[0]+'.wav -y')
            #fs, audio = wav.read(aud.split('.')[0]+'.wav')
            #processed_data = ds.stt(audio, fs)
            os.chdir('/home/ashu/Documents/bbuddy/dstorch/deepspeech.pytorch')
            transcript = json.loads(os.popen('python transcribe.py --model-path models/deepspeech_final.pth --audio-path ' + aud.split('.')[0] + '.wav' + ' --decoder beam --lm-path data/custom/text.binary ').read())['output'][0]['transcription']
            os.chdir('/home/ashu/Documents/bbuddy/deepspeechAPI/deepspeechapi/')
            decoded_output, decoded_offsets = transcribe(aud.split('.')[0]+'.wav', parser, model, decoder, device)
            res = decode_results(model, decoded_output, decoded_offsets)
            res["transcript"] = transcript
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
