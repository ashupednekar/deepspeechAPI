import argparse
import warnings

from deepspeechapi.opts import add_decoder_args, add_inference_args
from deepspeechapi.utils import load_model

warnings.simplefilter('ignore')

from deepspeechapi.decoder import GreedyDecoder

import torch

from deepspeechapi.data_loader import SpectrogramParser
from deepspeechapi.model import DeepSpeech
import os.path
import json


def decode_results(model, decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }
    #results['_meta']['acoustic_model'].update(DeepSpeech.get_meta(model))

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results



parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--audio-path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
parser = add_decoder_args(parser)
args = parser.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")
model = load_model(device, args.model_path, args.cuda)

from deepspeechapi.decoder import BeamCTCDecoder

decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                         cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                         beam_width=args.beam_width, num_processes=args.lm_workers)
#decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

parser = SpectrogramParser(model.audio_conf, normalize=True)

#decoded_output, decoded_offsets = transcribe(args.audio_path, parser, model, decoder, device)
#print(json.dumps(decode_results(model, decoded_output, decoded_offsets)))
