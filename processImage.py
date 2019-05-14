from google.cloud import vision
import io
import os
import sys
from google.protobuf.json_format import MessageToJson
import json
def parseJSON(json_string, path):
    words = []
    data = json.loads(json_string)
    for p in data['textAnnotations']:
        words.append(p['description'])
    with open(path[:-4] + '.txt', 'w') as f:
        for item in words:
            f.write("%s\n" % item)
    # return words

def detect_document(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)

    serialized = MessageToJson(response)
    json_string = str(serialized)
    parseJSON(json_string, path)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cs230-c9c6c6667e9a.json"
path = sys.argv[1]
detect_document(path)
