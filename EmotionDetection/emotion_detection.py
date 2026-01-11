import requests
import json

def emotion_detector(text_to_analyze):
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, headers=headers, json=payload)
    data = json.loads(response.text)

    # Tomar el bloque base donde viven las emociones
    emo_block = data["emotionPredictions"][0]["emotion"]

    # Intentar distintas rutas válidas (según el formato que devuelva el servicio)
    emotions = None

    # Formato A: document -> emotion
    if "document" in emo_block and "emotion" in emo_block["document"]:
        emotions = emo_block["document"]["emotion"]

    # Formato B: targets -> [0] -> emotions
    elif "targets" in emo_block and emo_block["targets"] and "emotions" in emo_block["targets"][0]:
        emotions = emo_block["targets"][0]["emotions"]

    # Formato C: emotion ya viene directo
    elif all(k in emo_block for k in ["anger", "disgust", "fear", "joy", "sadness"]):
        emotions = emo_block

    else:
        # Si el formato es inesperado, falla de forma clara
        raise KeyError(f"Formato inesperado en respuesta: {list(emo_block.keys())}")

    scores = {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
    }
    scores["dominant_emotion"] = max(scores, key=scores.get)
    return scores
