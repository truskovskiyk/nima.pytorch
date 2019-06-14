from flask import Flask, redirect, url_for, request, jsonify
from PIL import Image
from flasgger import Swagger

from nima.inference.inference_model import InferenceModel

app = Flask(__name__)
Swagger(app=app)
app.model = InferenceModel.create_model()


def format_output(mean_score, std_score, prob):
    return {
        'mean_score': float(mean_score),
        'std_score': float(std_score),
        'scores': [float(x) for x in prob]
    }


@app.route('/')
def index():
    return redirect(url_for('health_check'))


@app.route('/api/health_check')
def health_check():
    return "ok"


@app.route('/api/get_scores', methods=['POST'])
def get_scores():
    """
    NIMA Pytorch

    ---
    tags:
      - Get Scores
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        type: file
        name: file
        required: true
        description: Upload your file.
    responses:
      200:
        description: Scores for image
        schema:
          id: Palette
          type: object
          properties:
            mean_score:
              type: float
            std_score:
              type: float
            scores:
              type: array
              items:
                type: float
        examples:
          {
              "mean_score": 5.385255615692586,
              "scores": [
                0.0049467734061181545,
                0.018246186897158623,
                0.05434520170092583,
                0.16275958716869354,
                0.3268744945526123,
                0.24433879554271698,
                0.11257114261388779,
                0.05015537887811661,
                0.017528045922517776,
                0.00823438260704279
              ],
              "std_score": 1.451693009595486
            }
    """

    img = Image.open(request.files['file'])
    result = app.model.predict_from_pil_image(img)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
