import os
import ujson
import falcon
import bjoern
import fasttext


SAMPLES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "samples")
REVIEW_MODEL = os.path.join(SAMPLES_FOLDER, "amazon_review_full.ftz")
WEB_HOST = '127.0.0.1'
PORT = 9000

print('Loading amazon review polarity model ...')
review_classifier = fasttext.load_model(REVIEW_MODEL)


class ReviewResource(object):
    def on_post(self, req, resp):
        form = req.params
        # Evaluate if "text" have been sent
        if "text" in form and form["text"]:
            try:
                # Parse through model
                classification, confidence = review_classifier.predict(form['text'])

                # Fill in response body and status
                resp.body = ujson.dumps({f"{classification[0][-1]} star": confidence[0]})
                resp.status = falcon.HTTP_200
            except Exception:
                resp.body = ujson.dumps({"Error": "An internal server error has occurred"})
                resp.status = falcon.HTTP_500
        else:
            resp.body = ujson.dumps({"Error": "param \'text\' is mandatory"})
            resp.status = falcon.HTTP_400


def run_app():
    # Ex query:
    # curl -X POST http://localhost:9000/inferreview -H 'Content-Type: application/x-www-form-urlencoded' -d text="I love this product."

    # instantiate a callable WSGI app
    app = falcon.API()

    # Long-lived resource class instance
    infer_review = ReviewResource()

    # Landle all requests to the '/inferreview' URL path
    app.req_options.auto_parse_form_urlencoded = True
    app.add_route('/inferreview', infer_review)

    print('Listening on', WEB_HOST + ':' + str(PORT) + '/inferreview')
    bjoern.run(app, WEB_HOST, PORT, reuse_port=True)


if __name__ == "__main__":
    run_app()
