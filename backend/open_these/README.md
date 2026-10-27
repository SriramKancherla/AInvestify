Open these in the review.

website/api.py
    The website server. Every page request comes here.

diffusion_model/network.py
    The diffusion network. This is the model.

diffusion_model/diffusion_math.py
    How noise is added in training and removed at forecast time.

diffusion_model/live_forecast.py
    What the website runs when someone asks for a chart forecast.

diffusion_model/weights
    The saved weights the website loads. File inside: latest.pt

diffusion_model/training_that_saved_the_weights.py
    The training run that wrote those weights.

diffusion_model/chart_channels.py
    How a price window becomes the three image channels.

score_model/score_engine.py
    The score and the written insight on the dashboard.

score_model/how_each_score_was_trained
    One file per score model: random forest, classifier, XGBoost, neural net, sentiment.

website/chart_forecast.py
    The API call that starts the diffusion forecast.

website/insights.py
    The API call that starts the score.

website/sec_filings.py
    The API call that loads SEC filings.
