# 🏥 Insurance Charges Predictor — Django

> **Most models give you a number. This one tells you how much to trust it.**

🔗 **Live demo: [health-insurance-ml-project.onrender.com](https://health-insurance-ml-project.onrender.com/)**

> ℹ️ Hosted on Render's free tier, so the first load after a period of inactivity can take up to a minute while the app wakes up.

---

## The Problem

Health insurance pricing is noisy. Two people with the same age and BMI can receive very different charges, and a single-number prediction hides that uncertainty. A model that scores well on a test set is the starting line, not the finish line: what matters is whether you can tell *when it is likely to be wrong*.

## The Solution

Enter six details (**age, sex, BMI, children, smoker status, and region**) and the app returns:

- 💰 **A predicted annual insurance charge** from a gradient boosting model
- 📊 **A 90% confidence interval** around that prediction, calculated per input
- 🚩 **A low-confidence flag** that marks predictions the model is less certain about and recommends manual review

The confidence interval comes from two extra models trained with quantile loss (5th and 95th percentile), so it needs no ground truth at prediction time. Wide gap, less certainty.

## Try It

Open the [live app](https://health-insurance-ml-project.onrender.com/) and compare a young non-smoker with an older smoker with a high BMI. Watch how the interval widens for the harder-to-predict profiles.

---

## 1. Add the model artifacts

Run **Section 9 ("Deployment Prep")** of the training notebook. It exports:

```
gb_model.pkl      # point-estimate model (mean prediction)
gb_lower.pkl      # 5th percentile quantile model
gb_upper.pkl      # 95th percentile quantile model
metadata.pkl      # feature order, encoding maps, flag threshold, test metrics
```

Download `model_artifacts.zip` from Colab and unzip its contents into:

```
predictor/model_artifacts/
```

## 2. Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python manage.py migrate        # creates db.sqlite3 (admin/auth tables only)
python manage.py runserver
```

Visit http://127.0.0.1:8000/

## 3. Deploy

The live version runs on Render at
[health-insurance-ml-project.onrender.com](https://health-insurance-ml-project.onrender.com/).
Any host that runs Django works (Render, Railway, PythonAnywhere, Fly.io).
General steps for Render/Railway:

1. Push this project to GitHub (model artifacts included, or fetched at build time).
2. Set environment variables: `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=False`,
   `DJANGO_ALLOWED_HOSTS=yourapp.onrender.com`.
3. Build command: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
4. Start command: `gunicorn charges_project.wsgi:application`

`whitenoise` is included in requirements so static files serve correctly in
production without extra configuration. Add it to `MIDDLEWARE` in
`settings.py` (right after `SecurityMiddleware`) and set
`STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"`
if you deploy this.

## How the confidence flag works

Two extra models were trained with quantile loss (5th and 95th percentile)
alongside the main model. Their prediction gap is a per-input confidence
interval, computed with no ground truth needed. Inputs whose interval is
wider than the 90th-percentile threshold observed on the test set get
flagged as "low confidence — recommend manual review." See Sections 8 and 9
of the training notebook for the full reasoning and validation.
