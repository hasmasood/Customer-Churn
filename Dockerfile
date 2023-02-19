FROM python:3.7.0-slim

WORKDIR /app
COPY ["requirements.txt", "model_stack_C=1.0.pkl", "app_api.py", "/app/"] 

#For some reason installing dependencies failed on old pip, so have to upgrade pip first before libraries installation
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

EXPOSE 9696
CMD gunicorn --bind=0.0.0.0:9696 app_api:app
#ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app_api:app"]