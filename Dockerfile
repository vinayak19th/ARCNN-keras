# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:2.3.2-gpu-jupyter
RUN echo "Creating Docker image for ARCNN-keras"
LABEL version="1.0"

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN pip install --upgrade pip
RUN pip install jupyterlab
RUN pip uninstall jedi -y


WORKDIR /app
VOLUME /data

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app
USER appuser
