FROM python:3.8-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app \ 
    && chown user:user /opt/app 


USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

#RUN python -m pip install --user SimpleITK

COPY --chown=user:user requirements.txt /opt/app/
RUN pip3 install --no-cache-dir -r /opt/app/requirements.txt



# RUN python -m piptools sync requirements.txt

# COPY --chown=user:user weights /opt/app/weights
COPY --chown=user:user model_final.pth /opt/app/weights/

COPY --chown=user:user evaluate.py /opt/app/ 
COPY --chown=user:user saved_images /opt/app/saved_images
COPY --chown=user:user output_images /opt/app/output_images
COPY --chown=user:user post_output_images /opt/app/post_output_images
COPY --chown=user:user model_folder /opt/app/model_folder
COPY --chown=user:user nnunetv2 /opt/app/nnunetv2
COPY --chown=user:user ground-truth/ground_truth_segmentation.json /opt/app/ground-truth/

ENTRYPOINT [ "python", "-m", "evaluate" ]
