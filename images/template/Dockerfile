# ref: https://cloud.google.com/dataflow/docs/guides/templates/configuring-flex-templates#use_custom_container_images
FROM gcr.io/dataflow-templates-base/python3-template-launcher-base as template_launcher

# This is one of the worker images
FROM gcr.io/gcp-cset-projects/arxiv-classifier-v2/ai-weak-negs:latest

# Just copy over the template launcher binary
COPY --from=template_launcher /opt/google/dataflow/python_template_launcher /opt/google/dataflow/python_template_launcher

# And the beam runner
ARG RUNNER=beam_runner.py

RUN mkdir /template
COPY ${RUNNER} /template/

ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/template/${RUNNER}"

ENTRYPOINT ["/opt/google/dataflow/python_template_launcher"]
